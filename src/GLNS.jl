# Copyright 2017 Stephen L. Smith and Frank Imeson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
module GLNS
export solver
using Random
using Sockets
using Printf
using NPZ
using Polyester: @batch
using Base.Threads
using ThreadPinning
import Future
include("utilities.jl")
include("parse_print.jl")
include("tour_optimizations.jl")
include("adaptive_powers.jl")
include("insertion_deletion.jl")
include("parameter_defaults.jl")

function solver(problem_instance::String, start_time_for_tour_history::UInt64, sets::Vector{Function}, cost_fn::Function, max_threads::Int64, inf_val::Float64; args...)
  Random.seed!(1234)

  pinthreads(:cores)

	param = parameter_settings(length(sets), problem_instance, args)

	#####################################################
	init_time = time()

	count = Dict(:latest_improvement => 1,
	  			 :first_improvement => false,
	 		     :warm_trial => 0,
	  		     :cold_trial => 1,
				 :total_iter => 0,
				 :print_time => init_time)
	lowest = Tour(zeros(0, 0), Int64[], Float64[], typemax(Float64))

	start_time = time_ns()
	powers = initialize_powers(param)
  
  tour_history = Array{Tuple{Float64, Float64},1}()

  nthreads = min(Threads.nthreads(), max_threads)

  powers_lock = ReentrantLock()
  best_lock = ReentrantLock()
  current_lock = ReentrantLock()

  phase_lock = ReentrantLock()

  count_lock = ReentrantLock()

  iter_count_lock = ReentrantLock()

  temperature_lock = ReentrantLock()

	while true
    if count[:cold_trial] > param[:cold_trials] && lowest.cost <= param[:budget]
      break
    end

    best = initial_tour!(lowest, sets, cost_fn, count[:cold_trial], param, init_time + param[:max_time], inf_val)
    timer = (time_ns() - start_time)/1.0e9
		# print_cold_trial(count, param, best)
		phase = :early

		if count[:cold_trial] == 1
			powers = initialize_powers(param)
		else
			power_update!(powers, param)
		end

    while count[:warm_trial] <= param[:warm_trials]
      iter_count = 1
      current = tour_copy(best)
      temperature = 1.442 * param[:accept_percentage] * best.cost
      # accept a solution with 50% higher cost with 0.05% change after num_iterations.
      cooling_rate = ((0.0005 * lowest.cost)/(param[:accept_percentage] *
                  current.cost))^(1/param[:num_iterations])

      if count[:warm_trial] > 0	  # if warm restart, then use lower temperature
        temperature *= cooling_rate^(param[:num_iterations]/2)
        phase = :late
      end
      this_phase = phase
      # @threads for thread_idx=1:nthreads
      for thread_idx=1:1
        while true
          if @lock count_lock (count[:latest_improvement] > (count[:first_improvement] ?
                                                             param[:latest_improvement] : param[:first_improvement]))
            break
          end

          this_iter_count = 0
          @lock iter_count_lock this_iter_count = iter_count

          lock(phase_lock)
          try
            if this_iter_count > param[:num_iterations]/2 && phase == :early
              phase = :mid  # move to mid phase after half iterations
            end
            this_phase = phase
          finally
            unlock(phase_lock)
          end
          trial = remove_insert(current, sets, cost_fn, powers, param, this_phase, powers_lock, current_lock)

          # decide whether or not to accept trial
          this_temperature = 0.
          @lock temperature_lock this_temperature = temperature

          lock(current_lock)
          try
            if accepttrial_noparam(trial.cost, current.cost, param[:prob_accept]) ||
               accepttrial(trial.cost, current.cost, this_temperature)
              @assert(param[:mode] != "slow") # I don't want to perform an opt cycle while something is locked
              # TODO: implement opt_cycle
              param[:mode] == "slow" && opt_cycle!(current, sets, param, "full") # This seems incorrect. Why are we optimizing current, then setting current = trial?
              current = tour_copy(trial)
            else
              trial = tour_copy(current)
            end
          finally
            unlock(current_lock)
          end

          updated_best = false
          lock(best_lock)
          try
            if trial.cost < best.cost
              updated_best = true
              best = tour_copy(trial)
              println("Thread ", thread_idx, " found new best tour after ", timer, " s with cost ", best.cost, " (before opt cycle)")
            end
          finally
            unlock(best_lock)
          end

          lock(count_lock)
          try
            if updated_best
              count[:latest_improvement] = 1
              count[:first_improvement] = true
              if count[:cold_trial] > 1 && count[:warm_trial] > 1
                count[:warm_trial] = 1
              end
            else
              count[:latest_improvement] += 1
            end
          finally
            unlock(count_lock)
          end

          if updated_best
            # TODO: implement opt_cycle
            # opt_cycle!(trial, sets, param, "full")

            lock(best_lock)
            try
              if trial.cost < best.cost
                best = tour_copy(trial)
                # print_best(count, param, best, lowest, init_time)
                timer = (time_ns() - start_time)/1.0e9
                println("Thread ", thread_idx, " found new best tour after ", timer, " s with cost ", best.cost)

                if param[:output_file] != "None"
                  push!(tour_history, (round((time_ns() - start_time_for_tour_history)/1.0e9, digits=3), best.cost))
                end
              end
            finally
              unlock(best_lock)
            end

            lock(current_lock)
            try
              if trial.cost < current.cost
                current = tour_copy(trial)
              end
            finally
              unlock(current_lock)
            end
          end

          @lock temperature_lock temperature *= cooling_rate  # cool the temperature

          lock(iter_count_lock)
          try
            iter_count += 1
            count[:total_iter] += 1
          finally
            unlock(iter_count_lock)
          end

          if time() - init_time > param[:max_time]
            break
          end
        end
      end
      print_warm_trial(count, param, best, iter_count)
      count[:warm_trial] += 1
      count[:latest_improvement] = 1
      count[:first_improvement] = false

      if time() - init_time > param[:max_time]
        param[:timeout] = true
        break
      end
    end
		lowest.cost > best.cost && (lowest = best)
		count[:warm_trial] = 0
		count[:cold_trial] += 1
    if time() - init_time > param[:max_time]
      break
    end
	end
	timer = (time_ns() - start_time)/1.0e9
  if param[:output_file] != "None"
    push!(tour_history, (round((time_ns() - start_time_for_tour_history)/1.0e9, digits=3), lowest.cost))
  end

  print_summary(lowest, timer, param, tour_history, false)

  return lowest.cost
end

function parse_cmd(ARGS)
	if isempty(ARGS)
		println("no input instance given")
		exit(0)
	end
	if ARGS[1] == "-help" || ARGS[1] == "--help"
		println("Usage:  GTSPcmd.jl [filename] [optional flags]\n")
		println("Optional flags (vales are give in square brackets) :\n")
		println("-mode=[default, fast, slow]      (default is default)")
		println("-max_time=[Int]                  (default set by mode)")
		println("-trials=[Int]                    (default set by mode)")
		println("-restarts=[Int]                  (default set by mode)")
		println("-noise=[None, Both, Subset, Add] (default is Both)")
		println("-num_iterations=[Int]            (default set by mode. Number multiplied by # of sets)")
		println("-verbose=[0, 1, 2, 3]            (default is 3. 0 is no output, 3 is most.)")
		println("-output=[filename]               (default is None)")
		println("-epsilon=[Float in [0,1]]        (default is 0.5)")
		println("-reopt=[Float in [0,1]]          (default is 1.0)")
		println("-budget=[Int]                    (default has no budget)")
		println("-socket_port=[Int]               (default is 65432)")
		println("-lazy_edge_eval=[Int]            (default is 1)")
		println("-new_socket_each_instance=[filename]    (default is 0)")
		exit(0)
	end
	int_flags = ["-max_time", "-trials", "-restarts", "-verbose", "-budget", "-num_iterations", "-socket_port", "-lazy_edge_eval", "-new_socket_each_instance"]
	float_flags = ["-epsilon", "-reopt"]
	string_flags = ["-mode", "-output", "-noise", "-devel"]
	filename = ""
	optional_args = Dict{Symbol, Any}()
	for arg in ARGS
		temp = split(arg, "=")
		if length(temp) == 1 && filename == ""
			filename = temp[1]
		elseif length(temp) == 2
			flag = temp[1]
			value = temp[2]
			if flag in int_flags
				key = Symbol(flag[2:end])
				optional_args[key] = parse(Int64, value)
			elseif flag in float_flags
				key = Symbol(flag[2:end])
				optional_args[key] = parse(Float64, value)
			elseif flag in string_flags
				key = Symbol(flag[2:end])
				optional_args[key] = value
			else
				println("WARNING: skipping unknown flag ", flag, " in command line arguments")
			end
		else
			error("argument ", arg, " not in proper format")
		end
	end
	return filename, optional_args
end

function main(args::Vector{String}, max_time::Float64, max_threads::Int64, instance_parser::Function)
  start_time_for_tour_history = time_ns()
  problem_instance, optional_args = parse_cmd(args)
  problem_instance = String(problem_instance)

  if max_time >= 0
    optional_args[Symbol("max_time")] = max_time
  end

  (sets, cost_fn, inf_val) = instance_parser(problem_instance)

  timing_result = @timed GLNS.solver(problem_instance, start_time_for_tour_history, sets, cost_fn, max_threads, inf_val; optional_args...)
  println(timing_result)
  return timing_result.value
end

end
