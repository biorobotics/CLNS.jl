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

using Base.Threads
using IterTools
include("dag_dfs.jl")

"""
Select a removal and an insertion method using powers, and then perform
removal followed by insertion on tour.  Operation done in place.
"""
function remove_insert(current::Tour, 
                       sets::Vector{Function}, cost_fn::Function, 
                       powers, param::Dict{Symbol,Any}, phase::Symbol, powers_lock::ReentrantLock, current_lock::ReentrantLock)
	# make a new tour to perform the insertion and deletion on
  trial = Tour(zeros(0, 0), Vector{Int64}(), Vector{Float64}(), 0.)
  @lock current_lock trial = tour_copy(current)
	pivot_tour!(trial)
	num_removals = rand(param[:min_removals]:param[:max_removals])

  removal_idx = 0
  insertion_idx = 0
  noise_idx = 0
  # If we're only running one cold trial, we don't
  # update the heuristic weights which means we don't need
  # to lock
  if param[:cold_trials] != 1
    lock(powers_lock)
    try
      removal_idx = power_select(powers["removals"], powers["removal_total"], phase)
      insertion_idx = power_select(powers["insertions"], powers["insertion_total"], phase)
      noise_idx = power_select(powers["noise"], powers["noise_total"], phase)
    finally
      unlock(powers_lock)
    end
  else
    removal_idx = power_select(powers["removals"], powers["removal_total"], phase)
    insertion_idx = power_select(powers["insertions"], powers["insertion_total"], phase)
    noise_idx = power_select(powers["noise"], powers["noise_total"], phase)
  end
  removal = powers["removals"][removal_idx]
  insertion = powers["insertions"][insertion_idx]
  noise = powers["noise"][noise_idx]
	if removal.name == "distance"
		sets_to_insert, removed_points = distance_removal!(trial, cost_fn, num_removals,
                                                       removal.value)
  elseif removal.name == "worst"
		sets_to_insert, removed_points = worst_removal!(trial, cost_fn, num_removals,
                                                   removal.value)
	else
		sets_to_insert, removed_points = segment_removal!(trial, num_removals, cost_fn)
	end

  # Compute costs between sampled sets and tour points
  num_samples = 5
  samples_per_set = [cat(removed_points[set_idx]', set(num_samples), dims=1) for (set_idx, set) in enumerate(sets[sets_to_insert])]
  point_dim = size(samples_per_set[1], 2)
  costs_forward, costs_backward = cost_fn(cat(samples_per_set..., dims=1), trial.tour)
  cost_dict = CostDict(Dict(), Dict(), Dict(), Dict()) 
  num_points_so_far = 0
  for set_idx=1:length(sets_to_insert)
    set = sets_to_insert[set_idx]
    for (point_idx1, point_idx2)=IterTools.product(1:size(samples_per_set[set_idx], 1), 1:length(trial))
      # Sets we're inserting should not be visited in the tour
      @assert(trial.set_seq[point_idx2] != set)

      cost_dict.set_point_to_tour_point[set, point_idx1, point_idx2] = costs_forward[num_points_so_far + point_idx1, point_idx2]
      if haskey(cost_dict.set_to_tour_point, (set, point_idx2))
        cost_dict.set_to_tour_point[set, point_idx2] = min(cost_dict.set_to_tour_point[set, point_idx2], cost_dict.set_point_to_tour_point[set, point_idx1, point_idx2])
      else
        cost_dict.set_to_tour_point[set, point_idx2] = cost_dict.set_point_to_tour_point[set, point_idx1, point_idx2]
      end

      cost_dict.tour_point_to_set_point[point_idx2, set, point_idx1] = costs_backward[point_idx2, num_points_so_far + point_idx1]
      if haskey(cost_dict.tour_point_to_set, (point_idx2, set))
        cost_dict.tour_point_to_set[point_idx2, set] = min(cost_dict.tour_point_to_set[point_idx2, set], cost_dict.tour_point_to_set_point[point_idx2, set, point_idx1])
      else
        cost_dict.tour_point_to_set[point_idx2, set] = cost_dict.tour_point_to_set_point[point_idx2, set, point_idx1]
      end
    end
    num_points_so_far += size(samples_per_set[set_idx], 1)
  end

	# then perform insertion
	if insertion.name == "cheapest"
		cheapest_insertion!(trial, sets_to_insert, samples_per_set, cost_dict, cost_fn)
	else
		randpdf_insertion!(trial, sets_to_insert, samples_per_set, cost_dict, cost_fn,
                       insertion.value, noise)
	end
  # TODO: implement opt_cycle
	# rand() < param[:prob_reopt] && opt_cycle!(trial, samples_per_set, cost_dict, param, "partial")

	# update power scores for remove and insert
  # not needed if we're only doing 1 cold trial
  if param[:cold_trials] != 1
    score = 0.
    @lock current_lock score = 100 * max(current.cost - trial.cost, 0)/current.cost
    lock(powers_lock)
    try
      insertion.scores[phase] += score
      insertion.count[phase] += 1
      removal.scores[phase] += score
      removal.count[phase] += 1
      noise.scores[phase] += score
      noise.count[phase] += 1
    finally
      unlock(powers_lock)
    end
  end
	return trial
end


"""
Select an integer between 1 and num according to
and exponential distribution with lambda = power
# goes from left of array if power is positive
# and right of array if it is negative
"""
function select_k(num::Int64, power::Float64)
	base = (1/2)^abs(power)
	# (1 - base^num)/(1 - base) is sum of geometric series
	rand_select = (1 - base^num)/(1 - base) * rand()
	bin = 1
	@inbounds for k = 1:num
		if rand_select < bin
			return (power >= 0 ? (num - k + 1) : k)
		end
		rand_select -= bin
		bin *= base
	end
	return (power >=0 ? num : 1)
end


"""
selecting a random k in 1 to length(weights) according to power
and then selecting the kth smallest element in weights
"""
function pdf_select(weights::Vector, power::Float64)
    power == 0.0 && return rand(1:length(weights))
    power > 9.0 && return rand_select(weights, maximum(weights))
    power < - 9.0 && return rand_select(weights, minimum(weights))

	# select kth smallest.  If 1 or length(weights), simply return
	k = select_k(length(weights), power)
	k == 1 && return rand_select(weights, minimum(weights))
	k == length(weights) && return rand_select(weights, maximum(weights))
	val = partialsort(weights, k)

	return rand_select(weights, val)
end


"""  choose set with pdf_select, and then insert in best place with noise  """
function randpdf_insertion!(tour::Tour, sets_to_insert::Array{Int64,1},
                            samples_per_set::Vector{Matrix{Float64}}, cost_dict::CostDict, cost_fn::Function, power::Float64, noise::Power)

    mindist = [typemax(Float64) for i=1:length(sets_to_insert)]
    @inbounds for i = 1:length(sets_to_insert)
      set = sets_to_insert[i]
      for tour_idx=1:length(tour)
        if cost_dict.set_to_tour_point[set, tour_idx] < mindist[i]
          mindist[i] = cost_dict.set_to_tour_point[set, tour_idx]
        end
      end
    end
    best_tour_idx = -1

    @inbounds while length(sets_to_insert) > 0
      if best_tour_idx != -1
        for i = 1:length(sets_to_insert)
          set = sets_to_insert[i]
          if cost_dict.set_to_tour_point[set, best_tour_idx] < mindist[i]
            mindist[i] = cost_dict.set_to_tour_point[set, best_tour_idx]
          end
        end
      end
      set_index = pdf_select(mindist, power) # select set to insert from pdf
      # find the closest vertex and the best insertion in that vertex
      nearest_set = sets_to_insert[set_index]
      if noise.name == "subset"
        best_set_point_idx, best_tour_idx = insert_subset_lb(tour, size(samples_per_set[set_index], 1), nearest_set,
                                                             noise.value, cost_dict)
        @assert(best_tour_idx > 0)
      else
        best_set_point_idx, best_tour_idx =
            insert_lb(tour, size(samples_per_set[set_index], 1), nearest_set, noise.value, cost_dict)
        @assert(best_tour_idx > 0)
      end
      prev_tour_idx = prev_tour(tour, best_tour_idx)
      insert!(tour, best_tour_idx, samples_per_set[set_index][best_set_point_idx, :], sets_to_insert[set_index], cost_dict.tour_point_to_set_point[prev_tour_idx, nearest_set, best_set_point_idx], cost_dict.set_point_to_tour_point[nearest_set, best_set_point_idx, best_tour_idx])

      # remove the inserted set from data structures
      splice!(sets_to_insert, set_index)
      splice!(samples_per_set, set_index)
      splice!(mindist, set_index)
      if length(samples_per_set) != 0
        update_cost_dict!(cost_dict, cost_fn, samples_per_set, best_tour_idx, tour[best_tour_idx], sets_to_insert)
      end
    end
end


function cheapest_insertion!(tour::Tour, sets_to_insert::Array{Int64,1},
                             samples_per_set::Vector{Matrix{Float64}}, cost_dict::CostDict, cost_fn::Function)
    """
	choose vertex that can be inserted most cheaply, and insert it in that position
	"""
	while length(sets_to_insert) > 0
    best_cost = typemax(Float64)
    best_set_point_idx = 0
    best_tour_idx = 0
    best_set = 0
    for i = 1:length(sets_to_insert)
      set_ind = sets_to_insert[i]
      # find the best place to insert the vertex
      best_set_point_idx, best_tour_idx, cost = insert_cost_lb(tour, cost_dict, size(samples_per_set[i], 1), set_ind,
                                                  best_set_point_idx, best_tour_idx, best_cost)
      if cost < best_cost
        best_set = i
        best_cost = cost
      end
    end

    # now, perform the insertion
    prev_tour_idx = prev_tour(tour, best_tour_idx)
    insert!(tour, best_tour_idx, samples_per_set[best_set][best_set_point_idx, :], sets_to_insert[best_set], cost_dict.tour_point_to_set_point[prev_tour_idx, sets_to_insert[best_set], best_set_point_idx], cost_dict.set_point_to_tour_point[sets_to_insert[best_set], best_set_point_idx, best_tour_idx])
    # remove the inserted set from data structures
    splice!(sets_to_insert, set_index)
    splice!(samples_per_set, set_index)
    if length(samples_per_set) != 0
      update_cost_dict!(cost_dict, cost_fn, samples_per_set, best_tour_idx, tour[best_tour_idx], sets_to_insert)
    end
  end
end


"""
Given a tour and a set, this function finds the vertex in the set with minimum
insertion cost, along with the position of this insertion in the tour.  If
best_position is i, then vertex should be inserted between tour[i-1] and tour[i].
"""
@inline function insert_lb(tour::Tour, num_samples_for_set::Int,
                           setind::Int, noise::Float64, cost_dict::CostDict)
	best_cost = typemax(Float64)
	best_set_point_idx = 0
	best_tour_idx = 0

	@inbounds for i = 1:length(tour)
		prev_i = prev_tour(tour, i)
		lb = cost_dict.tour_point_to_set[prev_i, setind] + cost_dict.set_to_tour_point[setind, i] - tour.cost_seq[prev_i]
		lb > best_cost && continue

    for set_point_idx=1:num_samples_for_set
      insert_cost = cost_dict.tour_point_to_set_point[prev_i, setind, set_point_idx] + cost_dict.set_point_to_tour_point[setind, set_point_idx, i] - tour.cost_seq[prev_i]
      noise > 0.0 && (insert_cost += noise * rand() * abs(insert_cost))
      if insert_cost < best_cost
        best_cost = insert_cost
        best_set_point_idx = set_point_idx
        best_tour_idx = i
      end
    end
  end
  return best_set_point_idx, best_tour_idx
end


@inline function insert_subset_lb(tour::Tour, num_samples_for_set::Int,
                                  setind::Int, noise::Float64, cost_dict::CostDict)
	best_cost = typemax(Float64)
	best_set_point_idx = 0
	best_tour_idx = 0
	tour_inds = collect(1:length(tour))

	@inbounds for i = 1:ceil(Int64, length(tour) * noise)
		i = incremental_shuffle!(tour_inds, i)
		prev_i = prev_tour(tour, i)
		lb = cost_dict.tour_point_to_set[prev_i, setind] + cost_dict.set_to_tour_point[setind, i] - tour.cost_seq[prev_i]
		lb > best_cost && continue

		for set_point_idx=1:num_samples_for_set
      insert_cost = cost_dict.tour_point_to_set_point[prev_i, setind, set_point_idx] + cost_dict.set_point_to_tour_point[setind, set_point_idx, i] - tour.cost_seq[prev_i]
      # @assert(!isnan(insert_cost))
      if insert_cost < best_cost
				best_cost = insert_cost
				best_set_point_idx = set_point_idx
				best_tour_idx = i
      end
		end
  end
  return best_set_point_idx, best_tour_idx
end


############ Initial Tour Construction ##########################

"""build tour from scratch on a cold restart"""
function initial_tour!(lowest::Tour, sets::Vector{Function}, cost_fn::Function,
                       trial_num::Int64, param::Dict{Symbol,Any}, stop_time::Float64, inf_val::Float64)
	sets_to_insert = collect(1:param[:num_sets])
	best = Tour(zeros(0, 0), Int64[], Float64[], typemax(Float64))

	if true # param[:init_tour] == "dag_dfs"
    sample_then_dag_dfs!(best, sets, cost_fn, stop_time, inf_val)
	elseif param[:init_tour] == "rand" && (trial_num > 1) && (rand() < 0.5)
		random_initial_tour!(best.tour, sets, cost_fn)
	else
		random_insertion!(best.tour, sets, cost_fn)
	end
  # I think the original GLNS code had a bug here, since it set lowest = best, even though this does not change lowest upon returning.
  # On the other hand, I don't think this bug affected algorithm behavior
	if lowest.cost > best.cost
    lowest.cost = best.cost
    lowest.tour = best.tour
    lowest.set_seq = best.set_seq
  end
	return best
end


"""
TODO: implement if needed. This is still code from GLNS
"""
#=
function random_insertion!(tour::Array{Int64,1}, sets_to_insert::Array{Int64,1},
                           dist, sets::Vector{Vector{Int64}}, setdist::Distsv)
    shuffle!(sets_to_insert)  # randomly permute the sets
    for set in sets_to_insert
        # only have to compute the insert cost for the changed portion of the tour
        if isempty(tour)
            best_vertex = rand(sets[set])
            best_position = 1
        else
            best_vertex, best_position = insert_lb(tour, dist, sets[set], set, setdist, 0.75, ReentrantLock())
        end
        # now, perform the insertion
        insert!(tour, best_position, best_vertex)
    end
end
=#


"""
TODO: implement if needed. This is still code from GLNS
"""
#=
function random_initial_tour!(tour::Array{Int64,1}, sets_to_insert::Array{Int64,1},
							  dist::Array{Int64, 2}, sets::Vector{Vector{Int64}})
    shuffle!(sets_to_insert)
    for set in sets_to_insert
		push!(tour, rand(sets[set]))
    end
end
=#


######################### Removals ################################
"""
Remove the vertices randomly, but biased towards those that add the most length to the
tour.  Bias is based on the power input.  Vertices are then selected via pdf select.
"""
function worst_removal!(tour::Tour, cost_fn::Function,
                        num_to_remove::Int64, power::Float64)
  deleted_sets = Array{Int}(undef, 0)
  deleted_points = Vector{Vector{Float64}}()
	while length(deleted_sets) < num_to_remove
		removal_costs = worst_vertices(tour, cost_fn)
		ind = pdf_select(removal_costs, power)
		set_to_delete = tour.set_seq[ind]

    # perform the deletion
    push!(deleted_sets, set_to_delete)

    prev_idx = prev_tour(tour, ind)
    next_idx = next_tour(tour, ind)
    deleted_point = splice!(tour, ind, cost_fn(tour[prev_idx], tour[next_idx]))
    push!(deleted_points, deleted_point)
	end
  return deleted_sets, deleted_points
end


""" removing a single continuos segment of the tour of size num_remove """
function segment_removal!(tour::Tour, num_to_remove::Int64, cost_fn::Function)
	i = rand(1:length(tour))
	deleted_sets = Array{Int}(undef, 0)
  deleted_points = Vector{Vector{Float64}}()
	while length(deleted_sets) < num_to_remove
		i > length(tour) && (i = 1)
		push!(deleted_sets, tour.set_seq[i])
    prev_idx = prev_tour(tour, i)
    next_idx = next_tour(tour, i)
    deleted_point = splice!(tour, i, cost_fn(tour[prev_idx], tour[next_idx]))
    push!(deleted_points, deleted_point)
	end
	return deleted_sets, deleted_points
end


"""  pick a random vertex, and delete its closest neighbors  """
function distance_removal!(tour::Tour, cost_fn::Function,
                           num_to_remove::Int64, power::Float64)
    deleted_sets = Array{Int}(undef, 0)

    deleted_points = Vector{Vector{Float64}}()

    seed_index = rand(1:length(tour))
    push!(deleted_sets, tour.set_seq[seed_index])
    prev_idx = prev_tour(tour, seed_index)
    next_idx = next_tour(tour, seed_index)
    deleted_point = splice!(tour, seed_index, cost_fn(tour[prev_idx], tour[next_idx]))
    push!(deleted_points, deleted_point)

    while length(deleted_sets) < num_to_remove
        # pick a random point from the set of deleted vertices
        seed_point = rand(deleted_points)
        # find closest point to the seed point that's still in the tour
        mindist = zeros(Float64, length(tour))
        for i = 1:length(tour)
          # TODO: should be able to reuse cost computation here
          mindist[i] = min(cost_fn(seed_point, tour[i]), cost_fn(tour[i], seed_point))
        end
        del_index = pdf_select(mindist, power)
        push!(deleted_sets, tour.set_seq[del_index])
        prev_idx = prev_tour(tour, del_index)
        next_idx = next_tour(tour, del_index)
        deleted_point = splice!(tour, del_index, cost_fn(tour[prev_idx], tour[next_idx]))
        push!(deleted_points, deleted_point)
    end

    return deleted_sets, deleted_points
end


"""
determine the cost of removing each vertex from the tour, given that all others remain.
"""
# TODO: save the costs we compute here?
function worst_vertices(tour::Tour, cost_fn::Function)
    removal_cost = zeros(Float64, length(tour))
    @inbounds for i = 1:length(tour)
      if i == 1
          cost1 = tour.cost_seq[end]
          cost2 = tour.cost_seq[i]
          cost3 = cost_fn(tour[end], tour[i+1])
      elseif i == length(tour)
          cost1 = tour.cost_seq[i - 1]
          cost2 = tour.cost_seq[i]
          cost3 = cost_fn(tour[i - 1], tour[1])
      else
          cost1 = tour.cost_seq[i - 1]
          cost2 = tour.cost_seq[i]
          cost3 = cost_fn(tour[i - 1], tour[i + 1])
      end

      #=
      if (isinf(cost1) || isinf(cost2)) && !isinf(cost3)
        removal_cost[i] = -Inf
      elseif !(isinf(cost1) || isinf(cost2)) && isinf(cost3)
        removal_cost[i] = Inf
        # This shouldn't happen if triangle inequality holds, unless we're removing the depot
        # println(tour[i])
      elseif (isinf(cost1) || isinf(cost2)) && isinf(cost3)
        removal_cost[i] = 0.
      else
        removal_cost[i] = cost1 + cost2 - cost3
      end
      =#
      removal_cost[i] = cost1 + cost2 - cost3
    end
    # @assert(all(.!isnan.(removal_cost)))
    return removal_cost
end
