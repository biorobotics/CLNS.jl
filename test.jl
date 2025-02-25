import Pkg
Pkg.activate(".")
using NPZ
using GLNS
using Distances

struct SamplerFn
 tw_p0::Matrix{Float64}
 tw_vel::Matrix{Float64}
 tws::Matrix{Float64}
 tws_contiguous::Matrix{Float64}
 tmin_contiguous::Float64
 tmax_contiguous::Float64
end

function SamplerFn(tw_p0::Matrix{Float64}, tw_vel::Matrix{Float64}, tws::Matrix{Float64})
  tws_contiguous = zeros(size(tws)...)
  t_contiguous = 0.
  for tw_idx=1:size(tws, 1)
    tw = tws[tw_idx, :]
    tws_contiguous[tw_idx,:] = [t_contiguous, t_contiguous + (tw[2] - tw[1])]
    t_contiguous += tw[2] - tw[1]
  end
  tmin_contiguous = minimum([tws_contiguous[tw_idx, 1] for tw_idx=1:size(tws, 1)])
  tmax_contiguous = maximum([tws_contiguous[tw_idx, 2] for tw_idx=1:size(tws, 1)])
  return SamplerFn(tw_p0, tw_vel, tws, tws_contiguous, tmin_contiguous, tmax_contiguous)
end

function (f::SamplerFn)(num_samples::Int64)
  pt_dim = 1 + size(f.tw_p0, 2) # Time, then space
  samples = zeros(num_samples, pt_dim)
  for pt_idx in 1:num_samples
    t_contiguous = f.tmin_contiguous + rand()*(f.tmax_contiguous - f.tmin_contiguous)

    found_tw = false
    for tw_idx=1:size(f.tws, 1)
      t0_contiguous = f.tws_contiguous[tw_idx, 1]
      t1_contiguous = f.tws_contiguous[tw_idx, 2]
      if t0_contiguous <= t_contiguous <= t1_contiguous
        t0 = f.tws[tw_idx, 1]
        t = t_contiguous - t0_contiguous + t0
        @assert(t != 0.) # Because we only let the depot have time = 0
        p = f.tw_p0[tw_idx, :] + f.tw_vel[tw_idx, :]*t
        samples[pt_idx, 1] = t
        samples[pt_idx, 2:end] = p
        found_tw = true
        break
      end
    end

    @assert(found_tw)
  end
  return samples
end

struct CostFn
  vmax_agent::Float64
end

function (f::CostFn)(points::Matrix{Float64})
  dists = pairwise(euclidean, points[:, 2:end]')
  travel_times = dists / f.vmax_agent

  req_travel_times = (points[:, 1] .- points[:, 1]')'
  dists[travel_times .> req_travel_times] .= Inf

  # Assume the first point is the depot
  dists[:, 1] .= 0

  return dists
end

function (f::CostFn)(points1::Matrix{Float64}, points2::Matrix{Float64})
  dists = pairwise(euclidean, points1[:, 2:end]', points2[:, 2:end]')
  travel_times = dists / f.vmax_agent

  req_travel_times_forward = (points2[:, 1] .- points1[:, 1]')'
  dists_forward = copy(dists)
  dists_forward[travel_times .> req_travel_times_forward] .= Inf

  # Assume if the time equals zero, the point is the depot
  dists_forward[:, points2[:, 1] .== 0.] .= 0.

  req_travel_times_backward = -req_travel_times_forward'
  dists_backward = copy(dists')
  dists_backward[travel_times' .> req_travel_times_backward] .= Inf

  # Assume if the time equals zero, the point is the depot
  dists_backward[:, points1[:, 1] .== 0.] .= 0.

  return dists_forward, dists_backward
end

function (f::CostFn)(point1::Vector{Float64}, point2::Vector{Float64})
  return f(Matrix(point1'), Matrix(point2'))[1][1, 1]
end

function instance_parser(instance_path::String)
  tw_p0 = npzread(instance_path*"/tw_p0.npy")
  tw_vels = npzread(instance_path*"/tw_vels.npy")
  tws = npzread(instance_path*"/tws.npy")
  tw_to_target_ptr = npzread(instance_path*"/tw_to_target_ptr.npy") .+ 1
  num_targets = maximum(tw_to_target_ptr)
  target_to_tw_ptr = Vector{Vector{Int64}}()
  for target_idx=1:num_targets
    push!(target_to_tw_ptr, findall(tw_to_target_ptr .== target_idx))
  end
  depot_pos = npzread(instance_path*"/depot_pos.npy")
  vmax_agent = npzread(instance_path*"/vmax_agent.npy")

  sets = Vector{Function}()
  depot_set(num_samples) = Matrix(cat(0., depot_pos, dims=1)')
  push!(sets, depot_set)
  for target_idx=1:num_targets
    ptr = target_to_tw_ptr[target_idx]
    fn = SamplerFn(tw_p0[ptr, :], tw_vels[ptr, :], tws[ptr, :])
    wrapped_fn(num_samples) = fn(num_samples)
    push!(sets, wrapped_fn)
  end

  cost_fn = CostFn(vmax_agent)
  wrapped_cost_fn(points1, points2) = cost_fn(points1, points2)
  wrapped_cost_fn(points) = cost_fn(points)

  return sets, wrapped_cost_fn
end

instance_path = expanduser("~/catkin_ws/src/mapf/data/02_24_2025/mt_tsp_clns_test_instances/targ2_win108_random_seed0_occprob0.0_vmaxa5.0_2win_rad0/") # TODO: add an actual instance folder here
ARGS = [instance_path, "-output=custom.tour", "-verbose=3", "-mode=fast"]
sets, cost_fn = instance_parser(instance_path)
# dists_forward, dists_backward = cost_fn(sets[1](1), sets[2](1)[1:1, :])
# display(dists_forward)
# display(dists_backward)
# display(cost_fn(sets[1](1)[1, :], sets[2](1)[1, :]))
# exit()

GLNS.main(ARGS, 10., 1, instance_parser)
GLNS.main(ARGS, 10., 1, instance_parser)
