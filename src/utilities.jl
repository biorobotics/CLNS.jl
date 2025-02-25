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


#####################################################
#########  GTSP Utilities ###########################

using Base.Threads
import Base: getindex, splice!, length, insert!, lastindex

""" tour type that stores the order array and the length of the tour
"""
mutable struct Tour
	tour::Matrix{Float64}
  set_seq::Vector{Int64}
  cost_seq::Vector{Float64} # cost_seq[i] is the travel cost from point i to point i + 1
	cost::Float64
end

function tour_copy(tour)
  return Tour(copy(tour.tour), copy(tour.set_seq), copy(tour.cost_seq), tour.cost)
end

function length(tour)
  return length(tour.set_seq)
end

function lastindex(tour)
  return lastindex(tour.set_seq)
end

""" return the vertex before tour[i] on tour """
@inline function prev_tour(tour, i)
	i != 1 && return i - 1
	return length(tour.set_seq)
end

@inline function next_tour(tour, i)
	i != length(tour.set_seq) && return i + 1
	return 1
end

function getindex(tour::Tour, i::Int64)
  return tour.tour[i, :]
end

function splice!(tour::Tour, i::Int64, cost::Float64)
  prev_i = prev_tour(tour, i)
  deleted_point = tour[i]
  tour.cost = tour.cost - tour.cost_seq[prev_i] - tour.cost_seq[i] + cost
  tour.tour = cat(tour.tour[1:i-1, :], tour.tour[i + 1:end, :], dims=1)
  splice!(tour.set_seq, i)
  tour.cost_seq[prev_i] = cost
  splice!(tour.cost_seq, i)
  return deleted_point
end

function insert!(tour::Tour, i::Int64, point::Vector{Float64}, set::Int64, pre_cost::Float64, post_cost::Float64)
  prev_i = prev_tour(tour, i)
  tour.cost = tour.cost - tour.cost_seq[prev_i] + pre_cost + post_cost
  tour.tour = cat(tour.tour[1:i-1, :], point', tour.tour[i:end, :], dims=1) # perform the insertion
  insert!(tour.set_seq, i, set)
  tour.cost_seq[prev_i] = pre_cost
  insert!(tour.cost_seq, i, post_cost)
end

# function length(m::Matrix)
#   return size(m, 1)
# end

mutable struct CostDict
  set_to_tour_point::Dict
  tour_point_to_set::Dict

  set_point_to_tour_point::Dict
  tour_point_to_set_point::Dict
end

function update_cost_dict!(cost_dict::CostDict, cost_fn::Function, samples_per_set::Vector{Matrix{Float64}}, insert_tour_idx::Int64, insert_point::Vector{Float64})
  tmp_cost_dict = deepcopy(cost_dict)
  cost_dict.set_to_tour_point = Dict()
  cost_dict.tour_point_to_set = Dict()
  cost_dict.set_point_to_tour_point = Dict()
  cost_dict.tour_point_to_set_point = Dict()

  for (key,value) in tmp_cost_dict.set_to_tour_point
    set_idx = key[1]
    old_tour_idx = key[2]
    if old_tour_idx >= insert_tour_idx
      cost_dict.set_to_tour_point[set_idx, old_tour_idx + 1] = value
    else
      cost_dict.set_to_tour_point[key] = value
    end
  end

  for (key,value) in tmp_cost_dict.tour_point_to_set
    old_tour_idx = key[1]
    set_idx = key[2]
    if old_tour_idx >= insert_tour_idx
      cost_dict.tour_point_to_set[old_tour_idx + 1, set_idx] = value
    else
      cost_dict.tour_point_to_set[key] = value
    end
  end

  for (key,value) in tmp_cost_dict.set_point_to_tour_point
    set_idx = key[1]
    set_point_idx = key[2]
    old_tour_idx = key[3]
    if old_tour_idx >= insert_tour_idx
      cost_dict.set_point_to_tour_point[set_idx, set_point_idx, old_tour_idx + 1] = value
    else
      cost_dict.set_point_to_tour_point[key] = value
    end
  end

  for (key,value) in tmp_cost_dict.tour_point_to_set_point
    old_tour_idx = key[1]
    set_idx = key[2]
    set_point_idx = key[3]
    if old_tour_idx >= insert_tour_idx
      cost_dict.tour_point_to_set_point[old_tour_idx + 1, set_idx, set_point_idx] = value
    else
      cost_dict.tour_point_to_set_point[key] = value
    end
  end

  costs_forward, costs_backward = cost_fn(cat(samples_per_set..., dims=1), Matrix(insert_point'))
  num_points_so_far = 0
  for set_idx=1:sets_to_insert
    for point_idx1=1:length(samples_per_set[set_idx])
      cost_dict.set_point_to_tour_point[set, point_idx1, insert_tour_idx] = costs_forward[num_points_so_far + point_idx1, 1]
      if haskey(cost_dict.set_to_tour_point, (set, insert_tour_idx))
        cost_dict.set_to_tour_point[set, insert_tour_idx] = min(cost_dict.set_to_tour_point[set, insert_tour_idx], cost_dict.set_point_to_tour_point[set, point_idx1, insert_tour_idx])
      else
        cost_dict.set_to_tour_point[set, insert_tour_idx] = cost_dict.set_point_to_tour_point[set, point_idx1, insert_tour_idx]
      end
      point_pair_idx += 1

      cost_dict.tour_point_to_set_point[insert_tour_idx, set, point_idx1] = costs_backward[1, num_points_so_far + point_idx1]
      if haskey(cost_dict.set_to_tour_point, (insert_tour_idx, set))
        cost_dict.tour_point_to_set[insert_tour_idx, set] = min(cost_dict.set_to_tour_point[insert_tour_idx, set], cost_dict.tour_point_to_set_point[insert_tour_idx, set, point_idx1])
      else
        cost_dict.tour_point_to_set[insert_tour_idx, set] = cost_dict.tour_point_to_set_point[insert_tour_idx, set, point_idx1]
      end
    end
    num_points_so_far += size(samples_per_set[set_idx], 1)
  end
end

######################################################
#############  Randomizing tour before insertions ####

""" some insertions break tie by taking first minimizer -- this
randomization helps avoid getting stuck choosing same minimizer """
function pivot_tour!(tour::Tour)
	pivot = rand(1:length(tour.set_seq))
	tour.tour = cat(tour.tour[pivot:end, :], tour.tour[1:pivot-1, :], dims=1)
	tour.set_seq = cat(tour.set_seq[pivot:end], tour.set_seq[1:pivot-1], dims=1)
	tour.cost_seq = cat(tour.cost_seq[pivot:end], tour.cost_seq[1:pivot-1], dims=1)
end


############################################################
############ Trial acceptance criteria #####################

"""
decide whether or not to accept a trial based on simulated annealing criteria
"""
function accepttrial(trial_cost::Float64, current_cost::Float64, temperature::Float64)
    if trial_cost <= current_cost
        accept_prob = 2.0
	else
        accept_prob = exp((current_cost - trial_cost)/temperature)
    end
    return (rand() < accept_prob ? true : false)
end


"""
decide whether or not to accept a trial based on simple probability
"""
function accepttrial_noparam(trial_cost::Float64, current_cost::Float64, prob_accept::Float64)
    if trial_cost <= current_cost
        return true
	end
	return (rand() < prob_accept ? true : false)
end


#####################################################
#############  Incremental Shuffle ##################

@inline function incremental_shuffle!(a::AbstractVector, i::Int)
    j = i + floor(Int, rand() * (length(a) + 1 - i))
   	a[j], a[i] = a[i], a[j]
	return a[i]
end


""" rand_select for randomize over all minimizers """
@inline function rand_select(a::Vector{T}, val::T) where {T}
	inds = Int[]
	@inbounds for i=1:length(a)
		a[i] == val && (push!(inds, i))
	end
	return rand(inds)
end
