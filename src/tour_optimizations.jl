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


"""
Sequentially moves each vertex to its best point on the tour.
Repeats until no more moves can be found
"""
#=
# TODO: implement
function moveopt!(tour::Array{Int64, 1}, dist, sets::Vector{Vector{Int64}}, 
				  member, setdist::Distsv)
    improvement_found = true
    number_of_moves = 0
    start_position = 1

    @inbounds while improvement_found && number_of_moves < 10
        improvement_found = false
        for i = start_position:length(tour)
            select_vertex = tour[i]
            delete_cost = removal_cost(tour, dist, i)
            set_ind = member[select_vertex]
            splice!(tour, i)  # remove vertex from tour

            # find the best place to insert the vertex
            v, pos, cost = insert_cost_lb(tour, dist, sets[set_ind], set_ind, setdist, 
										  select_vertex, i, delete_cost)
            insert!(tour, pos, v)
            # check if we found a better position for vertex i
            if cost < delete_cost
                improvement_found = true
                number_of_moves += 1
                start_position = min(pos, i) # start looking for swaps where tour change began
                break
            end
        end
    end
end
=#


#=
# TODO: implement
function moveopt_rand!(tour::Array{Int64, 1}, dist, sets::Vector{Vector{Int64}}, 
				  member, iters::Int, setdist::Distsv)
	tour_inds = collect(1:length(tour))
	@inbounds for i = 1:iters # i = rand(1:length(tour), iters)
		i = incremental_shuffle!(tour_inds, i)
		select_vertex = tour[i]
		
		# first check if this vertex should be moved
		delete_cost = removal_cost(tour, dist, i)
    set_ind = member[select_vertex]
		splice!(tour, i)  # remove vertex from tour
    v, pos, cost = insert_cost_lb(tour, dist, sets[set_ind], set_ind, setdist, 
                    select_vertex, i, delete_cost)
		insert!(tour, pos, v)
  end
end
=#


"""
compute the cost of inserting vertex v into position i of tour
"""
@inline function insert_cost_lb(tour::Tour, cost_dict::CostDict, num_samples_per_set::Int64, setind::Int, 
                                best_set_point_idx::Int, best_tour_idx::Int, best_cost::Int)
  @inbounds for i = 1:length(tour.tour)
		prev_i = prev_tour(tour, i) # first check lower bound
		lb = cost_dict.tour_point_to_set[prev_i, setind] + cost_dict.set_to_tour_point[setind, i] - tour.cost_seq[prev_i]
		lb > best_cost && continue

    for set_point_idx=1:num_samples_per_set
      insert_cost = cost_dict.tour_point_to_set_point[prev_i, setind, set_point_idx] + cost_dict.set_point_to_tour_point[setind, set_point_idx, i] - tour.cost_seq[prev_i]
      if insert_cost < best_cost
        best_cost = insert_cost
        best_set_point_idx = set_point_idx
        best_tour_idx = i
      end
		end
  end
  return best_set_point_idx, best_tour_idx, best_cost
end


"""
determine the cost of removing the vertex at position i in the tour
"""
# TODO: save the costs we compute here?
@inline function removal_cost(tour::Array{Int64, 1}, cost_fn::Function, i::Int64)
    if i == 1
        return tour.cost_seq[end] + tour.cost_seq[i] - cost_fn(tour[end], tour[i+1])
    elseif i == length(tour.tour)
        return tour.cost_seq[i-1] + tour.cost_seq[i] - cost_fn(tour[i-1], tour[1])
    else
        return tour.cost_seq[i-1] + tour.cost_seq[i] - cost_fn(tour[i-1], tour[i+1])
	end
end


""" repeatedly perform moveopt and reopt_tour until there is no improvement """
# TODO: implement
#=
function opt_cycle!(current::Tour, dist, sets::Vector{Vector{Int64}}, 
					member, param::Dict{Symbol, Any}, setdist::Distsv, use)
	current.cost = tour_cost(current.tour, dist)
	prev_cost = current.cost
	for i=1:5
		if i % 2 == 1
      # TODO: add reopt back in?
			# current.tour = reopt_tour(current.tour, dist, sets, member, param)
		elseif param[:mode] == "fast" || use == "partial"
			moveopt_rand!(current.tour, dist, sets, member, param[:max_removals], setdist)
		else
			moveopt!(current.tour, dist, sets, member, setdist)
		end
		current.cost = sum(tour.cost_seq)
		if i > 1 && (current.cost >= prev_cost || use == "partial")
			return
		end
		prev_cost = current.cost
	end	
end
=#


"""
Given an ordering of the sets, this alg performs BFS to find the 
optimal vertex in each set
"""
# TODO: implement
#=
function reopt_tour(tour::Array{Int64,1}, dist, sets::Vector{Vector{Int64}}, 
					member, param::Dict{Symbol, Any})
    best_tour_cost = tour_cost(tour, dist)
	new_tour = copy(tour)
	min_index = min_setv(tour, sets, member, param)	
    tour = [tour[min_index:end]; tour[1:min_index-1]]

    prev = zeros(Int64, param[:num_vertices])   # initialize cost_to_come
    cost_to_come = zeros(Int64, param[:num_vertices])
    @inbounds for start_vertex in sets[member[tour[1]]]
 		relax_in!(cost_to_come, dist, prev, Int64[start_vertex], sets[member[tour[2]]])
        for i = 3:length(tour)  # cost to get to ith set on path through (i-1)th set
            relax_in!(cost_to_come, dist, prev, sets[member[tour[i-1]]], sets[member[tour[i]]])
        end
        # find the cost back to the start vertex.
        tour_cost, start_prev = relax(cost_to_come, dist, sets[member[tour[end]]], start_vertex)
        if tour_cost < best_tour_cost   # reconstruct the path
			best_tour_cost = tour_cost
            new_tour = extract_tour(prev, start_vertex, start_prev)
        end
    end
	return new_tour
end
=#


""" Find the set with the smallest number of vertices """
# TODO: maybe delete, maybe use
#=
function min_setv(tour::Array{Int64, 1}, sets::Vector{Vector{Int64}}, member, 
				  param::Dict{Symbol, Any})
	min_set = param[:min_set]
	@inbounds for i = 1:length(tour)
		member[tour[i]] == min_set && return i
	end
	return 1
end
=#


"""
extracting a tour from the prev pointers.
"""
# TODO: maybe delete, maybe use
#=
function extract_tour(prev::Array{Int64,1}, start_vertex::Int64, start_prev::Int64)
    tour = [start_vertex]
    vertex_step = start_prev
    while prev[vertex_step] != 0
        push!(tour, vertex_step)
        vertex_step = prev[vertex_step]
    end
    return reverse(tour)
end
=#


"""
outputs the new cost and prev for vertex v2 after relaxing
does not actually update the cost
"""
#=
@inline function relax(cost::Array{Int64, 1}, dist, set1::Array{Int64, 1}, v2::Int64)
	v1 = set1[1]
	min_cost = cost[v1] + dist[v1, v2]
	min_prev = v1
    @inbounds for i = 2:length(set1)
		v1 = set1[i]
		newcost = cost[v1] + dist[v1, v2]
        if min_cost > newcost
            min_cost, min_prev = newcost, v1
        end
    end
    return min_cost, min_prev
end
=#


"""
relaxes the cost of each vertex in the set set2 in-place.
"""
#=
@inline function relax_in!(cost::Array{Int64, 1}, dist, prev::Array{Int64, 1}, 
				 set1::Array{Int64, 1}, set2::Array{Int64, 1})
	@inbounds for v2 in set2
		v1 = set1[1]
		cost[v2] = cost[v1] + dist[v1, v2]
		prev[v2] = v1
        for i = 2:length(set1)
			v1 = set1[i]
			newcost = cost[v1] + dist[v1, v2]
            if cost[v2] > newcost
				cost[v2], prev[v2] = newcost, v1
            end
        end
    end
end
=#
