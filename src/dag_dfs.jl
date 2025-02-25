struct DFSNode
  parent::Vector{DFSNode}
  visited_set_indices::Set{Int64}
  final_node_idx::Int64
  key::Tuple{Set{Int64}, Int64}
end

function DFSNode(parent::Vector{DFSNode}, visited_set_indices::Set{Int64}, final_node_idx::Int64)
  return DFSNode(parent, visited_set_indices, final_node_idx, (visited_set_indices, final_node_idx))
end

function dag_dfs(dist::Array{Float64, 2}, sets::Vector{Vector{Int64}}, membership::Vector{Int64}, stop_time::Float64, inf_val::Float64)
  bt = time_ns()
  closed_list = Set{Tuple{Set{Int64}, Int64}}()
  dfs_stack = [DFSNode(Vector{DFSNode}(), Set(membership[1]), 1)]
  set_indices = Set(1:length(sets))

  # First dimension is node index, second dimension is set index. inf_cost_to_sets[i, j] = true if the cost from node i to all nodes of set j is infinity
  inf_cost_to_sets = stack([mapslices(minimum, dist[:, set], dims=2) for set=sets], dims=2) .== inf_val
  before = [setdiff(Set(findall(inf_cost_to_sets[node_idx, :])), membership[node_idx]) for node_idx=1:size(dist, 1)]

  num_outgoing_edges = sum(dist .!= inf_val, dims=2)

  solved = false
  cost = inf_val
  while length(dfs_stack) != 0
    if time() > stop_time
      println("Timeout during initial tour generation")
      break
    end

    pop = pop!(dfs_stack)
    if pop.key in closed_list
      continue
    end

    push!(closed_list, pop.key)

    if length(pop.visited_set_indices) == length(sets)
      tour = [pop.final_node_idx]
      node_tmp = pop.parent
      while length(node_tmp) != 0
        node = node_tmp[1]
        push!(tour, node.final_node_idx)
        node_tmp = node.parent
      end
      reverse!(tour)
      at = time_ns()
      println("Found initial tour after ", (at - bt)/1.0e9, " s")
      return tour
    else
      unvisited_set_indices = setdiff(set_indices, pop.visited_set_indices)
      neighbors_mask = zeros(Bool, size(dist, 1))
      for set_idx=unvisited_set_indices
        neighbors_mask[sets[set_idx]] .= true
      end
      neighbors_mask = neighbors_mask .& (dist[pop.final_node_idx, :] .!= inf_val)
      neighbors = findall(neighbors_mask)
    end

    if length(neighbors) == 0
      continue
    end

    sort_idx = sortperm(num_outgoing_edges[neighbors])
    for node_idx=neighbors[sort_idx]
      next_unvisited_set_indices = setdiff(unvisited_set_indices, membership[node_idx])
      # Dumas test 2
      if length(intersect(next_unvisited_set_indices, before[node_idx])) != 0
        continue
      end
      neighbor_node = DFSNode([pop], union(pop.visited_set_indices, membership[node_idx]), node_idx)
      if neighbor_node.key in closed_list
        continue
      end
      push!(dfs_stack, neighbor_node)
    end
  end

  at = time_ns()
  println("Failed to generate initial tour after ", (at - bt)/1.0e9, " s")
  return Vector{Int64}()
end

function sample_then_dag_dfs!(tour::Tour, sets::Vector{Function}, cost_fn::Function, stop_time::Float64, inf_val::Float64)
  num_samples = 5
  samples_per_set = [set(num_samples) for set=sets]
  while time() < stop_time
    all_points = cat(samples_per_set..., dims=1)
    cost_mat = cost_fn(all_points)
    discrete_sets = [Vector{Int64}() for set=sets]
    num_nodes = 0
    for set_idx=1:length(samples_per_set)
      discrete_sets[set_idx] = num_nodes .+ collect(1:size(samples_per_set[set_idx], 1))
      num_nodes += length(discrete_sets[set_idx])
    end
    membership = zeros(Int64, num_nodes)
    for (set_idx, discrete_set) in enumerate(discrete_sets)
      for node_idx in discrete_set
        membership[node_idx] = set_idx
      end
    end
    node_seq = dag_dfs(cost_mat, discrete_sets, membership, stop_time, inf_val)
    if length(node_seq) != 0
      @assert(length(node_seq) == length(sets))

      tour.tour = all_points[node_seq, :]
      tour.set_seq = membership[node_seq]
      tour.cost_seq = [cost_mat[node_idx1, node_idx2] for (node_idx1, node_idx2) in zip(node_seq[1:end-1], node_seq[2:end])]
      push!(tour.cost_seq, cost_mat[node_seq[end], node_seq[1]])
      tour.cost = sum(tour.cost_seq)
      println("Generated initial tour with cost ", tour.cost)

      return
    else
      # Don't sample more points for the depot, since there's only one depot
      samples_per_set = [set_idx == 1 ? samples : cat(samples, set(num_samples), dims=1) for (set_idx, (samples,set)) in enumerate(zip(samples_per_set, sets))]
    end
  end
  println("Timed out without generating a tour")
  exit()
end
