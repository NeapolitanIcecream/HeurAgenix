# This file is generated by generate_problem_state.py.
from src.problems.max_cut.components import Solution

import numpy as np

def get_instance_problem_state(instance_data: dict) -> dict:
    """Extract instance problem state from instance data.

    Args:
        instance_data (dict): The dictionary contains the instance data.

    Returns:
        dict: A dictionary with calculated problem_states:
            - average_operation_time (float): The average processing time for all operations across machines.
            - max_operation_time (int): The maximum processing time for any operation on any machine.
            - min_operation_time (int): The minimum non-zero processing time for any operation.
            - total_operation_time (int): The total sum of all operation times.
            - machine_utilization_variance (float): The variance in operation times for each machine.
            - critical_machine (int): The machine with the highest total operation time.
    """
    node_num = instance_data["node_num"]
    weight_matrix = instance_data["weight_matrix"]

    # Ensure the weight matrix is a numpy array
    weight_matrix = np.array(weight_matrix)
    
    # Calculate the number of edges and the sum of edge weights
    nonzero_edges = np.count_nonzero(weight_matrix)
    total_edge_weight = np.sum(weight_matrix)

    # Calculate problem_states
    average_node_degree = np.mean(np.count_nonzero(weight_matrix, axis=0))
    edge_density = nonzero_edges / (node_num * (node_num - 1))
    average_edge_weight = total_edge_weight / nonzero_edges
    max_edge_weight = np.max(weight_matrix)
    min_edge_weight = np.min(weight_matrix[weight_matrix.nonzero()])
    standard_deviation_edge_weight = np.std(weight_matrix[weight_matrix.nonzero()])
    weighted_degree_distribution = np.sum(weight_matrix, axis=0)

    # Construct the feature dictionary
    problem_states = {
        "average_node_degree": average_node_degree,
        "edge_density": edge_density,
        "average_edge_weight": average_edge_weight,
        "max_edge_weight": max_edge_weight,
        "min_edge_weight": min_edge_weight,
        "standard_deviation_edge_weight": standard_deviation_edge_weight,
        "weighted_degree_distribution": weighted_degree_distribution
    }

    return problem_states

import numpy as np

def get_solution_problem_state(instance_data: dict, solution: Solution) -> dict:
    """Extract solution problem state from instance data and solution.

    Args:
        instance_data (dict): The dictionary contains the instance data.
        solution (Solution): The target solution instance.

    Returns:
        dict: A dictionary with calculated problem_states for the current state:
            - set_a_count (int): The number of nodes in set A of the current partition.
            - set_b_count (int): The number of nodes in set B of the current partition.
            - selected_nodes (set[int]): The set of selected nodes.
            - selected_num (int): The number of nodes have been selected.
            - unselected_nodes (set[int]): The set of unselected nodes.
            - unselected_num (int): The number of nodes have not been selected.
            - current_cut_value (int or float): The total weight of edges between set A and set B in the current solution.
            - average_completion_time (float): The average time of job completion.
            - operation_balance (float): The variance in the number of completed operations across jobs.
            - scheduling_efficiency (float): Ratio of completed operations to total possible operations.
            - remaining_jobs (int): Number of jobs still in progress or not started.
    """
    node_num = instance_data["node_num"]
    weight_matrix = instance_data["weight_matrix"]
    current_solution = solution
    set_a_count = len(solution.set_a)
    set_b_count = len(solution.set_b)
    selected_nodes = solution.set_a.union(solution.set_b)
    unselected_nodes = set(range(node_num)) - solution.set_a - solution.set_b

    # Calculate problem states
    current_cut_value = 0
    for node_a in solution.set_a:
        for node_b in solution.set_b:
            current_cut_value += instance_data["weight_matrix"][node_a][node_b]
    imbalance_ratio = abs(set_a_count - set_b_count) / node_num
    average_cut_edge_weight = current_cut_value / len(selected_nodes) if selected_nodes else 0
    selected_nodes_ratio = len(selected_nodes) / node_num
    unselected_nodes_ratio = len(unselected_nodes) / node_num
    internal_edges = [weight_matrix[i][j] for i in current_solution.set_a for j in current_solution.set_a if i != j] + \
                     [weight_matrix[i][j] for i in current_solution.set_b for j in current_solution.set_b if i != j]
    edge_weight_variance_within_sets = np.var(internal_edges) if internal_edges else 0
    
    # Calculate boundary nodes (nodes in selected_nodes that have an edge to unselected_nodes)
    boundary_nodes = len([node for node in selected_nodes if any(neighbor in unselected_nodes for neighbor in np.nonzero(weight_matrix[node])[0])])
    boundary_node_ratio = boundary_nodes / node_num

    # Construct the feature dictionary
    problem_states = {
        "set_a_count": set_a_count,
        "set_b_count": set_b_count,
        "selected_nodes": selected_nodes,
        "selected_num": len(selected_nodes),
        "unselected_nodes": unselected_nodes,
        "unselected_num": len(unselected_nodes),
        "current_cut_value": current_cut_value,
        "imbalance_ratio": imbalance_ratio,
        "average_cut_edge_weight": average_cut_edge_weight,
        "selected_nodes_ratio": selected_nodes_ratio,
        "unselected_nodes_ratio": unselected_nodes_ratio,
        "edge_weight_variance_within_sets": edge_weight_variance_within_sets,
        "boundary_node_ratio": boundary_node_ratio
    }

    return problem_states

def get_observation_problem_state(problem_state: dict) -> dict:
    """Extract core problem state as observation.

    Args:
        problem_state (dict): The dictionary contains the problem state.

    Returns:
        dict: The dictionary contains the core problem state.
    """
    return {
        "selected_num": problem_state["selected_num"]
    }