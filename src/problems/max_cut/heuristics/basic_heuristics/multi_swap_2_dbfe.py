from src.problems.max_cut.components import *

def multi_swap_2_dbfe(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[SwapOperator, dict]:
    """
    This heuristic evaluates all possible pairs of nodes for swapping between set A and set B.
    It performs the swap that leads to the highest increase in the cut value, considering the combined effect
    of swapping two nodes simultaneously rather than one at a time. It utilizes numpy operations for efficient computation.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "weight_matrix" (numpy.ndarray): A 2D array representing the weight between nodes.
            - "current_solution" (Solution): The current solution of the Max Cut problem.
        algorithm_data (dict): Not used in this heuristic.

    Returns:
        SwapOperator: The operator that swaps a pair of nodes between sets to improve the cut value.
        dict: Empty dictionary as no algorithm data is updated.
    """

    current_solution = problem_state['current_solution']
    weight_matrix = problem_state['weight_matrix']
    best_increase = 0
    best_pair = None

    set_a = current_solution.set_a
    set_b = current_solution.set_b

    # Precompute the sum of weights to and from each node
    weight_to_a = weight_matrix[:, list(set_a)].sum(axis=1)
    weight_to_b = weight_matrix[:, list(set_b)].sum(axis=1)

    for i in set_a:
        for j in set_b:
            # Calculate the delta in cut value for swapping this pair of nodes
            delta = weight_to_a[i] - weight_to_a[j] + weight_to_b[j] - weight_to_b[i]

            # Adjust for the weight between i and j if they are connected
            if weight_matrix[i, j] != 0:
                delta += 2 * weight_matrix[i, j]
            # Check if this swap improves the cut value
            if delta > best_increase:
                best_increase = delta
                best_pair = (i, j)

    if best_pair is not None:
        return SwapOperator(list(best_pair)), {}
    else:
        return None, {}