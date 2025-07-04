from src.problems.mkp.components import *
import numpy as np
import itertools

def greedy_by_profit_1597(problem_state: dict, algorithm_data: dict, max_k: int = 2, epsilon: float = 0.01, **kwargs) -> tuple[BaseOperator, dict]:
    """Greedy heuristic for the Multidimensional Knapsack Problem, enhanced with opportunity cost scoring, k-flip exploration, and swap optimization.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "profits" (numpy.array): The profit value associated with each item.
            - "weights" (list of lists): A 2D list where each sublist represents the resource consumption of an item across all dimensions.
            - "capacities" (numpy.array): The maximum available capacity for each resource dimension.
            - "resource_num" (int): The number of resource dimensions.
            - "remaining_capacity" (numpy.array): The remaining capacity for each resource dimension.
            - "items_in_knapsack" (list[int]): A list of item indices that are currently included in the knapsack.
            - "items_not_in_knapsack" (list[int]): A list of item indices that are currently not included in the knapsack.
            - "feasible_items_to_add" (list[int]): A list of item indices that can be added without violating constraints.
            - "current_solution" (Solution): The current solution object.
            - "current_profit" (float): The current total profit of the solution.
            - get_problem_state (callable): def validation_solution(solution: Solution) -> bool: The function to get the problem state for given solution without modify it.
        algorithm_data (dict, optional): The algorithm dictionary for current algorithm only. In this algorithm, no specific data is necessary.
        max_k (int, optional): The maximum number of items to flip in k-flip exploration. Defaults to 2.
        epsilon (float, optional): A small constant added to the denominator to avoid division by zero. Defaults to 0.01.

    Returns:
        BaseOperator: The operator to apply to the current solution (e.g., AddOperator, RemoveOperator, FlipBlockOperator, SwapOperator).
        dict: An empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data from problem_state
    profits = problem_state["profits"]
    weights = problem_state["weights"]
    resource_num = problem_state["resource_num"]
    remaining_capacity = problem_state["remaining_capacity"]
    items_in_knapsack = problem_state["items_in_knapsack"]
    feasible_items_to_add = problem_state["feasible_items_to_add"]
    current_solution = problem_state["current_solution"]
    current_profit = problem_state["current_profit"]

    # Initialize variables to track the best operator and its corresponding score
    best_operator = None
    best_score = float('-inf')

    # Opportunity Cost Scoring
    for item in feasible_items_to_add:  # Ensure we only consider feasible items
        # Calculate the profit-to-weight ratio
        profit_to_weight_ratio = profits[item] / (sum(weights[res][item] for res in range(resource_num)) + epsilon)
        capacity_adjustment = np.min(
            [remaining_capacity[res] / (weights[res][item] + epsilon) for res in range(resource_num) if weights[res][item] > 0],
            initial=float('inf')
        )
        opportunity_cost = max(
            [profits[other_item] / (sum(weights[res][other_item] for res in range(resource_num)) + epsilon)
             for other_item in feasible_items_to_add if other_item != item],
            default=0
        )
        score = profit_to_weight_ratio * capacity_adjustment - opportunity_cost

        if score > best_score:
            best_operator = AddOperator(item)
            best_score = score

    # If a valid AddOperator is found, return it
    if best_operator is not None:
        return best_operator, {}

    # K-Flip Exploration
    best_profit = current_profit
    for k in range(1, max_k + 1):
        all_combinations = itertools.combinations(items_in_knapsack, k)
        for indices_to_flip in all_combinations:
            new_solution = current_solution.item_inclusion[:]
            for index in indices_to_flip:
                new_solution[index] = not new_solution[index]

            new_problem_state = problem_state["get_problem_state"](Solution(new_solution))
            if new_problem_state and new_problem_state["current_profit"] > best_profit:
                return FlipBlockOperator(list(indices_to_flip)), {}

    # Swap Optimization
    for item_in in items_in_knapsack:
        for item_out in feasible_items_to_add:  # Ensure we only swap with feasible items
            new_solution = current_solution.item_inclusion[:]
            new_solution[item_in], new_solution[item_out] = new_solution[item_out], new_solution[item_in]

            new_problem_state = problem_state["get_problem_state"](Solution(new_solution))
            if new_problem_state and new_problem_state["current_profit"] > best_profit:
                return SwapOperator(item_in, item_out), {}

    # If no operator improves the solution, return None
    return None, {}