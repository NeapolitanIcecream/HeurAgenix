from src.problems.dposp.components import *

def exchange_production_orders_eda2(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[SwapOperator, dict]:
    """
    The exchange production orders heuristic tries to improve the current production schedule by swapping two non-adjacent orders
    from the same or different production lines and assessing the impact on the total production and transition times. The heuristic
    is based on the 2-opt approach from TSP and is adapted to consider DPOSP-specific constraints such as varying production speeds,
    transition times, order deadlines, and production line capabilities.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - production_rate (numpy.array): 2D array of production time for each product on each production line.
            - transition_time (numpy.array): 3D array of transition time between products on each production line.
            - order_product (numpy.array): 1D array mapping each order to its required product.
            - current_solution (Solution): The current scheduling solution.
            - feasible_orders_to_fulfill (list): List of feasible orders that can be fulfilled without delaying other planned orders.
            - validation_single_production_schedule (callable): Function to check if a production schedule is valid for a given production line.
            - get_problem_state (callable): def validation_solution(solution: Solution) -> bool: The function to get the problem state for given solution without modify it.
        algorithm_data (dict): Used for storing data necessary for the algorithm.

    Returns:
        SwapOperator: Operator that swaps two orders in the production schedule.
        dict: Empty dictionary, as this algorithm does not update algorithm_data.
    """

    # Retrieve necessary information from problem_state
    production_rate = problem_state["production_rate"]
    transition_time = problem_state["transition_time"]
    order_product = problem_state["order_product"]
    current_solution = problem_state["current_solution"]
    feasible_orders = problem_state["feasible_orders_to_fulfill"]
    validation_schedule = problem_state["validation_single_production_schedule"]

    # Initialize best improvement and operator
    best_improvement = 0
    best_operator = None

    # Iterate through all pairs of feasible orders to find the best swap
    for line in range(len(production_rate)):
        for i in range(len(current_solution.production_schedule[line])):
            for j in range(i + 1, len(current_solution.production_schedule[line])):
                order_i = current_solution.production_schedule[line][i]
                order_j = current_solution.production_schedule[line][j]

                # Perform swap on a copy of the current production schedule
                new_schedule = [schedule[:] for schedule in current_solution.production_schedule]
                new_schedule[line][i], new_schedule[line][j] = new_schedule[line][j], new_schedule[line][i]

                # Validate the new schedule for the production line
                if not validation_schedule(line, new_schedule[line]):
                    continue

                # Get the state data for the new solution
                new_problem_state = problem_state["get_problem_state"](Solution(new_schedule))

                # Calculate the time cost difference
                old_time_cost = problem_state["total_time_cost_per_production_line"][line]
                new_time_cost = new_problem_state["total_time_cost_per_production_line"][line]

                # If the new schedule is better, store the improvement and the operator
                if new_time_cost < old_time_cost and (new_time_cost - old_time_cost) < best_improvement:
                    best_improvement = new_time_cost - old_time_cost
                    best_operator = SwapOperator(line, i, line, j)

    # If a best operator was found, return it, else return None
    if best_operator:
        return best_operator, {}
    else:
        return None, {}