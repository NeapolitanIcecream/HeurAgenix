from src.problems.dposp.components import *

def greedy_by_order_density_c702(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AppendOperator, dict]:
    """
    This heuristic for DPOSP selects the next order to schedule based on an 'order density' metric, defined as the ratio of the order's value to the combined time taken for production and transition. For each unscheduled order, compute the order density as 1 divided by the sum of the production time on the assigned production line and the transition time from the last scheduled product. Select the order with the highest density value that can be completed before its deadline and append it to the corresponding production line schedule.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "production_rate" (numpy.array): 2D array of production time for each product on each production line.
            - "transition_time" (numpy.array): 3D array of transition time between products on each production line.
            - "order_product" (numpy.array): 1D array mapping each order to its required product.
            - "order_deadline" (numpy.array): 1D array of the deadline for each order.
            - "current_solution" (Solution): Current scheduling solution.
            - "feasible_orders_to_fulfill" (list): The feasible orders that can be fulfilled based on the current solution without delaying other planned orders.
            - "validation_single_production_schedule" (callable): Function to check whether the production schedule is valid.

    Returns:
        AppendOperator: The operator to append the selected order to the production line's schedule.
        dict: Empty dictionary as no algorithm data is updated.
    """
    
    best_density = -1
    best_order_id = None
    best_production_line_id = None

    # Get necessary data from problem_state
    production_rate = problem_state["production_rate"]
    transition_time = problem_state["transition_time"]
    order_product = problem_state["order_product"]
    order_deadline = problem_state["order_deadline"]
    current_solution = problem_state["current_solution"]
    feasible_orders_to_fulfill = problem_state["feasible_orders_to_fulfill"]
    validation_single_production_schedule = problem_state["validation_single_production_schedule"]

    # Iterate through each feasible order
    for order_id in feasible_orders_to_fulfill:
        # Get the required product and deadline for this order
        product = order_product[order_id]
        deadline = order_deadline[order_id]
        
        # Check each production line for feasibility
        for production_line_id, production_schedule in enumerate(current_solution.production_schedule):
            # Calculate the production time for this order on the current line
            production_time = 1 / production_rate[production_line_id][product] if production_rate[production_line_id][product] > 0 else float('inf')
            
            # Calculate transition time from the last product in the schedule, if any
            last_product = order_product[production_schedule[-1]] if production_schedule else 0
            transition_time_from_last = transition_time[production_line_id][last_product][product]

            # Calculate total time (sum of production and transition times)
            total_time = production_time + transition_time_from_last

            # Calculate order density for this order on the current line
            order_density = 1 / total_time if total_time > 0 else 0

            # If this order density is better than the current best and can be completed by deadline
            if order_density > best_density:
                # Check if appending this order to the current schedule is valid
                new_schedule = production_schedule + [order_id]
                if validation_single_production_schedule(production_line_id, new_schedule):
                    # Update the best order and line found so far
                    best_density = order_density
                    best_order_id = order_id
                    best_production_line_id = production_line_id

    # If a valid order was found that improves the density
    if best_order_id is not None and best_production_line_id is not None:
        # Return the AppendOperator for this order and line, with no update to algorithm data
        return AppendOperator(best_production_line_id, best_order_id), {}
    
    # If no valid order could be found that improves the density without missing deadlines, return None
    return None, {}