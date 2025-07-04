from src.problems.dposp.components import Solution, InsertOperator

def longest_order_next_c9cb(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[InsertOperator, dict]:
    """
    Longest Order Next heuristic algorithm for DPOSP which selects the unfulfilled order with the longest processing time
    and schedules it in the most appropriate position on a production line that minimizes delays and respects deadlines.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "production_rate" (numpy.array): 2D array of production time for each product on each production line.
            - "order_quantity" (numpy.array): 1D array of the quantity required for each order.
            - "order_product" (numpy.array): 1D array mapping each order to its required product.
            - "feasible_orders_to_fulfill" (list): List of feasible orders that can be fulfilled without delaying other planned orders.
            - "current_solution" (Solution): Current scheduling solution.
            - "validation_single_production_schedule" (callable): Function to check the validity of a single production schedule.

    Returns:
        (InsertOperator): Operator to insert the longest order into the most appropriate position on a production line.
        (dict): Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data
    production_rate = problem_state["production_rate"]
    order_quantity = problem_state["order_quantity"]
    order_product = problem_state["order_product"]
    feasible_orders = problem_state["feasible_orders_to_fulfill"]
    current_solution = problem_state["current_solution"]
    validation_single_production_schedule = problem_state["validation_single_production_schedule"]

    # If there are no feasible orders to fulfill, return None
    if not feasible_orders:
        return None, {}

    # Calculate total processing time for each feasible order
    processing_times = {
        order_id: order_quantity[order_id] / production_rate[:, order_product[order_id]].mean()
        for order_id in feasible_orders if production_rate[:, order_product[order_id]].mean() > 0
    }

    # Find the order with the maximum processing time
    longest_order_id = max(processing_times, key=processing_times.get)

    # Find the best position to insert this order in any production line
    for line_id, line_schedule in enumerate(current_solution.production_schedule):
        for position in range(len(line_schedule) + 1):
            new_schedule = line_schedule[:]
            new_schedule.insert(position, longest_order_id)
            if validation_single_production_schedule(line_id, new_schedule):
                # Found a valid position to insert the order, return the corresponding InsertOperator
                return InsertOperator(production_line_id=line_id, order_id=longest_order_id, position=position), {}

    # If no valid position is found, return None
    return None, {}