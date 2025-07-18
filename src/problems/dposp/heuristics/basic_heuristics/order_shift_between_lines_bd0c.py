from src.problems.dposp.components import *

def order_shift_between_lines_bd0c(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[RelocateOperator, dict]:
    """
    This heuristic attempts to shift an unfulfilled order from one production line to another while adhering to machine capabilities, transition rules, and order deadlines.
    
    Args:
        problem_state (dict): The dictionary contains the problem state.
        algorithm_data (dict): Contains data specific to the algorithm (unused in this heuristic).
    
    Returns:
        (RelocateOperator, dict): The operator to shift an order between production lines and an empty dictionary, as the heuristic does not update algorithm_data.
    """
    # Retrieve necessary data from problem_state
    production_rate = problem_state["production_rate"]
    transition_time = problem_state["transition_time"]
    current_solution = problem_state["current_solution"]
    feasible_orders_to_fulfill = problem_state["feasible_orders_to_fulfill"]
    validation_single_production_schedule = problem_state["validation_single_production_schedule"]
    
    # Set a default value for optional hyper parameters
    max_attempts = kwargs.get('max_attempts', 100)

    # Initialize variables to store the best shift found
    best_order_id = None
    best_source_line_id = None
    best_target_line_id = None
    best_position = None
    best_delta_time_cost = float('inf')

    # Iterate through each feasible order to find a beneficial shift
    for source_line_id, source_schedule in enumerate(current_solution.production_schedule):
        for order_id in source_schedule:
            for target_line_id, target_schedule in enumerate(current_solution.production_schedule):
                # Skip if it's the same production line or the line cannot produce the product
                if source_line_id == target_line_id or production_rate[target_line_id][problem_state["order_product"][order_id]] == 0:
                    continue

                # Find the best position to insert the order in the target production line
                for position in range(len(target_schedule) + 1):
                    # Copy the current schedule and try inserting the order
                    trial_target_schedule = target_schedule[:]
                    trial_target_schedule.insert(position, order_id)
                    trial_source_schedule = source_schedule[:]
                    trial_source_schedule.remove(order_id)

                    # Validate the trial schedule
                    if not validation_single_production_schedule(source_line_id, trial_source_schedule) or not validation_single_production_schedule(target_line_id, trial_target_schedule):
                        continue  # Skip if the new schedule is not valid

                    # Calculate the delta time cost of the new schedule
                    new_production_schedule = [schedule[:] for schedule in current_solution.production_schedule]
                    new_production_schedule[target_line_id] = trial_target_schedule
                    new_production_schedule[source_line_id] = trial_source_schedule
                    state_data_for_trial = problem_state["get_problem_state"](Solution(new_production_schedule))
                    delta_time_cost = state_data_for_trial["total_time_cost_per_production_line"][target_line_id] + state_data_for_trial["total_time_cost_per_production_line"][source_line_id] - problem_state["total_time_cost_per_production_line"][target_line_id] - problem_state["total_time_cost_per_production_line"][source_line_id]

                    # Check if this shift leads to a better solution
                    if delta_time_cost < best_delta_time_cost:
                        best_order_id = order_id
                        best_source_line_id = source_line_id
                        best_target_line_id = target_line_id
                        best_position = position
                        best_delta_time_cost = delta_time_cost

    # If a beneficial shift is found, create and return the corresponding operator
    if best_order_id is not None:
        return RelocateOperator(
            source_production_line_id=best_source_line_id,
            source_position=problem_state["current_solution"].production_schedule[best_source_line_id].index(best_order_id),
            target_production_line_id=best_target_line_id,
            target_position=best_position
        ), {}

    # If no beneficial shift is found, return None
    return None, {}