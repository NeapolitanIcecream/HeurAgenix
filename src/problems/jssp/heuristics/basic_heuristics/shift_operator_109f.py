from src.problems.jssp.components import ShiftOperator

def shift_operator_109f(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[ShiftOperator, dict]:
    """
    This heuristic attempts to find a better schedule by shifting an operation within the same machine's queue.
    For each machine, it tries shifting each operation to all possible positions and evaluates the makespan.
    The shift that results in the best improvement (reduction in makespan) is selected.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "machine_num" (int): The total number of machines.
            - "current_solution" (Solution): The current solution state.
            - "current_makespan" (int): The current makespan of the schedule.
        problem_state["get_problem_state"] (callable): Function to get the new state data given a Solution instance.

    Returns:
        ShiftOperator: An operator that shifts a job in the schedule to achieve a local improvement.
        dict: An empty dictionary as this heuristic does not require algorithm data updates.
    """

    current_solution = problem_state['current_solution']
    machine_num = problem_state['machine_num']
    best_operator = None
    best_delta = float('inf')

    # Iterate over all machines
    for machine_id in range(machine_num):
        # Iterate over all operations in the machine's queue
        for current_position, job_id in enumerate(current_solution.job_sequences[machine_id]):
            # Try shifting the operation to all possible positions
            for new_position in range(len(current_solution.job_sequences[machine_id])):
                # Skip if the position is the same as the current one
                if current_position == new_position:
                    continue
                
                # Create a new solution with the operation shifted to the new position
                new_solution = ShiftOperator(machine_id, job_id, new_position).run(current_solution)
                new_state = problem_state["get_problem_state"](new_solution)
                
                # If the new solution is valid, evaluate its makespan
                if new_state is not None:
                    delta = new_state['current_makespan'] - problem_state['current_makespan']
                    
                    # If the makespan is improved, store this operator
                    if delta < best_delta:
                        best_operator = ShiftOperator(machine_id, job_id, new_position)
                        best_delta = delta

    # If a beneficial shift is found, return the corresponding operator
    if best_operator and best_delta < 0:
        return best_operator, {}
    else:
        return None, {}