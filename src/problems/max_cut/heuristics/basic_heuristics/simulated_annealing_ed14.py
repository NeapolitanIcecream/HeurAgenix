from src.problems.max_cut.components import *
import random
import math

def simulated_annealing_ed14(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[SwapOperator, dict]:
    """
    Simulated Annealing heuristic for the Max Cut problem. It probabilistically chooses to swap a node from one set to another,
    potentially accepting worse solutions early on to escape local optima, with the probability of accepting worse solutions
    decreasing over time.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "total_nodes" (int): The total number of vertices in the graph.
            - "current_solution" (Solution): The current solution of the Max Cut problem.
            - "current_cut_value" (int or float): The total weight of edges between set A and set B in the current solution.
            - get_problem_state (callable): def validation_solution(solution: Solution) -> bool: The function to get the problem state for given solution without modify it.
        algorithm_data (dict): Contains the data specific to the simulated annealing algorithm.
            - "temperature" (float): The current temperature for the simulated annealing process.
            - "cooling_rate" (float): The rate at which the temperature decreases.
        **kwargs: Hyperparameters for the algorithm.
            - "initial_temperature" (float): The starting temperature for the annealing process.
            - "final_temperature" (float): The temperature at which the annealing process stops.
            - "alpha" (float): The cooling rate factor.

    Returns:
        SwapOperator: The operator to swap a node between sets if a valid move is found.
        dict: Updated algorithm data with the new temperature.
    """
    # Hyperparameters with default values
    initial_temperature = kwargs.get('initial_temperature', 100.0)
    final_temperature = kwargs.get('final_temperature', 0.001)
    alpha = kwargs.get('alpha', 0.95)

    # Initialize temperature if not present in algorithm_data
    temperature = algorithm_data.get('temperature', initial_temperature)
    cooling_rate = algorithm_data.get('cooling_rate', alpha)

    # Current solution and cut value
    current_solution = problem_state['current_solution']
    current_cut_value = problem_state['current_cut_value']

    # If the temperature is already below the final temperature, do not perform any operation
    if temperature <= final_temperature:
        return None, {}

    # Select a random node to swap
    node = random.randint(0, problem_state['node_num'] - 1)

    # Create a new solution with the node swapped
    new_solution = SwapOperator([node]).run(current_solution)

    # Get the state data for the new solution
    new_problem_state = problem_state["get_problem_state"](new_solution)

    # If the new solution is invalid, return no operation
    if new_problem_state is None:
        return None, {}

    # Calculate the new cut value
    new_cut_value = new_problem_state['current_cut_value']

    # Calculate the change in cut value
    delta = new_cut_value - current_cut_value

    # If the new solution is better or equal, accept it
    if delta >= 0:
        return SwapOperator([node]), {'temperature': temperature * cooling_rate}

    # If the new solution is worse, accept it with a certain probability
    acceptance_probability = math.exp(delta / temperature)
    if random.random() < acceptance_probability:
        return SwapOperator([node]), {'temperature': temperature * cooling_rate}

    # If the new solution is not accepted, return no operation
    return None, {'temperature': temperature * cooling_rate}