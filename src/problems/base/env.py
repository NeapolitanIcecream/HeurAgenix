import os
import traceback
from src.problems.base.components import BaseSolution, BaseOperator
from src.util.util import load_function, search_file
from typing import Optional, Union, List, Dict, Callable


class BaseEnv:
    """Base env that stores the static global data, current solution, dynamic state and provide necessary to support algorithm."""
    def __init__(self, data_name: str, problem: str, **kwargs):
        self.problem = problem
        self.data_path = search_file(data_name, problem)
        self.data_ref_name = data_name.split(os.sep)[-1]
        assert self.data_path is not None
        self.instance_data: dict = self.load_data(self.data_path)
        self.current_solution: BaseSolution = self.init_solution()
        self.algorithm_data: dict = {}
        self.recordings: List[dict] = []
        self.output_dir: Optional[str] = None
        # Maximum step to constructive a complete solution
        self.construction_steps: Optional[int] = None
        # Key item in state to compare the solution
        self.key_item: Optional[str] = None
        # Returns the advantage of the first and second key value
        # A return value greater than 0 indicates that first is better and the larger the number, the greater the advantage.
        self.compare: Optional[Callable] = None

        problem_state_file = search_file("problem_state.py", problem=self.problem)
        assert problem_state_file is not None, f"Problem state code file {problem_state_file} does not exist"
        self.get_instance_problem_state = load_function(problem_state_file, problem=self.problem, function_name="get_instance_problem_state")
        self.get_solution_problem_state = load_function(problem_state_file, problem=self.problem, function_name="get_solution_problem_state")
        self.problem_state = self.get_problem_state()


    @property
    def is_complete_solution(self) -> bool:
        raise NotImplementedError

    @property
    def is_valid_solution(self) -> bool:
        return self.validation_solution(self.current_solution)

    @property
    def continue_run(self) -> bool:
        return True

    @property
    def key_value(self) -> float:
        """Get the key value of the current solution."""
        return self.get_key_value(self.current_solution)

    def get_key_value(self, solution: Optional[BaseSolution] = None) -> float:
        """Get the key value of the solution."""
        raise NotImplementedError

    def reset(self, output_dir: Optional[str] = None):
        self.current_solution = self.init_solution()
        self.problem_state = self.get_problem_state()
        self.algorithm_data = {}
        self.recordings = []
        if output_dir:
            if os.sep in output_dir:
                self.output_dir = output_dir
            else:
                amlt_output_dir = os.getenv("AMLT_OUTPUT_DIR")
                base_output_dir = os.path.join(amlt_output_dir, "..", "..", "output") if amlt_output_dir else "output"
                self.output_dir = os.path.join(base_output_dir, self.problem, "result", self.data_ref_name, output_dir)
            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self, data_path: str) -> dict:
        raise NotImplementedError

    def init_solution(self) -> BaseSolution:
        raise NotImplementedError

    def helper_function(self) -> dict:
        return {"get_problem_state": self.get_problem_state, "validation_solution": self.validation_solution}

    def get_problem_state(self, solution: Optional[BaseSolution] = None) -> Optional[dict]:
        target_solution = solution if solution is not None else self.current_solution
        
        instance_problem_state = self.get_instance_problem_state(self.instance_data)
        solution_problem_state = self.get_solution_problem_state(self.instance_data, target_solution)
        helper_function = self.helper_function()
        
        if solution_problem_state and self.key_item:
            return {
                **self.instance_data,
                "current_solution": target_solution,
                self.key_item: self.get_key_value(target_solution),
                **helper_function,
                **instance_problem_state,
                **solution_problem_state,
            }
        return None

    def validation_solution(self, solution: Optional[BaseSolution] = None) -> bool:
        """Check the validation of this solution"""
        raise NotImplementedError

    def run_heuristic(self, heuristic: Callable, parameters:dict={}, add_record_item: dict={}) -> Union[BaseOperator, str]:
        try:
            result = heuristic(
                problem_state=self.problem_state,
                algorithm_data=self.algorithm_data,
                **parameters
            )
            operator, delta = result if isinstance(result, tuple) and len(result) == 2 else (result, None)

            if isinstance(operator, BaseOperator):
                self.run_operator(operator)
                if delta:
                    self.algorithm_data.update(delta)
            
            record_item = {"operation_id": len(self.recordings), "heuristic": heuristic.__name__, "operator": str(operator)}
            record_item.update(add_record_item)
            self.recordings.append(record_item)
            if isinstance(operator, BaseOperator):
                return operator
            # Handle cases where the heuristic might return something unexpected
            return str(operator)
        except Exception as e:
            trace_string = traceback.format_exc()
            print(trace_string)
            return trace_string

    def run_operator(self, operator: BaseOperator) -> BaseOperator:
        if isinstance(operator, BaseOperator):
            self.current_solution = operator.run(self.current_solution)
            self.problem_state = self.get_problem_state()
        return operator

    def summarize_env(self) -> str:
        raise NotImplementedError

    def __getstate__(self):  
        state = self.__dict__.copy()  
        state.pop("get_instance_problem_state", None)
        state.pop("get_solution_problem_state", None)
        return state  
  
    def __setstate__(self, state):  
        self.__dict__.update(state)  
        self.get_instance_problem_state = load_function("problem_state.py", problem=self.problem, function_name="get_instance_problem_state")
        self.get_solution_problem_state = load_function("problem_state.py", problem=self.problem, function_name="get_solution_problem_state")

    def dump_result(self, content_dict: dict={}, dump_records: list=["operation_id", "operator", "heuristic"], result_file: str="result.txt") -> str:
        content = f"-data: {self.data_path}\n"
        content += f"-current_solution:\n{self.current_solution}\n"
        content += f"-is_complete_solution: {self.is_complete_solution}\n"
        content += f"-is_valid_solution: {self.is_valid_solution}\n"
        content += f"-{self.key_item}: {self.key_value}\n"
        for item, value in content_dict.items():
            content += f"-{item}: {value}\n"
        if dump_records and self.recordings:
            # Ensure all keys in dump_records exist in the first recording item
            valid_dump_records = [item for item in dump_records if item in self.recordings[0]]
            if valid_dump_records:
                content += "-trajectory:\n" + "\t".join(valid_dump_records) + "\n"
                trajectory_str = "\n".join([
                    "\t".join([str(recording_item.get(item, "None")) for item in valid_dump_records])
                    for recording_item in self.recordings
                ])
                content += trajectory_str

        if self.output_dir != None and result_file != None:
            output_file = os.path.join(self.output_dir, result_file)
            with open(output_file, "w") as file:  
                file.write(content) 
        
        return content

