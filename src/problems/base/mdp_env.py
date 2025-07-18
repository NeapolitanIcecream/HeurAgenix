from src.problems.base.env import BaseEnv
from src.problems.base.mdp_components import Solution, ActionOperator

class MDPEnv(BaseEnv):
    """Multi-agents env that stores the instance data, current solution, and problem state to support algorithm."""
    def __init__(self, data_name: str, env_class: type, problem: str, **kwargs):
        super().__init__(data_name, problem)
        self.gym_env = env_class(self.data_path)
        self.done = False
        self.reward = 0
        self.key_item = "total_reward"
        self.compare = lambda x, y: y - x


    @property
    def is_complete_solution(self) -> bool:
        return self.done

    @property
    def continue_run(self) -> bool:
        return not self.done

    def reset(self, output_dir: str=None):
        self.gym_env.reset()
        self.done = False
        self.reward = 0
        super().reset(output_dir)

    def load_data(self, data_path: str) -> None:
        pass

    def init_solution(self) -> None:
        return Solution()

    def get_key_value(self, solution: Solution=None) -> float:
        """Get the key value of the current solution based on the key item."""
        return self.reward

    def validation_solution(self, solution: Solution=None) -> bool:
        pass

    def run_operator(self, operator: ActionOperator, inplace: bool=True, heuristic_name: str=None) -> bool:
        if isinstance(operator, ActionOperator) and not self.done:
            solution = operator.run(self.current_solution)
            if inplace:
                self.current_solution = solution
                self.recording.append((str(heuristic_name), operator, str(solution)))
            _, reward, self.done, error_message = self.gym_env.step(operator.actions)
            if error_message:
                raise Exception(error_message)
            if isinstance(reward, int) or isinstance(reward, float):
                self.reward += reward
            elif isinstance(reward, list):
                self.reward += sum(reward)
            return True
        return False
    
    def summarize_env(self) -> str:
        if hasattr(self.gym_env, "summarize_env"):
            return self.gym_env.summarize_env()
        return None