from dataclasses import dataclass, field
from typing import List
import uuid
from src.problems.base.components import BaseSolution, BaseOperator

@dataclass
class MIRPeepholePattern:
    """
    Represents a single peephole optimization pattern for Machine IR,
    defined directly as a TableGen GICombineRule.
    """
    name: str  # Rule name, e.g., "CombineAddZero"
    td_rule: str  # The complete TableGen GICombineRule definition
    targeted_opcodes: List[str]  # Opcodes targeted by this rule, for analysis
    is_verified: bool = False
    id: int = -1  # To uniquely identify a pattern

    def __str__(self) -> str:
        return f"Pattern(id={self.id}, name={self.name}, verified={self.is_verified})"

class Solution(BaseSolution):
    """
    The solution for the LLVM MIR Peephole problem.
    It consists of a list of MIRPeepholePattern objects.
    """
    def __init__(self, patterns: List[MIRPeepholePattern]):
        super().__init__()
        self.patterns = patterns

    def __str__(self) -> str:
        if not self.patterns:
            return "Solution: (No patterns)"
        return "Solution:\n" + "\n".join(f"  - {str(p)}" for p in self.patterns)

class AddPatternOperator(BaseOperator):
    """Adds a new pattern to the solution."""
    def __init__(self, pattern: MIRPeepholePattern):
        super().__init__()
        self.pattern = pattern

    def run(self, solution: BaseSolution) -> Solution:
        assert isinstance(solution, Solution), "Operator must be run on a llvm_mir_peephole.Solution"
        # Use a UUID to ensure a globally unique ID and avoid concurrency issues.
        self.pattern.id = uuid.uuid4().int & ((1 << 32) - 1)  # Truncate to 32 bits for easier serialization
        new_patterns = solution.patterns + [self.pattern]
        return Solution(patterns=new_patterns)

class ModifyPatternOperator(BaseOperator):
    """Modifies an existing pattern in the solution."""
    def __init__(self, pattern_id: int, new_pattern: MIRPeepholePattern):
        super().__init__()
        self.pattern_id = pattern_id
        self.new_pattern = new_pattern

    def run(self, solution: BaseSolution) -> Solution:
        assert isinstance(solution, Solution), "Operator must be run on a llvm_mir_peephole.Solution"
        new_patterns = solution.patterns.copy()
        for i, p in enumerate(new_patterns):
            if p.id == self.pattern_id:
                # Keep the original ID.
                self.new_pattern.id = p.id
                # A modified pattern must be re-verified.
                self.new_pattern.is_verified = False
                new_patterns[i] = self.new_pattern
                break
        return Solution(patterns=new_patterns)

class RemovePatternOperator(BaseOperator):
    """Removes a pattern from the solution."""
    def __init__(self, pattern_id: int):
        super().__init__()
        self.pattern_id = pattern_id

    def run(self, solution: BaseSolution) -> Solution:
        assert isinstance(solution, Solution), "Operator must be run on a llvm_mir_peephole.Solution"
        new_patterns = [p for p in solution.patterns if p.id != self.pattern_id]
        return Solution(patterns=new_patterns)