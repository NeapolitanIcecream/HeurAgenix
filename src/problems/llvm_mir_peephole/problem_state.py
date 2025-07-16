from collections import Counter
from typing import Dict, List
from src.problems.llvm_mir_peephole.components import Solution
from src.problems.llvm_mir_peephole.isa_metadata import load_isa_metadata
from src.problems.llvm_mir_peephole.llvm_utils import get_ir_stats
import os

def get_instance_problem_state(instance_data: Dict[str, str]) -> Dict:
    """
    Extracts instance-level problem state from the benchmark data.

    Args:
        instance_data (Dict[str, str]): A dictionary of IR file paths to their content.

    Returns:
        Dict: A dictionary containing instance-level features:
            - total_instructions (int): Total instruction count across all files.
            - opcode_distribution (Dict[str, int]): Frequency of each opcode.
            - file_count (int): Number of files in the benchmark.
    """
    full_ir_text = "\n".join(instance_data.values())
    mtriple = os.getenv("LLVM_TRIPLE", "x86_64")
    
    # Use the robust utility to get accurate stats
    stats = get_ir_stats(full_ir_text, mtriple=mtriple)
    total_instructions = stats.get("total_instructions", 0) if stats else 0
    opcode_distribution = stats.get("opcode_distribution", {}) if stats else {}

    # ISA meta information
    isa_meta = load_isa_metadata()
    isa_instr_classes = isa_meta.get("instrs", {})
    isa_registers = isa_meta.get("registers", {})

    # Extract a sample subset for the LLM to observe instruction/register semantics,
    # as a full JSON dump might exceed the context window.
    isa_instr_sample = list(isa_instr_classes.keys())[:50]
    isa_register_sample = list(isa_registers.keys())[:50]

    return {
        "total_instructions": total_instructions,
        "opcode_distribution": opcode_distribution,
        "file_count": len(instance_data),
        "isa_instr_classes": len(isa_instr_classes),
        "isa_register_classes": len(isa_registers),
        "isa_instr_sample": isa_instr_sample,
        "isa_register_sample": isa_register_sample,
    }

def get_solution_problem_state(instance_data: Dict[str, str], solution: Solution) -> Dict:
    """
    Extracts solution-level problem state.

    Args:
        instance_data (Dict[str, str]): The benchmark data.
        solution (Solution): The current solution (set of patterns).

    Returns:
        Dict: A dictionary containing solution-level features:
            - num_patterns (int): The number of patterns in the current solution.
            - num_verified_patterns (int): The number of semantically verified patterns.
            - targeted_opcodes (List[str]): A list of opcodes targeted by the patterns.
    """
    num_patterns = len(solution.patterns)
    num_verified_patterns = sum(1 for p in solution.patterns if p.is_verified)
    
    # Extract opcodes directly from the structured pattern data
    targeted_opcodes = []
    for p in solution.patterns:
        targeted_opcodes.extend(p.targeted_opcodes)

    unique_targeted = list(set(targeted_opcodes))

    # Coverage against ISA instruction classes
    isa_meta = load_isa_metadata()
    isa_instr_classes = set(isa_meta.get("instrs", {}).keys())
    isa_coverage = 0.0
    if isa_instr_classes:
        isa_coverage = len(set(unique_targeted) & isa_instr_classes) / len(isa_instr_classes)

    # Coverage: ratio of opcodes covered by current patterns to all opcodes present.
    full_ir_text = "\n".join(instance_data.values())
    mtriple = os.getenv("LLVM_TRIPLE", "x86_64")
    stats = get_ir_stats(full_ir_text, mtriple=mtriple)
    opcode_distribution = stats.get("opcode_distribution", {}) if stats else {}
    
    coverage = 0.0
    if opcode_distribution:
        coverage = len([op for op in unique_targeted if op in opcode_distribution]) / len(opcode_distribution)
    
    return {
        "num_patterns": num_patterns,
        "num_verified_patterns": num_verified_patterns,
        "targeted_opcodes": unique_targeted,
        "opcode_coverage": coverage,
        "isa_coverage": isa_coverage,
    }

def get_observation_problem_state(problem_state: dict) -> dict:
    """
    Extracts the core problem state features for the LLM's observation.

    Args:
        problem_state (dict): The full problem state.

    Returns:
        dict: A simplified dictionary for the LLM.
    """
    return {
        "num_patterns": problem_state.get("num_patterns"),
        "num_verified_patterns": problem_state.get("num_verified_patterns"),
        "achieved_reduction_rate": problem_state.get("instruction_reduction_rate"),
        "targeted_opcodes": problem_state.get("targeted_opcodes"),
        "opcode_distribution": problem_state.get("opcode_distribution"),
        "opcode_coverage": problem_state.get("opcode_coverage"),
        "isa_coverage": problem_state.get("isa_coverage"),
        "isa_instr_classes": problem_state.get("isa_instr_classes"),
        "isa_register_classes": problem_state.get("isa_register_classes"),
        "isa_instr_sample": problem_state.get("isa_instr_sample"),
        "isa_register_sample": problem_state.get("isa_register_sample"),
    }