import os
import glob
from typing import List, Dict

from typing import Optional
from src.problems.base.env import BaseEnv
from src.problems.base.components import BaseSolution
from src.problems.llvm_mir_peephole.components import Solution, MIRPeepholePattern
from src.problems.llvm_mir_peephole.llvm_utils import get_ir_stats
import subprocess
import tempfile
import shutil
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed


class Env(BaseEnv):
    """
    Environment for the LLVM Peephole Optimization problem.
    It manages the benchmark data (LLVM IR files), evaluates solutions (sets of patterns),
    and validates their semantic correctness.
    """
    def __init__(self, data_name: str, **kwargs):
        # data_name is expected to be a directory containing .ll files
        super().__init__(data_name, "llvm_mir_peephole")
        
        # The key metric to optimize is the instruction reduction rate.
        self.key_item = "instruction_reduction_rate"
        
        # A higher reduction rate is better.
        self.compare = lambda x, y: x - y
        
        # For this problem, a solution is always "complete".
        self.construction_steps = 1
        cpu_count = os.cpu_count()
        cpu_count = 1 if cpu_count is None else cpu_count // 2
        self.max_workers = kwargs.get("max_workers", cpu_count)

        # Cache LLVM bin path to avoid searching multiple times
        self._llvm_bin = None
        self.mtriple = os.getenv("LLVM_TRIPLE", "x86_64")

    # ------------------------ LLVM Toolchain Helpers ------------------------

    @lru_cache(maxsize=1)
    def _get_llvm_bin(self) -> str:
        """
        Attempts to resolve the directory containing LLVM executables.
        Search order:
        1. Environment variables `LLVM_HOME` or `LLVM_INSTALL_DIR`.
        2. The output of `llvm-config --bindir`.
        3. Inference from `shutil.which("llc")`.
        Raises FileNotFoundError if all attempts fail.
        """
        # 1. Environment variables
        env_path = os.getenv("LLVM_HOME") or os.getenv("LLVM_INSTALL_DIR")
        if env_path and os.path.isdir(env_path):
            bin_dir = os.path.join(env_path, "bin") if os.path.basename(env_path) != "bin" else env_path
            llc_path = os.path.join(bin_dir, "llc")
            if os.path.isfile(llc_path):
                return bin_dir

        # 2. llvm-config
        try:
            bindir = subprocess.check_output(["llvm-config", "--bindir"], text=True).strip()
            if bindir and os.path.isdir(bindir):
                return bindir
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # 3. which llc
        llc_full = shutil.which("llc")
        if llc_full:
            return os.path.dirname(llc_full)

        raise FileNotFoundError("无法定位 LLVM 安装路径，请设置 LLVM_HOME 环境变量或确保 llvm-config/llc 可用")

    @property
    def is_complete_solution(self) -> bool:
        # Any set of patterns is a potentially valid solution.
        return True

    def load_data(self, data_path: str) -> Dict[str, str]:
        """
        Loads all LLVM IR files (.ll) from the specified directory.
        
        Args:
            data_path (str): The path to the directory containing benchmark .ll files.
            
        Returns:
            Dict[str, str]: A dictionary mapping file paths to their content.
        """
        assert os.path.isdir(data_path), f"Data path {data_path} must be a directory."
        ll_files = glob.glob(os.path.join(data_path, '*.ll'))
        
        if not ll_files:
            raise FileNotFoundError(f"No .ll files found in directory: {data_path}")
            
        return {file_path: open(file_path, 'r').read() for file_path in ll_files}

    def init_solution(self) -> Solution:
        """Initializes an empty solution with no patterns."""
        return Solution(patterns=[])

    @lru_cache(maxsize=None)
    def _get_baseline_instruction_count(self, ir_content: str) -> int:
        """
        Compiles the baseline IR (without any patterns) and returns its instruction count.
        The result is cached to avoid repeated compilations of the same IR.
        """
        baseline_asm = self._apply_patterns_to_ir(ir_content, [])
        if not baseline_asm:
            return 0
        return self._get_instruction_count(baseline_asm, is_ir=False)

    def get_key_value(self, solution: Optional[BaseSolution] = None) -> float:
        """
        Calculates the key value of a solution, which is the instruction reduction rate.
        It applies the patterns to the benchmark suite and measures the change in instruction count.
        This process is parallelized across the benchmark files.
        
        Args:
            solution (Solution): The solution to evaluate. If None, uses self.current_solution.
            
        Returns:
            float: The percentage of instruction reduction (e.g., 0.05 for 5% reduction).
        """
        target_solution = solution if solution is not None else self.current_solution
        assert isinstance(target_solution, Solution)

        if not self.validation_solution(target_solution):
            return -1.0  # Return a negative reward for invalid solutions
        
        total_original_instructions = 0
        total_optimized_instructions = 0

        # Helper function to process a single IR file
        def _evaluate_ir(ir_content):
            original_count = self._get_baseline_instruction_count(ir_content)

            if original_count == 0:
                return 0, 0

            # Apply patterns and get instruction count
            optimized_asm = self._apply_patterns_to_ir(ir_content, target_solution.patterns)
            if not optimized_asm: # If plugin fails, it returns None
                # Penalize failed application by treating it as infinite cost
                optimized_count = float('inf')
            else:
                optimized_count = self._get_instruction_count(optimized_asm, is_ir=False)
            
            return original_count, optimized_count

        # Parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ir = {executor.submit(_evaluate_ir, ir): ir for ir in self.instance_data.values()}
            for future in as_completed(future_to_ir):
                try:
                    original, optimized = future.result()
                    total_original_instructions += original
                    total_optimized_instructions += optimized
                except Exception:
                    # Optional: Log the exception
                    # logger.error(f"IR processing failed: {exc}")
                    pass
        
        if total_original_instructions == 0:
            return 0.0

        reduction_rate = (total_original_instructions - total_optimized_instructions) / total_original_instructions
        return reduction_rate

    # ---------------------- Peephole Application and Verification ----------------------

    # Note: `patterns` is a list and cannot be hashed directly, so lru_cache is not used here.
    def _apply_patterns_to_ir(self, ir_code: str, patterns: List[MIRPeepholePattern] = []) -> Optional[str]:
        """
        Applies a set of MIR peephole patterns to the given IR code by building and using a
        GICombiner pass plugin.
        
        If plugin creation or application fails, it returns None.
        """
        from src.problems.llvm_mir_peephole.plugin_builder import ensure_plugin
        plugin_path = ensure_plugin(patterns) if patterns else None

        if patterns and not plugin_path:
            # If the plugin can't be built (e.g., no valid patterns), return None.
            return None

        tmp_ir_path = None
        try:
            llvm_bin = self._get_llvm_bin()
            llc_exec = os.path.join(llvm_bin, "llc")

            with tempfile.NamedTemporaryFile("w", suffix=".ll", delete=False) as tmp_ir:
                tmp_ir.write(ir_code)
                tmp_ir_path = tmp_ir.name

            # Run llc with the custom pass plugin.
            cmd = [
                llc_exec,
                f"-mtriple={self.mtriple}",
                "-global-isel",
            ]
            if plugin_path:
                cmd.extend([
                    f"-load-pass-plugin={plugin_path}",
                    "-passes=my-micombiner",
                ])

            cmd.extend(["-o", "-", tmp_ir_path]) # Output assembly to stdout

            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                # Optional: Log the error for debugging.
                # print(f"WARNING: llc failed during pattern application:\n{result.stderr}")
                return None # Fallback to original code

            return result.stdout
        except (FileNotFoundError):
            # This catches if llc is not found
            return None
        finally:
            if tmp_ir_path and os.path.exists(tmp_ir_path):
                os.remove(tmp_ir_path)

    def _get_instruction_count(self, code: str, is_ir: bool) -> int:
        """
        Counts the number of instructions in the given code.
        - If the code is LLVM IR, it uses the robust `get_ir_stats` utility.
        - If the code is assembly, it falls back to a simple line-based count.
        """
        if is_ir:
            stats = get_ir_stats(code, self.mtriple)
            return stats.get("total_instructions", 0) if stats else 0

        # Fallback for assembly code (less accurate)
        instr_lines = 0
        for line in code.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith(('.', '#', ';')) or stripped.endswith(':'):
                continue
            if stripped.startswith(("#APP", "#NO_APP")):
                continue
            
            code_part = stripped.split("#", 1)[0].strip()
            if code_part:
                instr_lines += 1
        return instr_lines

    # --- Solution Verification ---

    def validation_solution(self, solution: Optional[BaseSolution] = None) -> bool:
        """
        Verifies the semantic correctness of all unverified patterns in the solution.
        Returns False if any pattern fails verification.
        """
        target_solution = solution if solution is not None else self.current_solution
        assert isinstance(target_solution, Solution)

        for pattern in target_solution.patterns:
            if not pattern.is_verified:
                if not self._run_verification_test(pattern):
                    return False
        return True

    def _run_verification_test(self, pattern: MIRPeepholePattern) -> bool:
        """
        Verifies a pattern by compiling with and without it using `llc -verify-machineinstrs`.
        
        Logic:
          1. Takes up to 5 benchmark IR files.
          2. For each IR, compiles it with both -O0 and -O2, once without the plugin
             and once with the single-pattern plugin.
          3. If any compilation fails, the pattern is considered invalid.
        """
        try:
            if not self.instance_data:
                return False

            llvm_bin = self._get_llvm_bin()
            llc_exec = os.path.join(llvm_bin, "llc")

            from src.problems.llvm_mir_peephole.plugin_builder import ensure_plugin

            plugin_path = ensure_plugin([pattern])

            def llc_compile(ir_text: str, use_plugin: bool, opt_level: str) -> bool:
                with tempfile.NamedTemporaryFile("w", suffix=".ll", delete=False) as tmp_ir:
                    tmp_ir.write(ir_text)
                    tmp_ir_path = tmp_ir.name
                cmd = [llc_exec, f"-mtriple={self.mtriple}", opt_level, "-verify-machineinstrs", "-o", "/dev/null"]
                if use_plugin and plugin_path:
                    cmd.extend(["-global-isel", f"-load-pass-plugin={plugin_path}", "-passes=my-micombiner"])
                cmd.append(tmp_ir_path)
                try:
                    # Use subprocess.run to capture stderr on failure
                    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if result.returncode != 0:
                        # Optional: Log the verification failure details
                        # print(f"Verification failed for command: {' '.join(cmd)}\n{result.stderr}")
                        return False
                    return True
                except FileNotFoundError:
                    return False
                finally:
                    if os.path.exists(tmp_ir_path):
                        os.remove(tmp_ir_path)

            sample_irs = list(self.instance_data.values())[:5]
            opt_levels = ["-O0", "-O2"]
            for sample_ir in sample_irs:
                for opt in opt_levels:
                    if not llc_compile(sample_ir, False, opt):
                        return False
                    if not llc_compile(sample_ir, True, opt):
                        return False

            pattern.is_verified = True
            return True
        except Exception:
            return False

    # ------------------ Summary ----------------------------------

    def summarize_env(self) -> str:
        """Provides a simple summary of the current environment for CLI output."""
        from src.problems.llvm_mir_peephole.components import Solution as _PeepholeSolution
        num_patterns = len(self.current_solution.patterns) if isinstance(self.current_solution, _PeepholeSolution) else 0
        summary = (
            f"Data set: {self.data_ref_name} | Files: {len(self.instance_data)}\n"
            f"Current patterns: {num_patterns}\n"
            f"{self.key_item}: {self.key_value:.4f}\n"
        )
        return summary