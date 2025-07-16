"""plugin_builder.py

This module automatically transforms a list of `MIRPeepholePattern` objects,
each containing a TableGen rule, into a GlobalISel **GICombiner plugin**
(a shared library). The process is as follows:

1. Concatenates the user-provided `td_rule` strings to generate a `MyCombiner.td` file.
2. Generates minimalistic C++ source code for the `CombinerHelper` and the `PassPlugin`.
3. Creates a `CMakeLists.txt` and uses `cmake` + `ninja` to build the plugin,
   which depends on `llvm-tblgen` and `clang++`.
4. If the build is successful, it returns the path to the compiled plugin
   (`.so`/`.dylib`), which can then be used with `llc -load-pass-plugin`.

If the build fails (e.g., due to missing LLVM dependencies or syntax errors
in the patterns), it returns `None`. The calling code is responsible for
handling this fallback scenario.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional
import logging


from src.problems.llvm_mir_peephole.components import MIRPeepholePattern

__all__ = ["ensure_plugin"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _patterns_hash(patterns: List[MIRPeepholePattern]) -> str:
    """Compute a stable hash for a set of patterns.
    
    The hash is computed based on the pattern's name, its TableGen rule,
    and its targeted opcodes. This ensures that any semantic change,
    including adjustments to the targeted opcodes, triggers a rebuild.
    """
    pat_json = json.dumps([
        {
            "name": p.name,
            "td_rule": p.td_rule,
            "targeted_opcodes": sorted(p.targeted_opcodes),
        }
        for p in patterns
    ], sort_keys=True).encode()
    return hashlib.sha256(pat_json).hexdigest()[:16]


def _detect_llvm_cmake_dir() -> Optional[str]:
    """Try to locate LLVM's CMake config directory via llvm-config."""
    try:
        cmake_dir = (
            subprocess.check_output(["llvm-config", "--cmakedir"], text=True).strip()
        )
        if cmake_dir and os.path.isdir(cmake_dir):
            return cmake_dir
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def _which_bin(bin_name: str) -> Optional[str]:
    path = shutil.which(bin_name)
    return path if path and os.path.isfile(path) else None


# ---------------------------------------------------------------------------
# LLVM capability detection
# ---------------------------------------------------------------------------


def _llvm_supports_pass_plugins() -> bool:
    """Detect whether the current LLVM toolchain was built with LLVM_ENABLE_PLUGINS=ON.

    It works by attempting to run `opt -load-pass-plugin` and `llc -load-pass-plugin`.
    If the error message contains 'pass plugins are not supported', it assumes
    plugins are not supported.
    """
    
    def check_support(bin_name: str) -> bool:
        bin_path = _which_bin(bin_name)
        if not bin_path:
            # Conservatively assume no support if the tool is not found.
            return False
        try:
            # Trigger an error with a non-existent plugin path.
            cmd = [bin_path, "-load-pass-plugin", "nonexistent.so"]
            if bin_name == "opt":
                cmd.extend(["-passes=no-op-function", "-"])
            else: # llc
                cmd.extend(["-o", os.devnull, "-"]) # llc requires an input and output

            proc = subprocess.run(
                cmd, input="", text=True, capture_output=True, check=False
            )
            out = proc.stdout.lower() + proc.stderr.lower()

            if "pass plugins are not supported" in out or "not compiled with llvm_enable_plugins" in out:
                return False
            # Other errors (e.g., .so not found) indicate that the argument was accepted,
            # so we assume plugins are supported.
            return True
        except Exception:
            return False
            
    return check_support("opt") and check_support("llc")


# ---------------------------------------------------------------------------
# TD / CPP Template Generation
# ---------------------------------------------------------------------------


def _generate_td(patterns: List[MIRPeepholePattern], td_path: Path) -> bool:
    """Generate a `.td` file that wraps user-provided rules into a valid GICombiner."""

    # 1) Filter for valid rules
    valid_patterns = [p for p in patterns if p.td_rule and p.td_rule.strip()]
    if not valid_patterns:
        return False

    rule_defs = [p.td_rule.strip() for p in valid_patterns]
    rule_names = [p.name for p in valid_patterns]

    # 2) Assemble the file content
    td_lines: List[str] = []
    td_lines.append('include "llvm/Target/GlobalISel/Combine.td"')
    td_lines.append('')

    # Add user-defined rules
    td_lines.extend(rule_defs)
    td_lines.append('')

    # Add the boilerplate for the RuleSet and Combiner to ensure tblgen can parse it.
    td_lines.append('def MyCombinerRuleSet : GICombinerRuleSet {')
    td_lines.append('  let Name = "MyCombinerRuleSet";')
    if rule_names:
        td_lines.append('  let Rules = [')
        td_lines.append(",\n".join(f"    {n}" for n in rule_names))
        td_lines.append('  ];')
    td_lines.append('}')
    td_lines.append('')
    td_lines.append('def MyCombiner : GICombiner<"MyCombinerHelper"> {')
    td_lines.append('  let RuleSets = [MyCombinerRuleSet];')
    td_lines.append('}')

    td_path.write_text("\n".join(td_lines))
    return True


def _generate_cpp(src_dir: Path):
    """Generate the C++ helper / pass source with basic LLVM version compatibility."""

    cpp_code = r"""#include "MyCombiner.inc"
#include "llvm/CodeGen/GlobalISel/GICombiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

namespace {
class MyCombinerHelper : public CombinerHelper {
public:
  MyCombinerHelper(const MachineFunction &MF, const TargetPassConfig &TPC,
                   const RegisterBankInfo &RBI, MachineDominatorTree *MDT)
      : CombinerHelper(MF, TPC, RBI, MDT) {}
};

class MyCombinerPass : public PassInfoMixin<MyCombinerPass> {
public:
  PreservedAnalyses run(MachineFunction &MF, MachineFunctionAnalysisManager &AM) {
#if LLVM_VERSION_MAJOR >= 18
    auto &TPC = AM.getResult<TargetMachineAnalysis>(MF);
#else
    auto &TPC = AM.getResult<TargetPassConfigAnalysis>(MF);
#endif
    auto &RBI = *MF.getSubtarget().getRegBankInfo();
    MachineDominatorTree *MDT = nullptr;
    MyCombinerHelper Helper(MF, TPC, RBI, MDT);
    return runCombinerOnMachineFunction(MF, AM, Helper, MyCombinerRuleSet);
  }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "peephole-plugin", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, MachineFunctionPassManager &PM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "my-micombiner") {
                    PM.addPass(MyCombinerPass());
                    return true;
                  }
                  return false;
                });
          }};
}
"""
    (src_dir / "MyCombinerPass.cpp").write_text(cpp_code)


def _generate_cmake(src_dir: Path):
    cmake_txt = r"""cmake_minimum_required(VERSION 3.13)
project(peephole_combiner LANGUAGES CXX)

find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})

include(HandleLLVMOptions)
include(AddLLVM)

set(LLVM_TARGET_DEFINITIONS MyCombiner.td)
tablegen(LLVM MyCombiner.inc -gen-global-isel-combiner -combiners "MyCombinerHelper")
add_public_tablegen_target(MyCombinerTableGen)

add_llvm_library(peephole_plugin SHARED MyCombinerPass.cpp MyCombiner.inc)

set_target_properties(peephole_plugin PROPERTIES PREFIX "")
"""
    (src_dir / "CMakeLists.txt").write_text(cmake_txt)


# ---------------------------------------------------------------------------
# Build entry
# ---------------------------------------------------------------------------


def _build_plugin(src_dir: Path, build_dir: Path) -> bool:
    """Invoke cmake & ninja to build the plugin."""

    # 0) Check for basic tool availability
    if not _llvm_supports_pass_plugins():
        logger.warning("[plugin_builder] The current LLVM build does not support plugins (LLVM_ENABLE_PLUGINS=ON is required).")
        return False

    cmake_bin = _which_bin("cmake")
    ninja_bin = _which_bin("ninja") or _which_bin("ninja-build")
    if not cmake_bin or not ninja_bin:
        return False

    llvm_cmake_dir = _detect_llvm_cmake_dir()
    if not llvm_cmake_dir:
        return False

    env = os.environ.copy()
    env["LLVM_CMAKE_DIR"] = llvm_cmake_dir

    try:
        subprocess.run([
            cmake_bin,
            "-G",
            "Ninja",
            f"-DLLVM_DIR={llvm_cmake_dir}",
            str(src_dir),
        ], cwd=build_dir, env=env, capture_output=True, text=True, check=True)

        ninja_cmd = [ninja_bin]
        ninja_jobs = os.getenv("LLVM_MIR_PEEPHOLE_NINJA_JOBS") or os.getenv("NINJA_NUM_JOBS")
        if ninja_jobs:
            ninja_cmd.extend(["-j", ninja_jobs])
        ninja_cmd.append("peephole_plugin")

        subprocess.run(ninja_cmd, cwd=build_dir, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"ERROR: Plugin build failed.\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_plugin(patterns: List[MIRPeepholePattern]) -> Optional[str]:
    """Ensure there is a compiled plugin (.so/.dylib) for these patterns.

    1. Computes a hash of the patterns. If a cached plugin exists, returns its path.
    2. If not, it generates the TD/C++/CMake files and attempts to build the plugin.
    3. If successful, it caches the plugin and returns its path; otherwise, returns None.
    """

    if not patterns:
        return None

    cache_dir = os.getenv("LLVM_MIR_PEEPHOLE_PLUGIN_CACHE")
    if cache_dir:
        cache_root = Path(cache_dir)
    else:
        cache_root = Path(tempfile.gettempdir()) / "llvm_mir_peephole_plugins"
    cache_root.mkdir(parents=True, exist_ok=True)

    plugin_hash = _patterns_hash(patterns)

    # Shared library suffix
    system = platform.system()
    if system == "Darwin":
        lib_ext = ".dylib"
    elif system == "Windows":
        lib_ext = ".dll"
    else:
        lib_ext = ".so"

    plugin_file = cache_root / f"peephole_{plugin_hash}{lib_ext}"
    if plugin_file.is_file():
        return str(plugin_file)

    # --------------- Generate source -----------------
    src_dir = cache_root / f"src_{plugin_hash}"
    build_dir = cache_root / f"build_{plugin_hash}"
    build_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    td_ok = _generate_td(patterns, src_dir / "MyCombiner.td")
    if not td_ok:
        return None  # No valid patterns to build

    _generate_cpp(src_dir)
    _generate_cmake(src_dir)

    # --------------- Build ---------------------------
    success = _build_plugin(src_dir, build_dir)
    if not success:
        return None

    # Determine the name of the built artifact based on the platform.
    # The PREFIX is set to "" in CMakeLists.txt.
    if system == "Windows":
        built_lib = build_dir / "peephole_plugin.dll"
    else:
        built_lib = build_dir / f"peephole_plugin{lib_ext}"

    if built_lib.is_file():
        shutil.copy2(built_lib, plugin_file)
        return str(plugin_file)

    return None 