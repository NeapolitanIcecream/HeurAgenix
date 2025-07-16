"""llvm_utils.py

Utilities for robust interaction with LLVM, specifically for analyzing LLVM IR.
This module provides a reliable way to get instruction counts and opcode
distributions by compiling and using a dedicated LLVM analysis pass, avoiding
the fragility of text-based parsing.
"""

import os
import subprocess
import json
import tempfile
from pathlib import Path
import shutil
import hashlib
from functools import lru_cache
from typing import Dict, Optional
import logging

# ---------------------------------------------------------------------------
# C++ and CMake Templates for the Analysis Pass
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

_ANALYSIS_PASS_CPP = r"""
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Function.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <map>
#include <string>

using namespace llvm;

namespace {

struct MIRStatsAnalysis : public PassInfoMixin<MIRStatsAnalysis> {
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MAM) {
    long long instruction_count = 0;
    std::map<std::string, int> opcode_counts;

    const TargetSubtargetInfo &STI = MF.getSubtarget();
    const TargetInstrInfo *TII = STI.getInstrInfo();

    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        instruction_count++;
        opcode_counts[TII->getName(MI.getOpcode())]++;
      }
    }

    json::Object Root;
    Root["total_instructions"] = instruction_count;
    json::Object Opcodes;
    for (const auto &Pair : opcode_counts) {
      Opcodes[Pair.first] = Pair.second;
    }
    Root["opcode_distribution"] = json::Value(std::move(Opcodes));

    errs() << "HEURAGENIX_STATS_OUTPUT: " << json::Value(std::move(Root)) << "\n";

    return PreservedAnalyses::all();
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "MIRStatsPlugin", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, MachineFunctionPassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "mir-stats") {
                    MPM.addPass(MIRStatsAnalysis());
                    return true;
                  }
                  return false;
                });
          }};
}
"""

_ANALYSIS_PASS_CMAKE = r"""
cmake_minimum_required(VERSION 3.13)
project(mir_stats_plugin LANGUAGES CXX)

find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})

include(HandleLLVMOptions)
include(AddLLVM)

add_llvm_library(MIRStatsPlugin MODULE SHARED MIRStatsAnalysis.cpp)

# Needed for plugins
set_target_properties(MIRStatsPlugin PROPERTIES
  FOLDER "LLVM/Plugins"
  LINKER_LANGUAGE CXX)
"""

# ---------------------------------------------------------------------------
# Helper functions from plugin_builder (could be refactored into a common place)
# ---------------------------------------------------------------------------

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
# Analysis Pass Builder
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _ensure_analysis_plugin() -> Optional[str]:
    """
    Ensures the analysis pass is compiled and cached for performance.
    If the build fails, it logs an error and returns None.
    """
    cache_root = Path(tempfile.gettempdir()) / "heuragenix_llvm_plugins"
    cache_root.mkdir(parents=True, exist_ok=True)

    plugin_hash = hashlib.sha256(_ANALYSIS_PASS_CPP.encode()).hexdigest()[:16]
    
    system = os.name
    lib_ext = ".dll" if system == 'nt' else ".dylib" if system == 'darwin' else ".so"

    plugin_file = cache_root / f"lib_mir_stats_{plugin_hash}{lib_ext}"
    if plugin_file.is_file():
        return str(plugin_file)

    src_dir = cache_root / f"src_{plugin_hash}"
    build_dir = cache_root / f"build_{plugin_hash}"
    src_dir.mkdir(exist_ok=True)
    build_dir.mkdir(exist_ok=True)

    (src_dir / "MIRStatsAnalysis.cpp").write_text(_ANALYSIS_PASS_CPP)
    (src_dir / "CMakeLists.txt").write_text(_ANALYSIS_PASS_CMAKE)

    cmake_bin = _which_bin("cmake")
    if not cmake_bin: return None
    ninja_bin = _which_bin("ninja") or _which_bin("ninja-build")
    if not ninja_bin: return None
    llvm_cmake_dir = _detect_llvm_cmake_dir()
    if not llvm_cmake_dir: return None

    try:
        env = os.environ.copy()
        env["LLVM_CMAKE_DIR"] = llvm_cmake_dir
        
        subprocess.check_call(
            [cmake_bin, "-G", "Ninja", f"-DLLVM_DIR={llvm_cmake_dir}", str(src_dir)],
            cwd=build_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        subprocess.check_call(
            [ninja_bin, "MIRStatsPlugin"],
            cwd=build_dir, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

        built_lib = build_dir / f"MIRStatsPlugin{lib_ext}"
        if built_lib.is_file():
            shutil.copy2(built_lib, plugin_file)
            return str(plugin_file)
    except subprocess.CalledProcessError as e:
        # Capture and print build errors for easier debugging
        logger.error(f"ERROR: Failed to build LLVM analysis plugin.\n{e.stderr.decode()}")
    return None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1024)
def get_ir_stats(ir_code: str, mtriple: str = "x86_64") -> Optional[Dict]:
    """
    Analyzes LLVM IR code to get accurate instruction counts and opcode distribution
    by running a custom analysis pass via `llc`.
    
    The pass prints its JSON output to stderr, which is then parsed here.
    If `llc` fails or the plugin cannot be built, it returns None.
    """
    plugin_path = _ensure_analysis_plugin()
    llc_bin = _which_bin("llc")
    
    if not plugin_path or not llc_bin:
        return None

    tmp_ir_path = None
    try:
        # Use a context manager for the temporary file
        with tempfile.NamedTemporaryFile("w", suffix=".ll", delete=False) as tmp_ir:
            tmp_ir.write(ir_code)
            tmp_ir_path = tmp_ir.name

        cmd = [
            llc_bin,
            "-mtriple", mtriple,
            f"-load-pass-plugin={plugin_path}",
            "-passes=mir-stats",
            "-o", os.devnull, # We don't need the output artifact
            tmp_ir_path,
        ]
        
        # The pass prints its JSON output to stderr.
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            # This can happen for valid IR that the pass doesn't handle, so log as warning.
            logger.warning(f"llc failed during stats collection:\n{result.stderr}")
            return None

        # Aggregate results from multiple functions if necessary
        total_instructions = 0
        opcode_distribution: Dict[str, int] = {}
        
        for line in result.stderr.splitlines():
            if line.startswith("HEURAGENIX_STATS_OUTPUT: "):
                json_str = line.removeprefix("HEURAGENIX_STATS_OUTPUT: ")
                data = json.loads(json_str)
                total_instructions += data.get("total_instructions", 0)
                for opcode, count in data.get("opcode_distribution", {}).items():
                    opcode_distribution[opcode] = opcode_distribution.get(opcode, 0) + count
        
        if total_instructions == 0 and not opcode_distribution:
             return None # No stats were collected

        return {
            "total_instructions": total_instructions,
            "opcode_distribution": opcode_distribution,
        }

    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Failed to get IR stats: {e}")
        return None
    finally:
        if tmp_ir_path and os.path.exists(tmp_ir_path):
            os.remove(tmp_ir_path)
