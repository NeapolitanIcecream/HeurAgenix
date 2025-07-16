"""
Utilities for loading and caching ISA meta information such as
TargetInstrInfo, TargetRegisterInfo, and MCInstrDesc.

This information can be generated as a JSON file using the following command:
    llvm-tblgen --dump-json -I $(LLVM_SRC_ROOT)/llvm/lib/Target/X86 \
        -gen-target-desc $(LLVM_SRC_ROOT)/llvm/lib/Target/X86/X86.td > x86_isa.json

The current implementation loads this data by:
1. Reading the file path from the `LLVM_ISA_JSON` environment variable.
2. Returning an empty dictionary if the variable is not set or the file is invalid.

Downstream logic can then consume this dictionary to extract the required fields
for statistical analysis and problem state representation.
"""

import json
import os
from functools import lru_cache
from typing import Dict


@lru_cache(maxsize=1)
def load_isa_metadata() -> Dict:
    """Load ISA meta information from JSON file specified by LLVM_ISA_JSON env."""
    json_path = os.getenv("LLVM_ISA_JSON")
    if json_path and os.path.isfile(json_path):
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    # fallback: empty dict
    return {} 