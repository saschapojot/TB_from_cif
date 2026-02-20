import numpy as np
import sys
import json
import re
import copy
from pathlib import Path
import pickle

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from name_conventions import  symmetry_matrices_file_name
# ==============================================================================
# Space group representation computation script
# ==============================================================================
# This script computes space group representations for atomic orbitals
#  It transforms space group operations in  Cartesian basis (x, y, z coordinates)
# It also computes how symmetry operations act on atomic orbitals (s, p, d, f)
# Exit codes for different error conditions
json_err_code = 4   # JSON parsing error
key_err_code = 5    # Required key missing from configuration
val_err_code = 6    # Invalid value in configuration
file_err_code = 7   # File not found or IO error


# ==============================================================================
# STEP 1: Read and parse JSON input from stdin
# ==============================================================================

try:
    config_json = sys.stdin.read()
    parsed_config = json.loads(config_json)

except json.JSONDecodeError as e:
    print(f"Error parsing JSON input: {e}", file=sys.stderr)
    exit(json_err_code)


# ==============================================================================
# STEP 2: Extract space group configuration data
# ==============================================================================
# Note: All operations assume primitive cell basis unless otherwise specified
try:
    # unit cell lattice basis vectors (3x3 matrix)
    # Each row is a lattice vector in Cartesian coordinates\
    lattice_basis=parsed_config['lattice_basis']
    conf_file_path=parsed_config["config_file_path"]
    conf_file_dir=Path(conf_file_path).parent
    symmetry_matrices_file_name_path=str(conf_file_dir/symmetry_matrices_file_name)
    # Load the pickle file
    with open(symmetry_matrices_file_name_path, 'rb') as f:
        symmetry_matrices = pickle.load(f)

    print(symmetry_matrices,file=sys.stdout)


except FileNotFoundError:
    print(f"Error: Symmetry matrices file not found at {symmetry_matrices_file_name_path}", file=sys.stderr)
    exit(file_err_code)
except KeyError as e:
    print(f"Error: Missing required key in configuration: {e}", file=sys.stderr)
    exit(key_err_code)
except Exception as e:
    print(f"An unexpected error occurred: {e}", file=sys.stderr)
    exit(val_err_code)