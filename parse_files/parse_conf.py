import re
import sys
import json
import os

# ==============================================================================
# Configuration parser for tight-binding model input files
# ==============================================================================
# This script parses .conf files containing lattice, atom, and orbital information

# Exit codes for different error types
fmtErrStr = "format error: "
formatErrCode = 1        # Format/syntax errors in conf file
valueMissingCode = 2     # Required values are missing
paramErrCode = 3         # Wrong command-line parameters
fileNotExistErrCode = 4  # Configuration file doesn't exist
blankErrCode=5# value not given in .conf file

# ==============================================================================
# STEP 0: Validate command-line arguments
# ==============================================================================
if len(sys.argv) != 2:
    print("wrong number of arguments.", file=sys.stderr)
    print("usage: python parse_conf.py /path/to/xxx.conf", file=sys.stderr)
    exit(paramErrCode)


conf_file = sys.argv[1]

# Check if configuration file exists
if not os.path.exists(conf_file):
    print(f"file not found: {conf_file}", file=sys.stderr)
    exit(fileNotExistErrCode)

# ==============================================================================
# STEP 1: Define regex patterns for parsing
# ==============================================================================
# General key=value pattern
key_value_pattern = r'^([^=\s]+)\s*=\s*([^=]+)\s*$'

# Pattern for floating point numbers (including scientific notation)
float_pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"

# Pattern for atom type definitions: AtomSymbol = orbital1, orbital2, ...
# Example: O=2px,2py,2pz
# Modified to remove the count and semicolon requirement
atom_orbital_pattern = r'^([A-Za-z]+\d*)\s*=\s*([1-7](?:s|px|py|pz|dxy|dxz|dyz|dx2-y2|dz2|fxyz|fx3-3xy2|f3yx2-y3|fxz2|fyz2|fzx2-zy2|fz3)(?:\s*,\s*[1-7](?:s|px|py|pz|dxy|dxz|dyz|dx2-y2|dz2|fxyz|fx3-3xy2|f3yx2-y3|fxz2|fyz2|fzx2-zy2|fz3))*)\s*$'


# Pattern for system name
name_pattern = r'^name\s*=\s*([a-zA-Z0-9_-]+)\s*$'

# Pattern for dimensionality (1, 2 or 3)
dim_pattern = r"^dim\s*=\s*(\d+)\s*$"

# Pattern for directions to study: x, y, z (comma separated, no duplicates)
# The (?!.*([xyz]).*\1) part ensures uniqueness
directions_to_study_pattern = r"^directions_to_study\s*=\s*(?!.*([xyz]).*\1)([xyz](?:\s*,\s*[xyz])*)\s*$"

truncation_radius_pattern=rf"truncation_radius\s*=\s*({float_pattern})\s*"

# Pattern for number of Wyckoff position types
Wyckoff_type_num_pattern = r"^Wyckoff_type_num\s*=\s*(\d+)\s*$"

# Pattern for lattice basis vectors: v1 ; v2 ; v3
# Example: lattice_basis = 1.0,0.0,0.0 ; 0.0,1.0,0.0 ; 0.0,0.0,1.0
lattice_basis_pattern = rf'^lattice_basis\s*=\s*({float_pattern}\s*,\s*{float_pattern}\s*,\s*{float_pattern})(?:\s*;\s*({float_pattern}\s*,\s*{float_pattern}\s*,\s*{float_pattern})){{2}}\s*$'

# Pattern for space group number (1-230)
space_group_pattern = r"^space_group\s*=\s*(\d+)\s*$"

space_group_name_H_M_pattern= r"^space_group_name_H_M\s*=\s*(.+?)\s*$"

cell_setting_pattern=r"^cell_setting\s*=\s*(.+?)\s*$"

# Pattern for space group origin in fractional coordinates
space_group_origin_pattern = rf"^space_group_origin\s*=\s*({float_pattern})\s*,\s*({float_pattern})\s*,\s*({float_pattern})\s*$"
# Pattern for spin flag (true/false)
spin_pattern = r'^spin\s*=\s*((?i:true|false))\s*$'

# ==============================================================================
# STEP 2: Define helper function to clean file contents
# ==============================================================================
def removeCommentsAndEmptyLines(file):
    """
    Remove comments and empty lines from configuration file

    Comments start with # and continue to end of line
    Empty lines (or lines with only whitespace) are removed

    :param file: conf file path
    :return: list of cleaned lines (comments and empty lines removed)
    """
    with open(file, "r") as fptr:
        lines = fptr.readlines()

    linesToReturn = []
    for oneLine in lines:
        # Remove comments (everything after #) and strip whitespace
        oneLine = re.sub(r'#.*$', '', oneLine).strip()

        # Only add non-empty lines
        if oneLine:
            linesToReturn.append(oneLine)

    return linesToReturn


# ==============================================================================
# STEP 3: Define main parsing function
# ==============================================================================
def subroutine_parseConfContents(file):
    """
     Parse configuration file contents into structured dictionary
    :param file:
    :return:
    """
    # Get cleaned lines from file
    linesWithCommentsRemoved = removeCommentsAndEmptyLines(file)
    # Initialize result dictionary with all expected fields
    config = {
        'config_file_path': os.path.abspath(file),  # Store full absolute path
        'name': '',  # System name
        'dim': '',  # Dimensionality (2 or 3)
        "directions_to_study":"",
        'spin': '',  # Spin consideration (true/false)
        'truncation_radius': '',
        'Wyckoff_type_num': '',  # Total number of Wyckoff position types
        'lattice_basis': '',  # Lattice basis vectors (3x3 matrix)
        'space_group': '',  # Space group number
        "space_group_name_H_M":'',# Space group name
        "space_group_origin":'',# origin
        'Wyckoff_position_types': {},  # Dictionary: Wyckoff_position_types -> {orbitals}
        'Wyckoff_positions': []  # List of atom positions with types
    }
    # Parse each line
    for oneLine in linesWithCommentsRemoved:
        # Check if line matches key=value format
        matchLine = re.match(key_value_pattern, oneLine)
        if matchLine:
            # ==========================================
            # Parse system name
            # ==========================================
            name_match = re.match(name_pattern, oneLine)
            if name_match:
                config['name'] = name_match.group(1)
                continue
            # ==========================================
            # Parse dimensionality (1D, 2D or 3D)
            # ==========================================
            dim_match = re.match(dim_pattern, oneLine)
            if dim_match:
                config['dim'] = int(dim_match.group(1))
                continue
            directions_to_study_match=re.match(directions_to_study_pattern,oneLine)
            if directions_to_study_match:
                # Get the raw string (e.g., "x, y")
                # We use group(2) because group(1) is inside the negative lookahead
                raw_val = directions_to_study_match.group(2)
                # Split by comma and remove whitespace to create a list: ['x', 'y']
                config['directions_to_study'] = sorted([d.strip() for d in raw_val.split(',')])
                continue

            # ==========================================
            # Parse spin flag
            # ==========================================
            spin_match = re.match(spin_pattern, oneLine)
            if spin_match:
                config['spin'] = spin_match.group(1)
                continue

            # ==========================================
            # Parse truncation radius
            # ==========================================
            match_truncation_radius = re.match(truncation_radius_pattern, oneLine)
            if match_truncation_radius:
                config['truncation_radius'] = float(match_truncation_radius.group(1))
                continue

            # ==========================================
            # Parse number of Wyckoff types
            # ==========================================
            match_wyckoff_num = re.match(Wyckoff_type_num_pattern, oneLine)
            if match_wyckoff_num:
                config['Wyckoff_type_num'] = int(match_wyckoff_num.group(1))
                continue

            # ==========================================
            # Parse lattice basis vectors
            # Format: v1x,v1y,v1z ; v2x,v2y,v2z ; v3x,v3y,v3z
            # ==========================================
            match_lattice_basis = re.match(lattice_basis_pattern, oneLine)
            if match_lattice_basis:
                # Extract the full lattice basis value after the = sign
                full_value = oneLine.split('=')[1].strip()
                # Split into 3 vectors separated by semicolons
                vectors = []
                for vector in full_value.split(';'):
                    # Parse x,y,z coordinates for each vector
                    coords = [float(x.strip()) for x in vector.strip().split(',')]
                    vectors.append(coords)
                config['lattice_basis'] = vectors
                continue



    return config


# ==============================================================================
# STEP 4: Parse configuration and output as JSON
# ==============================================================================
# Parse the configuration file
parsed_config = subroutine_parseConfContents(conf_file)
# Output the parsed configuration as JSON to stdout
# This allows the data to be piped to other scripts
print(json.dumps(parsed_config, indent=2), file=sys.stdout)