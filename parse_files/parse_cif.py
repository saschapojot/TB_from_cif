import re
import sys
import json
import os
import pickle
import numpy as np
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from name_conventions import symmetry_matrices_file_name
#this script parse .cif file to get atom positions, symmetry operations
#.cif file is generated from material studio

paramErrCode = 3         # Wrong command-line parameters
fileNotExistErrCode = 4  # Configuration file doesn't exist
if len(sys.argv) != 2:
    print("wrong number of arguments.", file=sys.stderr)
    print("usage: python parse_cif.py /path/to/xxx.cif", file=sys.stderr)
    exit(paramErrCode)

cif_file_name = str(sys.argv[1])
# Check if configuration file exists
if not os.path.exists(cif_file_name):
    print(f"file not found: {cif_file_name}", file=sys.stderr)
    exit(fileNotExistErrCode)


def remove_comments_and_empty_lines_cif(file):
    """
    Remove comments and empty lines from a CIF-like configuration file.

    - Comments start with # and continue to end of line.
    - # inside single (') or double (") quotes are preserved.
    - Empty lines (or lines with only whitespace) are removed.

    :param file: conf file path
    :return: list of cleaned lines
    """
    with open(file, "r") as fptr:
        lines = fptr.readlines()

    linesToReturn = []
    # Regex explanation:
    # 1. (["\'])  -> Capture group 1: Match a quote (single or double)
    # 2. (?:      -> Non-capturing group for content inside quotes
    #    \\.      -> Match escaped characters (like \")
    #    |        -> OR
    #    [^\\]    -> Match any character that isn't a backslash
    #    )*?      -> Match as few times as possible (lazy)
    # 3. \1       -> Match the closing quote (same as group 1)
    # 4. |        -> OR
    # 5. (#.*$)   -> Capture group 2: Match a real comment (starts with #, goes to end)
    # This pattern finds quoted strings OR comments.
    # We will use a callback to decide what to do.
    pattern = re.compile(r'(["\'])(?:\\.|[^\\])*?\1|(#.*$)')
    for oneLine in lines:
        # We use a callback function for re.sub
        # If group 2 (the comment) exists, replace it with empty string.
        # If group 1 (the quote) matches, return the whole match (preserve it).
        def replace_func(match):
            if match.group(2):
                return ""  # It's a comment, remove it
            else:
                return match.group(0)  # It's a quoted string, keep it
        # Apply the regex
        cleaned_line = pattern.sub(replace_func, oneLine).strip()
        # Only add non-empty lines
        if cleaned_line:
            linesToReturn.append(cleaned_line)

    return linesToReturn

coef_pattern = re.compile(r"([+-]?)\s*([xyz])", re.IGNORECASE)

# Matches: optional sign, optional space, number (integer, decimal, or fraction)
# Example: "+1/2", "-0.5", "1/4", "+1"
translation_pattern = re.compile(r"([+-]?)\s*(\d+(?:[./]\d+)?)")

# The Master Regex
# Group 1,2: Variable (sign, char)
# Group 3,4: Number (sign, val)
term_pattern = re.compile(r"([+-]?)\s*([xyz])|([+-]?)\s*(\d+(?:[./]\d+)?)", re.IGNORECASE)

def parse_single_expression(expression_str):
    """
    Parses a string like '-x+1/2' or 'x-y' into numerical coefficients.
    :param expression_str:
    :return: (coeff_x, coeff_y, coeff_z, translation)
    """
    # Initialize values
    c_x = 0.0
    c_y = 0.0
    c_z = 0.0
    trans = 0.0
    # Find all matches in the string
    for match in term_pattern.finditer(expression_str):
        # --- CASE 1: It's a Variable: x, y, z ---
        if match.group(2):
            sign_str = match.group(1)
            variable = match.group(2).lower()
            # Determine value: +1 or -1
            value = -1.0 if sign_str == '-' else 1.0
            if variable == 'x':
                c_x += value
            elif variable == 'y':
                c_y += value
            elif variable == 'z':
                c_z += value
        # --- CASE 2: It's a Number: Translation ---
        elif match.group(4):
            sign_str = match.group(3)
            number_str = match.group(4)
            # Handle fraction conversion (e.g., "1/2" -> 0.5)
            if '/' in number_str:
                num, den = number_str.split('/')
                val = float(num) / float(den)
            else:
                val = float(number_str)
            # Apply sign
            if sign_str == '-':
                val = -val
            trans += val

    return c_x, c_y, c_z, trans



# # --- TEST EXAMPLES ---
# examples = [
#     "-x+1/2",       # Standard
#     "1/2-x",        # Non-standard (Translation first)
#     "0.5+y",        # Non-standard (Decimal + Variable)
#     "z",            # Simple
#     "-y-x+1/4"      # Complex
# ]
#
# print(f"{'Expression':<15} | {'x':<4} {'y':<4} {'z':<4} {'Trans':<5}",file=sys.stdout)
# print("-" * 45,file=sys.stdout)
#
# for ex in examples:
#     cx, cy, cz, tr = parse_single_expression(ex)
#     print(f"{ex:<15} | {cx:<4} {cy:<4} {cz:<4} {tr:<5}",file=sys.stdout)


def parse_cif_contents_xyz_transformations(file):
    """
    Parses the CIF file to extract symmetry operations.
    Returns a list of symmetry operations.
    Each operation is a list of 3 dictionaries (for x', y', z' components).
    """
    # Get cleaned lines from file
    lines = remove_comments_and_empty_lines_cif(file)

    symmetry_operations = []

    # State variables for parsing
    in_loop = False
    current_loop_headers = []

    # We iterate through the lines to find the symmetry loop
    for line in lines:
        line = line.strip()

        # 1. Check for start of a loop
        if line == "loop_":
            in_loop = True
            current_loop_headers = []
            continue

        # 2. Check for headers (lines starting with underscore)
        if line.startswith("_"):
            if in_loop:
                current_loop_headers.append(line)
            else:
                # If we hit a header but aren't in a loop, reset loop state
                in_loop = False
                current_loop_headers = []
            continue

        # 3. Process Data Lines
        # We only care if we are inside a loop that contains the symmetry header
        if in_loop and "_symmetry_equiv_pos_as_xyz" in current_loop_headers:
            # CIF symmetry strings are usually comma-separated, e.g., "-x, y+1/2, -z"
            # Sometimes they might be quoted like 'x, y, z'. Remove quotes first.
            clean_line = line.replace("'", "").replace('"', "")

            # Split into the three components (x, y, z)
            # We assume comma separation based on standard CIF and the provided example
            raw_components = clean_line.split(",")

            if len(raw_components) == 3:
                op_matrix = []
                for comp in raw_components:
                    # Use the existing helper function to parse specific string (e.g., "-x")
                    cx, cy, cz, tr = parse_single_expression(comp.strip())

                    # Store as a dictionary for clarity
                    op_matrix.append({
                        "raw_string": comp.strip(),
                        "cx": cx,
                        "cy": cy,
                        "cz": cz,
                        "trans": tr
                    })
                symmetry_operations.append(op_matrix)
            else:
                # Handle edge case where split failed or format is unexpected
                print(f"Warning: Could not parse symmetry line: {line}", file=sys.stderr)

    return symmetry_operations


def parse_cell_parameters(file):
    """
    Parses unit cell lengths and angles from a CIF file.
    Returns a dictionary containing a, b, c, alpha, beta, gamma.
    Raises ValueError if any parameter is missing.
    """
    # Get cleaned lines
    lines = remove_comments_and_empty_lines_cif(file)

    # Initialize dictionary with None to detect missing values later
    cell_params = {
        'a': None,
        'b': None,
        'c': None,
        'alpha': None,
        'beta': None,
        'gamma': None
    }

    # Map CIF keywords to our dictionary keys
    keyword_map = {
        '_cell_length_a': 'a',
        '_cell_length_b': 'b',
        '_cell_length_c': 'c',
        '_cell_angle_alpha': 'alpha',
        '_cell_angle_beta': 'beta',
        '_cell_angle_gamma': 'gamma'
    }

    for line in lines:
        # Split line by whitespace
        parts = line.strip().split()

        # We expect at least "KEY VALUE" (2 parts)
        if len(parts) >= 2:
            key = parts[0]
            val_str = parts[1]

            if key in keyword_map:
                # Handle uncertainty notation often found in CIFs (e.g., "12.34(5)")
                clean_val_str = val_str.split('(')[0]

                try:
                    value = float(clean_val_str)
                    target_key = keyword_map[key]
                    cell_params[target_key] = value
                except ValueError:
                    # We raise an error here immediately if the number format is wrong
                    raise ValueError(f"Error parsing numerical value for {key}: '{val_str}' in file {file}")

    # --- CHECK FOR MISSING VALUES ---
    missing_keys = [k for k, v in cell_params.items() if v is None]

    if missing_keys:
        # Raise an error stopping execution if parameters are missing
        raise ValueError(f"Missing required cell parameters in {file}: {', '.join(missing_keys)}")

    return cell_params


def parse_atom_sites(file):
    """
    Parses the atom site loop to extract fractional coordinates and labels.
    Returns a list of dictionaries.

    Required headers:
      - _atom_site_label
      - _atom_site_type_symbol
      - _atom_site_fract_x
      - _atom_site_fract_y
      - _atom_site_fract_z

    Raises ValueError if:
      1. Any required header is missing.
      2. Parsing coordinates fails (non-numeric).
      3. No atoms are found in the file.
    """
    lines = remove_comments_and_empty_lines_cif(file)

    atoms = []

    # State variables
    in_loop = False
    loop_headers = []

    # Define mandatory headers
    required_headers = [
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z"
    ]

    for line in lines:
        line = line.strip()

        # 1. Detect Loop Start
        if line == "loop_":
            in_loop = True
            loop_headers = []
            continue

        # 2. Collect Headers
        if line.startswith("_"):
            if in_loop:
                loop_headers.append(line)
            else:
                in_loop = False
                loop_headers = []
            continue

        # 3. Process Data Lines
        # We identify the atom loop by checking if it contains one of our unique keys (e.g., fract_x)
        if in_loop and "_atom_site_fract_x" in loop_headers:

            # --- VALIDATION: Check for Missing Headers ---
            # Check if all required headers are present in the current loop
            missing_headers = [h for h in required_headers if h not in loop_headers]
            if missing_headers:
                raise ValueError(f"Missing mandatory headers in atom site loop: {', '.join(missing_headers)}")

            parts = line.split()
            atom_data = {}

            try:
                # --- Map Headers to Indices ---
                idx_lbl = loop_headers.index("_atom_site_label")
                idx_sym = loop_headers.index("_atom_site_type_symbol")
                idx_x = loop_headers.index("_atom_site_fract_x")
                idx_y = loop_headers.index("_atom_site_fract_y")
                idx_z = loop_headers.index("_atom_site_fract_z")

                # --- Validate Column Count ---
                # Ensure the line has enough columns to cover the highest index we need
                max_idx = max(idx_lbl, idx_sym, idx_x, idx_y, idx_z)
                if len(parts) <= max_idx:
                    raise ValueError(f"Line has fewer columns than headers: '{line}'")

                # --- Extract Strings ---
                atom_data['label'] = parts[idx_lbl]
                atom_data['symbol'] = parts[idx_sym]

                # --- Extract Floats (Handle uncertainty like '0.123(4)') ---
                atom_data['x'] = float(parts[idx_x].split('(')[0])
                atom_data['y'] = float(parts[idx_y].split('(')[0])
                atom_data['z'] = float(parts[idx_z].split('(')[0])

                # Optional: Extract Occupancy and Uiso if present (not mandatory for error)
                if "_atom_site_occupancy" in loop_headers:
                    idx_occ = loop_headers.index("_atom_site_occupancy")
                    if idx_occ < len(parts):
                        atom_data['occupancy'] = float(parts[idx_occ].split('(')[0])

                if "_atom_site_U_iso_or_equiv" in loop_headers:
                    idx_u = loop_headers.index("_atom_site_U_iso_or_equiv")
                    if idx_u < len(parts):
                        atom_data['u_iso'] = float(parts[idx_u].split('(')[0])

                atoms.append(atom_data)

            except ValueError as e:
                # This catches float conversion errors
                raise ValueError(f"Error parsing atom data in line: '{line}'. Reason: {e}")

    # --- FINAL CHECK ---
    if not atoms:
        raise ValueError("No atom sites found. The file may be missing the atom loop or the loop is empty.")

    return atoms


def remove_comments_and_empty_lines_cif(file_path):
    """
    Helper function to read a file and strip comments (#) and empty lines.
    """
    clean_lines = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Remove comments (everything after #)
                line = line.split('#')[0].strip()
                if line:
                    clean_lines.append(line)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    return clean_lines


def parse_symmetry_metadata(file_path):
    """
    Parses the CIF file to extract:
    1. The Data Block Name (e.g., Te2W)
    2. Space Group Name (H-M)
    3. Int Tables Number
    4. Cell Setting
    """
    lines = remove_comments_and_empty_lines_cif(file_path)

    # Initialize dictionary with None
    metadata = {
        'data_name': None,
        'space_group_name_H_M': None,
        'int_tables_number': None,
        'cell_setting': None
    }

    for line in lines:
        # 1. Extract Data Block Name (starts with 'data_')
        if line.startswith("data_"):
            # Everything after "data_" is the name
            metadata['data_name'] = line[5:].strip()
            continue

        # For the other fields, we split the line into Key and Value.
        # maxsplit=1 ensures we only split on the first whitespace, preserving spaces in the value if any.
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue

        key = parts[0]
        value = parts[1]

        # Remove single quotes (') and double quotes (") from the value
        clean_value = value.strip("'\"")

        # 2. Extract Space Group
        # Note: The standard CIF tag is _symmetry_space_group_name_H-M
        if key == "_symmetry_space_group_name_H-M":
            metadata['space_group_name_H_M'] = clean_value

        # 3. Extract Int Tables Number
        elif key == "_symmetry_Int_Tables_number":
            try:
                metadata['int_tables_number'] = int(clean_value)
            except ValueError:
                metadata['int_tables_number'] = clean_value  # Keep as string if conversion fails

        # 4. Extract Cell Setting
        elif key == "_symmetry_cell_setting":
            metadata['cell_setting'] = clean_value

    return metadata


def metadata_to_key_value(metadata):
    """

    Args:
        metadata: output from parse_symmetry_metadata

    Returns: dictionary for constructing .conf file

    """
    out_dict={}
    for key, value in metadata.items():
        if key=="data_name":
            out_dict["name"]=value
        else:
            out_dict[key]=value

    return out_dict


def parse_transformation_one_expression_to_vector(one_transformation_dict):
    """

    Args:
        one_transformation_dict: a dict represents one expression in a row of symmetry transformations

    Returns:  a row of vector, length=4, first 3 numbers are coefficients for rotation matrix,
                the last number is for translation

    """
    ret_vec=np.zeros(4)
    for key,value in one_transformation_dict.items():
        if key=="cx":
            ret_vec[0]=value
        elif key=="cy":
            ret_vec[1]=value
        elif key=="cz":
            ret_vec[2]=value
        elif key=="trans":
            ret_vec[3]=value
    return ret_vec

def parse_transformation_one_row_to_matrix(a_row_of_transformation_list):
    """

    Args:
        a_row_of_transformation_list: a row containing 3 dict, representing a row of symmetry transformations
        in .cif file

    Returns: 3 by 4 symmetry operation matrix
    """
    mat=np.zeros((3,4))
    for row, dict_item in enumerate(a_row_of_transformation_list):
        ret_vec=parse_transformation_one_expression_to_vector(dict_item)
        mat[row,:]=ret_vec
    return mat

def generate_all_symmetry_transformation_matrices(cif_file_path):
    """
    this function write the symmetry transformation matrices to pkl file
    Args:
        file_path: .cif file

    Returns:

    """
    symmetry_operations=parse_cif_contents_xyz_transformations(cif_file_path)
    out_dict={}
    for counter, list_item in enumerate(symmetry_operations):
        mat=parse_transformation_one_row_to_matrix(list_item)
        key=f"mat{counter}"
        out_dict[key]=mat
    cif_dir = Path(cif_file_path).resolve().parent
    out_pickle_file_name=str(cif_dir/symmetry_matrices_file_name)
    # --- SAVE TO PICKLE ---
    with open(out_pickle_file_name, 'wb') as f:
        pickle.dump(out_dict, f)
    return out_pickle_file_name





# metadata=parse_symmetry_metadata(cif_file_name)
print(generate_all_symmetry_transformation_matrices(cif_file_name),file=sys.stdout)

