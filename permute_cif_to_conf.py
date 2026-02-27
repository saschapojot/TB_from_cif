import re
import sys
import os
import pickle
import numpy as np
from pathlib import Path
import copy

# --- PATH SETUP ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Try importing name_conventions, fallback if not present for standalone usage
try:
    from name_conventions import symmetry_matrices_file_name
except ImportError:
    symmetry_matrices_file_name = "symmetry_matrices.pkl"

# --- CONSTANTS ---
paramErrCode = 3
fileNotExistErrCode = 4
tol = 1e-3

# --- ARGUMENT PARSING ---
if len(sys.argv) < 2:
    print("Wrong number of arguments.", file=sys.stderr)
    print("Usage: python parse_cif_permute.py /path/to/xxx.cif [permutation_order]", file=sys.stderr)
    print("Example: python parse_cif_permute.py file.cif 2,0,1", file=sys.stderr)
    print("  (2,0,1 means: Old Z -> New X, Old X -> New Y, Old Y -> New Z)", file=sys.stderr)
    exit(paramErrCode)

cif_file_name = str(sys.argv[1])

# Default permutation is Identity (0, 1, 2) -> (x, y, z)
perm_arg = "1,2,0"
if len(sys.argv) >= 3:
    perm_arg = sys.argv[2]

if not os.path.exists(cif_file_name):
    print(f"File not found: {cif_file_name}", file=sys.stderr)
    exit(fileNotExistErrCode)

# --- PERMUTATION HELPERS ---

def get_permutation_matrix(order):
    """
    Generates a 3x3 permutation matrix P based on the order list.
    If order is [2, 0, 1]:
    New X comes from Old Z (index 2)
    New Y comes from Old X (index 0)
    New Z comes from Old Y (index 1)
    """
    P = np.zeros((3, 3))
    for new_idx, old_idx in enumerate(order):
        P[new_idx, old_idx] = 1.0
    return P

def apply_permutation_to_cell(cell_params, order):
    """
    Permutes cell lengths and angles.
    order: list of 3 ints, e.g., [2, 0, 1]
    """
    # Map indices to keys
    idx_to_len = {0: 'a', 1: 'b', 2: 'c'}

    # Current lengths
    lengths = [cell_params['a'], cell_params['b'], cell_params['c']]

    # 1. Permute Lengths
    new_lengths = [lengths[i] for i in order]

    # 2. Permute Angles
    # Angles are defined by the axes they are *not* on.
    # alpha is angle between axes 1 and 2 (b and c)
    # beta is angle between axes 0 and 2 (a and c)
    # gamma is angle between axes 0 and 1 (a and b)

    def get_angle_between_old_indices(i1, i2):
        s = sorted([i1, i2])
        if s == [1, 2]: return cell_params['alpha']
        if s == [0, 2]: return cell_params['beta']
        if s == [0, 1]: return cell_params['gamma']
        return 90.0 # Should not happen if indices are distinct

    # The new alpha corresponds to the angle between the NEW axis 1 and NEW axis 2
    # New axis 1 comes from old axis order[1]
    # New axis 2 comes from old axis order[2]
    new_alpha = get_angle_between_old_indices(order[1], order[2])
    new_beta  = get_angle_between_old_indices(order[0], order[2])
    new_gamma = get_angle_between_old_indices(order[0], order[1])

    new_params = {
        'a': new_lengths[0],
        'b': new_lengths[1],
        'c': new_lengths[2],
        'alpha': new_alpha,
        'beta': new_beta,
        'gamma': new_gamma
    }
    return new_params

def apply_permutation_to_atoms(atoms_list, order):
    """
    Permutes x, y, z coordinates for each atom.
    """
    new_atoms = copy.deepcopy(atoms_list)
    for atom in new_atoms:
        old_coords = [atom['x'], atom['y'], atom['z']]
        new_coords = [old_coords[i] for i in order]
        atom['x'] = new_coords[0]
        atom['y'] = new_coords[1]
        atom['z'] = new_coords[2]
    return new_atoms

def apply_permutation_to_symmetry_matrix(mat_3x4, P):
    """
    Transforms a symmetry matrix M = [R | T] using permutation matrix P.
    New Coordinate x' = P * x
    Symmetry op: x_new = R * x_old + T
    Transformed: P*x_new = P*(R * P_inv * P * x_old + T)
                 x_new' = (P * R * P_inv) * x_old' + (P * T)
    Since P is orthogonal, P_inv = P_transpose.
    """
    R = mat_3x4[0:3, 0:3]
    T = mat_3x4[0:3, 3]

    # R_new = P @ R @ P.T
    R_new = P @ R @ P.T

    # T_new = P @ T
    T_new = P @ T

    # Reassemble
    new_mat = np.zeros((3, 4))
    new_mat[0:3, 0:3] = R_new
    new_mat[0:3, 3] = T_new

    return new_mat

# --- ORIGINAL PARSING FUNCTIONS ---

def remove_comments_and_empty_lines_cif(file):
    with open(file, "r") as fptr:
        lines = fptr.readlines()
    linesToReturn = []
    pattern = re.compile(r'(["\'])(?:\\.|[^\\])*?\1|(#.*$)')
    for oneLine in lines:
        def replace_func(match):
            if match.group(2): return ""
            else: return match.group(0)
        cleaned_line = pattern.sub(replace_func, oneLine).strip()
        if cleaned_line:
            linesToReturn.append(cleaned_line)
    return linesToReturn

coef_pattern = re.compile(r"([+-]?)\s*([xyz])", re.IGNORECASE)
term_pattern = re.compile(r"([+-]?)\s*([xyz])|([+-]?)\s*(\d+(?:[./]\d+)?)", re.IGNORECASE)

def parse_single_expression(expression_str):
    c_x = 0.0; c_y = 0.0; c_z = 0.0; trans = 0.0
    for match in term_pattern.finditer(expression_str):
        if match.group(2):
            sign_str = match.group(1)
            variable = match.group(2).lower()
            value = -1.0 if sign_str == '-' else 1.0
            if variable == 'x': c_x += value
            elif variable == 'y': c_y += value
            elif variable == 'z': c_z += value
        elif match.group(4):
            sign_str = match.group(3)
            number_str = match.group(4)
            if '/' in number_str:
                num, den = number_str.split('/')
                val = float(num) / float(den)
            else:
                val = float(number_str)
            if sign_str == '-': val = -val
            trans += val
    return c_x, c_y, c_z, trans

def parse_cif_contents_xyz_transformations(file):
    lines = remove_comments_and_empty_lines_cif(file)
    symmetry_operations = []
    in_loop = False
    current_loop_headers = []
    for line in lines:
        line = line.strip()
        if line == "loop_":
            in_loop = True
            current_loop_headers = []
            continue
        if line.startswith("_"):
            if in_loop: current_loop_headers.append(line)
            else: in_loop = False; current_loop_headers = []
            continue
        if in_loop and "_symmetry_equiv_pos_as_xyz" in current_loop_headers:
            clean_line = line.replace("'", "").replace('"', "")
            raw_components = clean_line.split(",")
            if len(raw_components) == 3:
                op_matrix = []
                for comp in raw_components:
                    cx, cy, cz, tr = parse_single_expression(comp.strip())
                    op_matrix.append({"cx": cx, "cy": cy, "cz": cz, "trans": tr})
                symmetry_operations.append(op_matrix)
    return symmetry_operations

def parse_cell_parameters(file):
    lines = remove_comments_and_empty_lines_cif(file)
    cell_params = {'a': None, 'b': None, 'c': None, 'alpha': None, 'beta': None, 'gamma': None}
    keyword_map = {
        '_cell_length_a': 'a', '_cell_length_b': 'b', '_cell_length_c': 'c',
        '_cell_angle_alpha': 'alpha', '_cell_angle_beta': 'beta', '_cell_angle_gamma': 'gamma'
    }
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            key = parts[0]
            val_str = parts[1]
            if key in keyword_map:
                clean_val_str = val_str.split('(')[0]
                try:
                    cell_params[keyword_map[key]] = float(clean_val_str)
                except ValueError:
                    raise ValueError(f"Error parsing {key}: '{val_str}'")
    if any(v is None for v in cell_params.values()):
        raise ValueError(f"Missing cell parameters in {file}")
    return cell_params

def parse_atom_sites(file):
    lines = remove_comments_and_empty_lines_cif(file)
    atoms = []
    in_loop = False
    loop_headers = []
    required_headers = ["_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]

    for line in lines:
        line = line.strip()
        if line == "loop_":
            in_loop = True; loop_headers = []; continue
        if line.startswith("_"):
            if in_loop: loop_headers.append(line)
            else: in_loop = False; loop_headers = []
            continue
        if in_loop and "_atom_site_fract_x" in loop_headers:
            if any(h not in loop_headers for h in required_headers):
                raise ValueError("Missing mandatory headers in atom site loop")
            parts = line.split()
            atom_data = {}
            try:
                idx_lbl = loop_headers.index("_atom_site_label")
                idx_sym = loop_headers.index("_atom_site_type_symbol")
                idx_x = loop_headers.index("_atom_site_fract_x")
                idx_y = loop_headers.index("_atom_site_fract_y")
                idx_z = loop_headers.index("_atom_site_fract_z")

                atom_data['label'] = parts[idx_lbl]
                atom_data['symbol'] = parts[idx_sym]
                atom_data['x'] = float(parts[idx_x].split('(')[0])
                atom_data['y'] = float(parts[idx_y].split('(')[0])
                atom_data['z'] = float(parts[idx_z].split('(')[0])
                atoms.append(atom_data)
            except ValueError as e:
                raise ValueError(f"Error parsing atom data: {e}")
    if not atoms: raise ValueError("No atom sites found.")
    return atoms

def parse_symmetry_metadata(file_path):
    lines = remove_comments_and_empty_lines_cif(file_path)
    metadata = {'data_name': None, 'space_group_name_H_M': None, 'int_tables_number': None, 'cell_setting': None}
    for line in lines:
        if line.startswith("data_"):
            metadata['data_name'] = line[5:].strip()
            continue
        parts = line.split(maxsplit=1)
        if len(parts) < 2: continue
        key = parts[0]; value = parts[1].strip("'\"")
        if key == "_symmetry_space_group_name_H-M": metadata['space_group_name_H_M'] = value
        elif key == "_symmetry_Int_Tables_number":
            try: metadata['int_tables_number'] = int(value)
            except: metadata['int_tables_number'] = value
        elif key == "_symmetry_cell_setting": metadata['cell_setting'] = value
    return metadata

def parse_transformation_one_expression_to_vector(one_transformation_dict):
    ret_vec=np.zeros(4)
    ret_vec[0] = one_transformation_dict.get("cx", 0)
    ret_vec[1] = one_transformation_dict.get("cy", 0)
    ret_vec[2] = one_transformation_dict.get("cz", 0)
    ret_vec[3] = one_transformation_dict.get("trans", 0)
    return ret_vec

def parse_transformation_one_row_to_matrix(a_row_of_transformation_list):
    mat=np.zeros((3,4))
    for row, dict_item in enumerate(a_row_of_transformation_list):
        mat[row,:] = parse_transformation_one_expression_to_vector(dict_item)
    return mat

def generate_unit_cell_basis(cell_params, tol):
    a = cell_params["a"]; b = cell_params["b"]; c = cell_params["c"]
    alpha_rad = np.radians(cell_params["alpha"])
    beta_rad = np.radians(cell_params["beta"])
    gamma_rad = np.radians(cell_params["gamma"])

    v0_row_vec = np.array([a, 0, 0])
    v1_row_vec = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0])

    v2_0 = c * np.cos(beta_rad)
    frac = (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
    v2_1 = c * frac
    term_under_sqrt = np.sin(beta_rad) ** 2 - frac ** 2
    if term_under_sqrt < 1e-9:
        raise ValueError("Invalid angular configuration (volume ~0)")
    v2_2 = c * np.sqrt(term_under_sqrt)

    basis_mat = np.array([v0_row_vec, v1_row_vec, [v2_0, v2_1, v2_2]])
    basis_mat[np.abs(basis_mat) < tol] = 0.0
    return basis_mat

def rename_labels(atoms_list):
    atoms_copy = copy.deepcopy(atoms_list)
    symbol_counts = {}
    for atom in atoms_copy:
        sym = atom["symbol"]
        if sym not in symbol_counts: symbol_counts[sym] = 0
        atom["original_label"] = atom["label"]
        atom["label"] = f"{sym}{symbol_counts[sym]}"
        symbol_counts[sym] += 1
    return atoms_copy

# --- MAIN SUBROUTINES WITH PERMUTATION LOGIC ---

def subroutine_generate_all_symmetry_transformation_matrices(cif_file_path, perm_order):
    """
    Generates symmetry matrices, applies permutation, and saves to pickle.
    """
    symmetry_operations = parse_cif_contents_xyz_transformations(cif_file_path)

    # Generate Permutation Matrix P
    P = get_permutation_matrix(perm_order)

    out_dict = {}
    for counter, list_item in enumerate(symmetry_operations):
        # 1. Parse original matrix from CIF strings
        original_mat = parse_transformation_one_row_to_matrix(list_item)

        # 2. Apply Permutation: M_new = P * M_old (conceptually adjusted for R and T)
        permuted_mat = apply_permutation_to_symmetry_matrix(original_mat, P)

        key = f"mat{counter}"
        out_dict[key] = permuted_mat

    cif_dir = Path(cif_file_path).resolve().parent
    out_pickle_file_name = str(cif_dir / symmetry_matrices_file_name)

    with open(out_pickle_file_name, 'wb') as f:
        pickle.dump(out_dict, f)
    print(f"Generated symmetry matrices at: {out_pickle_file_name}")
    return out_pickle_file_name

def subroutine_generate_conf_file(cif_file_name, tol, perm_order):
    """
    Generates .conf file with permuted cell and atoms.
    """
    metadata = parse_symmetry_metadata(cif_file_name)
    data_name = metadata["data_name"]
    space_group_name_H_M = metadata["space_group_name_H_M"]
    int_tables_number = metadata["int_tables_number"]
    cell_setting = metadata["cell_setting"]

    # 1. Parse and Permute Atoms
    atoms_list = parse_atom_sites(cif_file_name)
    atoms_list_permuted = apply_permutation_to_atoms(atoms_list, perm_order)
    atoms_list_renamed = rename_labels(atoms_list_permuted)
    Wyckoff_position_num = len(atoms_list)

    # 2. Parse and Permute Cell
    cell_params = parse_cell_parameters(cif_file_name)
    cell_params_permuted = apply_permutation_to_cell(cell_params, perm_order)

    # 3. Generate Basis from Permuted Cell
    basis_mat = generate_unit_cell_basis(cell_params_permuted, tol)
    v0, v1, v2 = basis_mat
    basis_str = f"{v0[0]}, {v0[1]}, {v0[2]}; {v1[0]}, {v1[1]}, {v1[2]}; {v2[0]}, {v2[1]}, {v2[2]}"

    text_list = [
        f"#This is the configuration file for {data_name} computations\n",
        f"#Permutation applied: {perm_order} (Indices of old axes mapped to new X, Y, Z)\n",
        "#the format is key=value\n",
        "\n",
        "#name of the system\n",
        f"name={data_name}\n",
        "\n",
        "#dimension of system\n",
        "dim=\n",
        "\n",
        "#directions to study, available directions x,y,z\n",
        "directions_to_study=\n",
        "\n",
        "#whether spin is considered\n",
        "spin=False\n",
        "\n",
        "truncation_radius=\n",
        "\n",
        f"lattice_basis={basis_str}\n",
        "\n",
        f"space_group={int_tables_number}\n",
        "\n",
        f"space_group_origin=0,0,0\n"
        "\n",
        f"space_group_name_H_M={space_group_name_H_M}\n",
        "\n",
        f"cell_setting={cell_setting}\n",
        "\n",
        f"Wyckoff_position_num={Wyckoff_position_num}\n",
        "\n",
    ]

    for dict_item in atoms_list_renamed:
        label = dict_item["label"]
        x = dict_item["x"]
        y = dict_item["y"]
        z = dict_item["z"]

        str0 = f"#Wyckoff position label {label}, input orbitals\n"
        str1 = f"{label}_orbitals=\n"
        str2 = f"#one position of {label}, coefficients  (fractional coordinates)\n"
        str3 = f"{label}_position_coefs={x}, {y}, {z}\n"
        str_list = ["\n", str0, str1, str2, str3]
        text_list.extend(str_list)

    out_dir = Path(cif_file_name).resolve().parent
    out_conf_file = str(out_dir / f"{data_name}.conf")
    with open(out_conf_file, 'w', encoding='utf-8') as f:
        f.writelines(text_list)
        print(f"Successfully generated configuration file: {out_conf_file}")

# --- EXECUTION ---

# Parse the permutation string (e.g., "2,0,1") into a list of integers
try:
    permutation_order = [int(x) for x in perm_arg.split(',')]
    if len(permutation_order) != 3 or set(permutation_order) != {0, 1, 2}:
        raise ValueError
except:
    print("Error: Permutation order must be 3 unique integers 0,1,2 separated by commas (e.g., 2,0,1)", file=sys.stderr)
    exit(paramErrCode)

print(f"Applying permutation order: {permutation_order}")
print(f"  New X axis <--- Old Axis {permutation_order[0]}")
print(f"  New Y axis <--- Old Axis {permutation_order[1]}")
print(f"  New Z axis <--- Old Axis {permutation_order[2]}")

subroutine_generate_conf_file(cif_file_name, tol, permutation_order)
subroutine_generate_all_symmetry_transformation_matrices(cif_file_name, permutation_order)