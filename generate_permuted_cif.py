import re
import sys
import os
import numpy as np
import copy
from pathlib import Path

# --- CONSTANTS ---
paramErrCode = 3
fileNotExistErrCode = 4
tol = 1e-3

# --- ARGUMENT PARSING ---
if len(sys.argv) < 2:
    print("Wrong number of arguments.", file=sys.stderr)
    print("Usage: python generate_permuted_cif.py input.cif [permutation_order]", file=sys.stderr)
    print("Example: python generate_permuted_cif.py POSCAR.cif 1,2,0", file=sys.stderr)
    exit(paramErrCode)

cif_file_name = str(sys.argv[1])

# Default permutation
perm_arg = "1,2,0"
if len(sys.argv) >= 3:
    perm_arg = sys.argv[2]

if not os.path.exists(cif_file_name):
    print(f"File not found: {cif_file_name}", file=sys.stderr)
    exit(fileNotExistErrCode)

# --- PARSING HELPERS (Reused from your script) ---

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
                    pass
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
                continue
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

                # Capture extra columns (occupancy, etc) if they exist, just in case
                atom_data['extra'] = []
                for i, part in enumerate(parts):
                    if i not in [idx_lbl, idx_sym, idx_x, idx_y, idx_z]:
                        atom_data['extra'].append(part)

                atoms.append(atom_data)
            except ValueError:
                pass
    return atoms

def parse_symmetry_metadata(file_path):
    lines = remove_comments_and_empty_lines_cif(file_path)
    metadata = {}
    for line in lines:
        if line.startswith("data_"):
            metadata['data_name'] = line[5:].strip()
            continue
        parts = line.split(maxsplit=1)
        if len(parts) < 2: continue
        key = parts[0]
        value = parts[1].strip("'\"")
        if key.startswith("_symmetry_") or key.startswith("_audit_"):
            metadata[key] = value
    return metadata

# --- PERMUTATION LOGIC ---

def get_permutation_matrix(order):
    P = np.zeros((3, 3))
    for new_idx, old_idx in enumerate(order):
        P[new_idx, old_idx] = 1.0
    return P

def apply_permutation_to_cell(cell_params, order):
    lengths = [cell_params['a'], cell_params['b'], cell_params['c']]
    new_lengths = [lengths[i] for i in order]

    def get_angle_between_old_indices(i1, i2):
        s = sorted([i1, i2])
        if s == [1, 2]: return cell_params['alpha']
        if s == [0, 2]: return cell_params['beta']
        if s == [0, 1]: return cell_params['gamma']
        return 90.0

    new_alpha = get_angle_between_old_indices(order[1], order[2])
    new_beta  = get_angle_between_old_indices(order[0], order[2])
    new_gamma = get_angle_between_old_indices(order[0], order[1])

    return {
        'a': new_lengths[0], 'b': new_lengths[1], 'c': new_lengths[2],
        'alpha': new_alpha, 'beta': new_beta, 'gamma': new_gamma
    }

def apply_permutation_to_atoms(atoms_list, order):
    new_atoms = copy.deepcopy(atoms_list)
    for atom in new_atoms:
        old_coords = [atom['x'], atom['y'], atom['z']]
        new_coords = [old_coords[i] for i in order]
        atom['x'] = new_coords[0]
        atom['y'] = new_coords[1]
        atom['z'] = new_coords[2]
    return new_atoms

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

def apply_permutation_to_symmetry_matrix(mat_3x4, P):
    R = mat_3x4[0:3, 0:3]
    T = mat_3x4[0:3, 3]
    R_new = P @ R @ P.T
    T_new = P @ T
    new_mat = np.zeros((3, 4))
    new_mat[0:3, 0:3] = R_new
    new_mat[0:3, 3] = T_new
    return new_mat

# --- OUTPUT GENERATION ---

def matrix_to_xyz_string(mat_3x4):
    """Converts a 3x4 matrix back to 'x, y, z' string format for CIF."""
    labels = ['x', 'y', 'z']
    res_strs = []

    for i in range(3): # Row (New X, New Y, New Z)
        terms = []
        # Rotation part
        for j in range(3): # Col (Old x, Old y, Old z)
            val = mat_3x4[i, j]
            if abs(val) > 1e-5:
                # Determine sign
                if val > 0: sign = "+"
                else: sign = "-"

                # If it's the first term, suppress +
                if not terms and sign == "+": sign = ""

                if abs(abs(val) - 1.0) < 1e-5:
                    terms.append(f"{sign}{labels[j]}")
                else:
                    terms.append(f"{val:.4f}{labels[j]}")

        # Translation part
        trans = mat_3x4[i, 3]

        # Normalize translation to [0, 1) if desired, but usually CIF keeps original shifts
        # trans = trans % 1.0

        if abs(trans) > 1e-5:
            # Try to convert to common fractions
            frac_found = False
            for den in [2, 3, 4, 6]:
                if abs((trans * den) - round(trans * den)) < 1e-5:
                    num = int(round(trans * den))
                    if num != 0:
                        sign = "+" if num > 0 else "-"
                        terms.append(f"{sign}{abs(num)}/{den}")
                        frac_found = True
                        break
            if not frac_found:
                sign = "+" if trans > 0 else "-"
                terms.append(f"{sign}{abs(trans):.5f}")

        if not terms:
            res_strs.append("0")
        else:
            # Join and clean up leading +
            full_str = "".join(terms)
            if full_str.startswith("+"): full_str = full_str[1:]
            res_strs.append(full_str)

    return ",".join(res_strs)

def write_cif(filename, metadata, cell, atoms, symmetry_ops):
    with open(filename, 'w') as f:
        # Header
        name = metadata.get('data_name', 'permuted_system')
        f.write(f"data_{name}\n")
        f.write("_audit_creation_method   'Permutation Script'\n")

        # Metadata (Space group, etc)
        # Note: Permuting axes might technically change the H-M symbol setting
        # (e.g. P21/c vs P21/n), but we keep the label and rely on explicit operators.
        for k, v in metadata.items():
            if k != 'data_name':
                f.write(f"{k:<30} '{v}'\n")

        f.write("\n")

        # Cell Parameters
        f.write(f"_cell_length_a                  {cell['a']:.5f}\n")
        f.write(f"_cell_length_b                  {cell['b']:.5f}\n")
        f.write(f"_cell_length_c                  {cell['c']:.5f}\n")
        f.write(f"_cell_angle_alpha               {cell['alpha']:.5f}\n")
        f.write(f"_cell_angle_beta                {cell['beta']:.5f}\n")
        f.write(f"_cell_angle_gamma               {cell['gamma']:.5f}\n")

        f.write("\n")

        # Symmetry Operations
        if symmetry_ops:
            f.write("loop_\n")
            f.write("_symmetry_equiv_pos_as_xyz\n")
            for op in symmetry_ops:
                f.write(f"  {op}\n")
            f.write("\n")

        # Atoms
        f.write("loop_\n")
        f.write("_atom_site_label\n")
        f.write("_atom_site_type_symbol\n")
        f.write("_atom_site_fract_x\n")
        f.write("_atom_site_fract_y\n")
        f.write("_atom_site_fract_z\n")
        # Add extra columns if they were preserved, but for safety usually just xyz is enough

        for atom in atoms:
            f.write(f"  {atom['label']:<6} {atom['symbol']:<4} {atom['x']:>10.5f} {atom['y']:>10.5f} {atom['z']:>10.5f}\n")

# --- EXECUTION ---

try:
    permutation_order = [int(x) for x in perm_arg.split(',')]
    if len(permutation_order) != 3 or set(permutation_order) != {0, 1, 2}:
        raise ValueError
except:
    print("Error: Permutation order must be 3 unique integers 0,1,2", file=sys.stderr)
    exit(paramErrCode)

print(f"Reading: {cif_file_name}")
print(f"Applying permutation: {permutation_order}")

# 1. Parse
metadata = parse_symmetry_metadata(cif_file_name)
cell = parse_cell_parameters(cif_file_name)
atoms = parse_atom_sites(cif_file_name)
raw_sym_ops = parse_cif_contents_xyz_transformations(cif_file_name)

# 2. Permute Cell
new_cell = apply_permutation_to_cell(cell, permutation_order)

# 3. Permute Atoms
new_atoms = apply_permutation_to_atoms(atoms, permutation_order)

# 4. Permute Symmetry
P = get_permutation_matrix(permutation_order)
new_sym_ops_strs = []
for op_list in raw_sym_ops:
    mat = parse_transformation_one_row_to_matrix(op_list)
    new_mat = apply_permutation_to_symmetry_matrix(mat, P)
    new_str = matrix_to_xyz_string(new_mat)
    new_sym_ops_strs.append(new_str)

# 5. Write
out_name = f"{os.path.splitext(cif_file_name)[0]}_permuted.cif"
write_cif(out_name, metadata, new_cell, new_atoms, new_sym_ops_strs)

print(f"Successfully created: {out_name}")