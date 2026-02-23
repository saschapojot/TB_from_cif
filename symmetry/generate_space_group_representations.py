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
    lattice_basis=np.array(lattice_basis)
    conf_file_path=parsed_config["config_file_path"]
    conf_file_dir=Path(conf_file_path).parent
    symmetry_matrices_file_name_path=str(conf_file_dir/symmetry_matrices_file_name)
    # Load the pickle file
    with open(symmetry_matrices_file_name_path, 'rb') as f:
        symmetry_matrices = pickle.load(f)




except FileNotFoundError:
    print(f"Error: Symmetry matrices file not found at {symmetry_matrices_file_name_path}", file=sys.stderr)
    exit(file_err_code)
except KeyError as e:
    print(f"Error: Missing required key in configuration: {e}", file=sys.stderr)
    exit(key_err_code)
except Exception as e:
    print(f"An unexpected error occurred: {e}", file=sys.stderr)
    exit(val_err_code)


# ==============================================================================
# Define coordinate transformation functions
# ==============================================================================
def space_group_to_cartesian_basis(symmetry_matrices,lattice_basis):
    """
    in .cif file, the symmetry_matrices are compatible with the lattice_basis
    :param symmetry_matrices:
    :param lattice_basis:
    :return:
    """
    AT = lattice_basis.T  # Transpose for column-vector representation
    AT_inv = np.linalg.inv(AT)
    space_group_matrices=[]
    for key,value in symmetry_matrices.items():
        space_group_matrices.append(value)
    space_group_matrices=np.array(space_group_matrices)
    num_operators = len(space_group_matrices)
    # print(space_group_matrices,file=sys.stdout)
    space_group_matrices_cartesian = np.zeros((num_operators, 3, 4), dtype=float)
    for j in range(num_operators):
        # Transform rotation/reflection part
        space_group_matrices_cartesian[j, :, 0:3] = AT @ space_group_matrices[j, :, 0:3] @ AT_inv
        # Transform translation part
        space_group_matrices_cartesian[j, :, 3] = AT @ space_group_matrices[j, :, 3]

    return space_group_matrices_cartesian


# ==============================================================================
#  Define orbital representation functions
# ==============================================================================
def space_group_representation_D_orbitals(R):
    """
    Compute how a symmetry operation acts on d orbitals

    Original function: GetSymD(R) in cd/SymGroup.py

    The d orbitals transform as quadratic functions of coordinates:
    d_xy, d_yz, d_xz, d_(x²-y²), d_(3z²-r²)

    This function computes the 5x5 representation matrix showing how
    the rotation R transforms the d orbital basis.

    :param R: Linear part of space group operation (3x3 rotation matrix) in Cartesian basis
    :return: Representation matrix (5x5) for d orbitals
    """
    [[R_11, R_12, R_13], [R_21, R_22, R_23], [R_31, R_32, R_33]] = R
    RD = np.zeros((5, 5))
    sr3 = np.sqrt(3)

    # Row 0: d_xy orbital transformation
    RD[0, 0] = R_11*R_22 + R_12*R_21
    RD[0, 1] = R_21*R_32 + R_22*R_31
    RD[0, 2] = R_11*R_32 + R_12*R_31
    RD[0, 3] = 2*R_11*R_12 + R_31*R_32
    RD[0, 4] = sr3*R_31*R_32

    # Row 1: d_yz orbital transformation
    RD[1, 0] = R_12*R_23 + R_13*R_22
    RD[1, 1] = R_22*R_33 + R_23*R_32
    RD[1, 2] = R_12*R_33 + R_13*R_32
    RD[1, 3] = 2*R_12*R_13 + R_32*R_33
    RD[1, 4] = sr3*R_32*R_33

    # Row 2: d_zx orbital transformation
    RD[2, 0] = R_11*R_23 + R_13*R_21
    RD[2, 1] = R_21*R_33 + R_23*R_31
    RD[2, 2] = R_11*R_33 + R_13*R_31
    RD[2, 3] = 2*R_11*R_13 + R_31*R_33
    RD[2, 4] = sr3*R_31*R_33

    # Row 3: d_(x²-y²) orbital transformation
    RD[3, 0] = R_11*R_21 - R_12*R_22
    RD[3, 1] = R_21*R_31 - R_22*R_32
    RD[3, 2] = R_11*R_31 - R_12*R_32
    RD[3, 3] = (R_11**2 - R_12**2) + 1/2*(R_31**2 - R_32**2)
    RD[3, 4] = sr3/2*(R_31**2 - R_32**2)

    # Row 4: d_(3z²-r²) orbital transformation
    RD[4, 0] = 1/sr3*(2*R_13*R_23 - R_11*R_21 - R_12*R_22)
    RD[4, 1] = 1/sr3*(2*R_23*R_33 - R_21*R_31 - R_22*R_32)
    RD[4, 2] = 1/sr3*(2*R_13*R_33 - R_11*R_31 - R_12*R_32)
    RD[4, 3] = 1/sr3*(2*R_13**2 - R_11**2 - R_12**2) + 1/sr3/2*(2*R_33**2 - R_31**2 - R_32**2)
    RD[4, 4] = 1/2*(2*R_33**2 - R_31**2 - R_32**2)

    return RD.T


def space_group_representation_F_orbitals(R):
    """
    Compute how a symmetry operation acts on f orbitals

    Original function: GetSymF(R) in cd/SymGroup.py

    The f orbitals transform as cubic functions of coordinates:
    fz³, fxz², fyz², fxyz, fz(x²-y²), fx(x²-3y²), fy(3x²-y²)

    This function computes the 7x7 representation matrix showing how
    the rotation R transforms the f orbital basis.

    :param R: Linear part of space group operation (3x3 rotation matrix) in Cartesian basis
    :return: Representation matrix (7x7) for f orbitals
    """
    sr3 = np.sqrt(3)
    sr5 = np.sqrt(5)
    sr15 = np.sqrt(15)

    # Define cubic monomials: x³, y³, z³, x²y, xy², x²z, xz², y²z, yz², xyz
    x1x2x3 = np.array([
        [1, 1, 1],  # x³
        [2, 2, 2],  # y³
        [3, 3, 3],  # z³
        [1, 1, 2],  # x²y
        [1, 2, 2],  # xy²
        [1, 1, 3],  # x²z
        [1, 3, 3],  # xz²
        [2, 2, 3],  # y²z
        [2, 3, 3],  # yz²
        [1, 2, 3]   # xyz
    ], int)

    # Compute how rotation R acts on cubic monomials
    # Rx1x2x3[i,j] = coefficient of monomial j in transformed monomial i
    Rx1x2x3 = np.zeros((10, 10))
    for i in range(10):
        n1, n2, n3 = x1x2x3[i]
        # Transform each cubic monomial by applying R to each factor
        Rx1x2x3[i, 0] = R[1-1, n1-1] * R[1-1, n2-1] * R[1-1, n3-1]  # x³
        Rx1x2x3[i, 1] = R[2-1, n1-1] * R[2-1, n2-1] * R[2-1, n3-1]  # y³
        Rx1x2x3[i, 2] = R[3-1, n1-1] * R[3-1, n2-1] * R[3-1, n3-1]  # z³
        # x²y (sum of all permutations)
        Rx1x2x3[i, 3] = (R[1-1, n1-1] * R[1-1, n2-1] * R[2-1, n3-1] +
                         R[1-1, n1-1] * R[2-1, n2-1] * R[1-1, n3-1] +
                         R[2-1, n1-1] * R[1-1, n2-1] * R[1-1, n3-1])
        # xy² (sum of all permutations)
        Rx1x2x3[i, 4] = (R[1-1, n1-1] * R[2-1, n2-1] * R[2-1, n3-1] +
                         R[2-1, n1-1] * R[2-1, n2-1] * R[1-1, n3-1] +
                         R[2-1, n1-1] * R[1-1, n2-1] * R[2-1, n3-1])
        # x²z (sum of all permutations)
        Rx1x2x3[i, 5] = (R[1-1, n1-1] * R[1-1, n2-1] * R[3-1, n3-1] +
                         R[1-1, n1-1] * R[3-1, n2-1] * R[1-1, n3-1] +
                         R[3-1, n1-1] * R[1-1, n2-1] * R[1-1, n3-1])
        # xz² (sum of all permutations)
        Rx1x2x3[i, 6] = (R[1-1, n1-1] * R[3-1, n2-1] * R[3-1, n3-1] +
                         R[3-1, n1-1] * R[3-1, n2-1] * R[1-1, n3-1] +
                         R[3-1, n1-1] * R[1-1, n2-1] * R[3-1, n3-1])
        # y²z (sum of all permutations)
        Rx1x2x3[i, 7] = (R[2-1, n1-1] * R[2-1, n2-1] * R[3-1, n3-1] +
                         R[2-1, n1-1] * R[3-1, n2-1] * R[2-1, n3-1] +
                         R[3-1, n1-1] * R[2-1, n2-1] * R[2-1, n3-1])
        # yz² (sum of all permutations)
        Rx1x2x3[i, 8] = (R[2-1, n1-1] * R[3-1, n2-1] * R[3-1, n3-1] +
                         R[3-1, n1-1] * R[3-1, n2-1] * R[2-1, n3-1] +
                         R[3-1, n1-1] * R[2-1, n2-1] * R[3-1, n3-1])
        # xyz (sum of all 6 permutations)
        Rx1x2x3[i, 9] = (R[1-1, n1-1] * R[2-1, n2-1] * R[3-1, n3-1] +
                         R[1-1, n1-1] * R[3-1, n2-1] * R[2-1, n3-1] +
                         R[2-1, n1-1] * R[1-1, n2-1] * R[3-1, n3-1] +
                         R[2-1, n1-1] * R[3-1, n2-1] * R[1-1, n3-1] +
                         R[3-1, n1-1] * R[1-1, n2-1] * R[2-1, n3-1] +
                         R[3-1, n1-1] * R[2-1, n2-1] * R[1-1, n3-1])

    # Matrix to express f orbitals as linear combinations of cubic monomials
    # Rows: fz³, fxz², fyz², fxyz, fz(x²-y²), fx(x²-3y²), fy(3x²-y²)
    # Columns: x³, y³, z³, x²y, xy², x²z, xz², y²z, yz², xyz
    F = np.array([
        [       0,        0,   1/sr15,        0,        0, -3/2/sr15,        0, -3/2/sr15,        0,        0],  # fz³
        [-1/2/sr5,        0,        0,        0, -1/2/sr5,        0,    2/sr5,        0,        0,        0],  # fxz²
        [       0, -1/2/sr5,        0, -1/2/sr5,        0,        0,        0,        0,    2/sr5,        0],  # fyz²
        [       0,        0,        0,        0,        0,        0,        0,        0,        0,        1],  # fxyz
        [       0,        0,        0,        0,        0,      1/2,        0,     -1/2,        0,        0],  # fz(x²-y²)
        [ 1/2/sr3,        0,        0,        0,   -sr3/2,        0,        0,        0,        0,        0],  # fx(x²-3y²)
        [       0, -1/2/sr3,        0,    sr3/2,        0,        0,        0,        0,        0,        0]   # fy(3x²-y²)
    ])

    # Transform f orbitals: FR = F @ Rx1x2x3
    FR = F @ Rx1x2x3  # Shape: (7, 10)

    # Matrix to convert back from cubic monomials to f orbitals
    # Rows: fz³, fxz², fyz², fxyz, fz(x²-y²), fx(x²-3y²), fy(3x²-y²)
    # Columns: x³, y³, z³, x²y, xy², x²z, xz², y²z, yz², xyz
    CF = np.array([
        [     0,      0,   sr15,      0,      0,      0,      0,      0,      0,      0],  # fz³
        [     0,      0,      0,      0,      0,      0,  sr5/2,      0,      0,      0],  # fxz²
        [     0,      0,      0,      0,      0,      0,      0,      0,  sr5/2,      0],  # fyz²
        [     0,      0,      0,      0,      0,      0,      0,      0,      0,      1],  # fxyz
        [     0,      0,      3,      0,      0,      2,      0,      0,      0,      0],  # fz(x²-y²)
        [ 2*sr3,      0,      0,      0,      0,      0,  sr3/2,      0,      0,      0],  # fx(x²-3y²)
        [     0, -2*sr3,      0,      0,      0,      0,      0,      0, -sr3/2,      0]   # fy(3x²-y²)
    ])

    # Final representation matrix for f orbitals
    RF = FR @ CF.T
    return RF.T

def space_group_representation_orbitals_all(space_group_matrices_cartesian):
    """
    Compute space group representations for all atomic orbital types
    For each symmetry operation in the space group, compute how it transforms:
    - s orbitals (scalar, trivial representation)
    - p orbitals (3D vector: px, py, pz)
    - d orbitals (5D: dxy, dyz, dxz, d(x²-y²), d(3z²-r²))
    - f orbitals (7D: fz³, fxz², fyz², fxyz, fz(x²-y²), fx(x²-3y²), fy(3x²-y²))

    Args:
        space_group_matrices_cartesian:  Space group matrices (affine) under Cartesian basis

    Returns: List of representations [repr_s, repr_p, repr_d, repr_f]

    """
    num_matrices, _, _ = space_group_matrices_cartesian.shape
    # s orbitals: spherically symmetric, trivial representation (all 1's)
    repr_s = np.ones((num_matrices, 1, 1))

    # p orbitals: transform as vectors (px, py, pz)
    # Use the rotation part of the space group matrices
    repr_p = copy.deepcopy(space_group_matrices_cartesian[:, :3, :3])

    # d orbitals: 5x5 representation
    # Basis: dxy, dyz, dxz, d(x²-y²), d(3z²-r²)
    repr_d = np.zeros((num_matrices, 5, 5))
    for i in range(num_matrices):
        R = space_group_matrices_cartesian[i, :3, :3]
        repr_d[i] = space_group_representation_D_orbitals(R)

    # f orbitals: 7x7 representation
    # Basis: fz³, fxz², fyz², fxyz, fz(x²-y²), fx(x²-3y²), fy(3x²-y²)
    #TODO: check order of basis
    repr_f = np.zeros((num_matrices, 7, 7))
    for i in range(num_matrices):
        R = space_group_matrices_cartesian[i, :3, :3]
        repr_f[i] = space_group_representation_F_orbitals(R)

    repr_s_p_d_f = [repr_s, repr_p, repr_d, repr_f]
    return repr_s_p_d_f


def subroutine_generate_all_representations(symmetry_matrices,lattice_basis):
    """

    Args:
        symmetry_matrices:
        lattice_basis:

    Returns:

    """
    space_group_matrices = []
    for key, value in symmetry_matrices.items():
        space_group_matrices.append(value)
    space_group_matrices = np.array(space_group_matrices)
    space_group_matrices_cartesian=space_group_to_cartesian_basis(symmetry_matrices,lattice_basis)
    repr_s_p_d_f = space_group_representation_orbitals_all(space_group_matrices_cartesian)
    # Create output dictionary with all computed representations
    space_group_representations = {
        # space group operations reads from .cif file
        "space_group_matrices":space_group_matrices.tolist(),
        # Space group matrices in Cartesian coordinates
        "space_group_matrices_cartesian": space_group_matrices_cartesian.tolist(),
        # Orbital representations (s, p, d, f)

        "repr_s_p_d_f": [
            repr_s_p_d_f[0].tolist(),  # s orbital representation
            repr_s_p_d_f[1].tolist(),  # p orbital representation
            repr_s_p_d_f[2].tolist(),  # d orbital representation
            repr_s_p_d_f[3].tolist()  # f orbital representation
        ]
    }
    print(json.dumps(space_group_representations, indent=2), file=sys.stdout)


subroutine_generate_all_representations(symmetry_matrices,lattice_basis)