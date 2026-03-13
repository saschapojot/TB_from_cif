import numpy as np
import sys
import pandas as pd
from multiprocessing import Pool, cpu_count
from datetime import datetime
import matplotlib.pyplot as plt
if len(sys.argv)!=2:
    print("wrong number of arguments")


#python Hamiltonian_exact.py xxx.csv

csv_file_name=str(sys.argv[1])
dfstr=pd.read_csv(csv_file_name,sep='\t')
oneRow=dfstr.iloc[0,:]
name=str(oneRow["name"])
mu_p=float(oneRow["mu_p"])
mu_d=float(oneRow["mu_d"])
d1=float(oneRow["d1"])
d2=float(oneRow["d2"])
d3=float(oneRow["d3"])
d4=float(oneRow["d4"])
d5=float(oneRow["d5"])
d6=0
p1=float(oneRow["p1"])
p2=float(oneRow["p2"])
p3=float(oneRow["p3"])
p4=float(oneRow["p4"])
p5=0
p6=float(oneRow["p6"])
t1=float(oneRow["t1"])
t2=float(oneRow["t2"])
t3=0
t4=0
t5=0
t6=0
interpolate_point_num=200
print(f"name={name}")
print(f"mu_p={mu_p}")
print(f"mu_d={mu_d}")
print(f"d1={d1}")
print(f"d2={d2}")
print(f"d3={d3}")
print(f"d4={d4}")
print(f"d5={d5}")
print(f"d6={d6}")
print(f"p1={p1}")
print(f"p2={p2}")
print(f"p3={p3}")
print(f"p4={p4}")
print(f"p5={p5}")
print(f"p6={p6}")
print(f"t1={t1}")
print(f"t2={t2}")
print(f"t3={t3}")
print(f"t4={t4}")
print(f"t5={t5}")
print(f"t6={t6}")

def H00(kx,ky):
    part0=mu_d

    part1=2*(d2+2*d5*np.cos(ky))*np.cos(kx)+2*d4*np.cos(ky)

    return part0+part1

def H11(kx,ky):
    return H00(kx,ky)


def H22(kx,ky):
    part1=mu_p
    part2=2*(p1+2*p6*np.cos(ky))*np.cos(kx)
    part3=2*p3*np.cos(ky)
    return part1+part2+part3

def H33(kx,ky):
    return H22(kx,ky)

def H01(kx,ky):

    part1=1+np.exp(1j*kx)

    part2=d3+d1*np.exp(1j*ky)+d6*np.exp(-1j*ky)
    return part1*part2


def H02(kx,ky):
    part1=-2j*np.sin(kx)
    part2=t2+t4*np.exp(1j*ky)+t5*np.exp(-1j*ky)
    return part1*part2

def H03(kx,ky):
    part1=1-np.exp(1j*kx)

    part2=-t1+t3*np.exp(1j*ky)+t6*np.exp(-1j*ky)

    return part1*part2


def H12(kx,ky):
    part1=1-np.exp(-1j*kx)
    part2=t1-t3*np.exp(-1j*ky)-t6*np.exp(1j*ky)
    return part1*part2

def H13(kx,ky):
    part1=-2j*np.sin(kx)

    part2=t2+t4*np.exp(-1j*ky)+t5*np.exp(1j*ky)
    return part1*part2

def H23(kx,ky):
    part1=1+np.exp(1j*kx)
    part2=p2+p4*np.exp(-1j*ky)+p5*np.exp(1j*ky)
    return part1*part2

def H10(kx,ky):
    return np.conj(H01(kx,ky))

def H20(kx,ky):
    return np.conj(H02(kx,ky))

def H21(kx,ky):
    return np.conj(H12(kx,ky))

def H30(kx,ky):
    return np.conj(H03(kx,ky))
def  H31(kx,ky):
    return np.conj(H13(kx,ky))

def H32(kx,ky):
    return np.conj(H23(kx,ky))

def H_mat(kx,ky):
    val_H00=H00(kx,ky)
    val_H01=H01(kx,ky)
    val_H02=H02(kx,ky)
    val_H03=H03(kx,ky)

    val_H10=H10(kx,ky)
    val_H11=H11(kx,ky)
    val_H12=H12(kx,ky)
    val_H13=H13(kx,ky)

    val_H20=H20(kx,ky)
    val_H21=H21(kx,ky)
    val_H22=H22(kx,ky)
    val_H23=H23(kx,ky)

    val_H30=H30(kx,ky)
    val_H31=H31(kx,ky)
    val_H32=H32(kx,ky)
    val_H33=H33(kx,ky)

    mat=np.array([
        [val_H00,val_H01,val_H02,val_H03],
        [val_H10, val_H11, val_H12, val_H13],
        [val_H20, val_H21, val_H22, val_H23],
        [val_H30, val_H31, val_H32, val_H33]
    ])
    return mat

#
# def is_hermitian(matrix):
#     return np.allclose(matrix, matrix.conj().T)


#basis
a0=np.array([3.5186, 0.0, 0.0])
a1=np.array([0.0, 6.3023, 0.0])
a2=np.array([0.0, 0.0, 15.0058])

point_X={
    "label":"X",
    "coords":np.array([0.5,0,0])
}

point_Gamma={
    "label":"Gamma",
    "coords":np.array([0,0,0])
}

point_Y={
    "label":"Y",
    "coords":np.array([0,0.5,0])
}
point_M={
    "label":"M",
    "coords":np.array([0.5,0.5,0])
}

parsed_k_points=[point_X,point_Gamma,point_Y,point_M,point_Gamma]

# volume, may be signed if a0,a1,a2 do not have positive orientation
Omega=np.dot(a0,np.cross(a1,a2))
b0=2*np.pi*np.cross(a1,a2)/Omega

b1=2*np.pi*np.cross(a2,a0)/Omega

b2=2*np.pi*np.cross(a0,a1)/Omega


def generate_interpolation(point_start_frac, point_end_frac,BZ_basis_vectors,interpolate_point_num=15):
    # 1. Convert Fractional to Cartesian
    # We use zip to pair the coordinate component (u, v, w) with the basis vector (b0, b1, b2)
    # This automatically handles 1D, 2D, or 3D depending on the length of the inputs.
    start_cart = sum(c * b for c, b in zip(point_start_frac, BZ_basis_vectors))
    end_cart = sum(c * b for c, b in zip(point_end_frac, BZ_basis_vectors))



    # 2. Linear Interpolation
    # Create a parameter t going from 0 to 1
    t = np.linspace(0, 1, interpolate_point_num)
    # Vector from start to end
    vector_diff = end_cart - start_cart
    # Calculate path: Start + t * (End - Start)
    # np.outer allows us to multiply the shape (N,) t array by the shape (3,) vector
    # each row is an interpolated point
    interpolated_cart_coords = start_cart + np.outer(t, vector_diff)
    # 3. Calculate Distances
    # Euclidean distance of the full segment
    segment_length = np.linalg.norm(vector_diff)
    distances = t * segment_length

    return interpolated_cart_coords, distances

BZ_basis_vectors = [b0,b1]
all_coords = []
all_distances = []
high_symmetry_indices = []
high_symmetry_labels = []

cumulative_distance = 0.0
current_index_count = 0
# 2. Loop through consecutive pairs
for i in range(len(parsed_k_points) - 1):
    start_point = parsed_k_points[i]
    end_point = parsed_k_points[i + 1]

    start_frac = start_point['coords']
    end_frac = end_point['coords']

    # Call the helper function
    # Note: generate_interpolation returns (coords, distances_from_start_of_segment)
    segment_coords, segment_distances = generate_interpolation(
        start_frac,
        end_frac,
        BZ_basis_vectors,
        interpolate_point_num
    )

    # 3. Accumulate Data
    # For the very first point of the entire path, we add everything.
    # For subsequent segments, we skip the first point to avoid duplication
    # because the end of segment i is the start of segment i+1.
    if i == 0:
        # Record the start label
        high_symmetry_indices.append(current_index_count)
        high_symmetry_labels.append(start_point['label'])

        # Add all points
        all_coords.append(segment_coords)
        # Add cumulative distance offset
        all_distances.append(segment_distances + cumulative_distance)

        current_index_count += len(segment_coords)
    else:
        # Skip the first point (it overlaps with previous segment's last point)
        all_coords.append(segment_coords[1:])

        # Add cumulative distance offset, skipping first distance
        all_distances.append(segment_distances[1:] + cumulative_distance)

        current_index_count += len(segment_coords) - 1

    # Update cumulative distance for the next segment
    # segment_distances[-1] is the length of the current segment
    cumulative_distance += segment_distances[-1]

    # Record the end label of this segment
    high_symmetry_indices.append(current_index_count - 1)
    high_symmetry_labels.append(end_point['label'])

# 4. Concatenate arrays
all_coords = np.vstack(all_coords)
all_distances = np.concatenate(all_distances)


def obtain_quantum_number_k(all_coords):
    """
    Calculates the projection of Brillouin zone points (p) onto the real-space lattice vectors (a_j).
    These projections k_i = (p · a_j) represent the dimensionless quantum numbers (phases)
     associated with the Periodic Boundary Conditions (PBC) along each lattice vector.
    Args:
        all_coords:   A numpy array of shape (N, 3) containing Cartesian coordinates
                    of points p in the Brillouin Zone.
        processed_input_data:  A dictionary containing the system configuration,
                              specifically the 'lattice_basis' (a0, a1, a2).

    Returns:
        A numpy array of shape (N, 3) containing the dimensionless quantum numbers k.
                       - Column 0: k_0 = p · a0
                       - Column 1: k_1 = p · a1
                       - Column 2: k_2 = p · a2

    """


    # 2. Stack vectors into a (3, 3) matrix where each row is a basis vector
    # Matrix A = [ -- a0 -- ]
    #            [ -- a1 -- ]
    #            [ -- a2 -- ]
    # Shape: (3, 3)
    real_space_basis_matrix = np.array([a0, a1, a2])
    # 3. Perform vectorized dot product to obtain quantum numbers k
    # Calculation: k = p @ A^T
    #
    # Dimensions:
    # all_coords (p): (N, 3)
    # real_space_basis_matrix.T (A^T): (3, 3) (Vectors become columns)
    #
    # Resulting Matrix (k): (N, 3)
    # Column 0: Projection of p onto a0 (p · a0)
    # Column 1: Projection of p onto a1 (p · a1)
    # Column 2: Projection of p onto a2 (p · a2)
    quantum_numbers_k = all_coords @ real_space_basis_matrix.T
    return quantum_numbers_k

quantum_numbers_k = obtain_quantum_number_k(all_coords)
dim=2
quantum_numbers_input = quantum_numbers_k[:, 0:dim]
n_row, n_col = quantum_numbers_k.shape
print(n_row)
# Initialize a list to store the numerical Hamiltonian matrices
Hk_matrices_list = []

# 4. Iterate over each k-point (each row in the input)
for i in range(n_row):
    # Get the specific k-point coordinates for this step
    k_point = quantum_numbers_input[i, :]
    # Pass the components of the k-point as separate arguments to the lambdified function.
    # The * operator unpacks the numpy array into positional arguments (k0, k1, etc.)
    H_k_numerical=H_mat(*k_point)
    # Ensure the output is a numpy array (lambdify sometimes returns lists/scalars depending on backend)
    H_k_numerical = np.array(H_k_numerical, dtype=complex)
    Hk_matrices_list.append(H_k_numerical)

Hk_matrices_all = np.array(Hk_matrices_list)

# --- Parallel Diagonalization Functions ---
def diagonalize_chunk(matrix_chunk):
    """
     Worker function: Diagonalizes a subset (chunk) of matrices.
    Args:
        matrix_chunk:  matrix_chunk: A 3D numpy array of shape (n_chunk, dim, dim).

    Returns:
        eigenvalues_sorted: 2D array (n_chunk, matrix_dim), sorted ascending.
        eigenvectors_sorted: 3D array (n_chunk, matrix_dim, matrix_dim), columns sorted matching eigenvalues.
    """
    # 1. Diagonalize
    # np.linalg.eigh usually sorts by default, but we will enforce it below to be safe.
    eigenvalues_chunk, eigenvectors_chunk = np.linalg.eigh(matrix_chunk)
    # --- Explicit Sorting (Optional but safe) ---
    # 2. Get the indices that would sort the eigenvalues along the last axis (axis 1)
    # argsort returns indices of shape (n_chunk, dim)
    sort_indices = np.argsort(eigenvalues_chunk, axis=1)
    # 3. Reorder eigenvalues
    # We use take_along_axis to apply the sort indices to the 2D array
    eigenvalues_sorted = np.take_along_axis(eigenvalues_chunk, sort_indices, axis=1)
    # 4. Reorder eigenvectors
    # Eigenvectors are columns. The array shape is (n_chunk, row, col).
    # We need to sort the columns (axis 2) based on the eigenvalue indices.
    # We must expand dimensions of sort_indices to match the eigenvector shape: (n_chunk, 1, dim)
    sort_indices_expanded = sort_indices[:, np.newaxis, :]
    eigenvectors_sorted = np.take_along_axis(eigenvectors_chunk, sort_indices_expanded, axis=2)
    return eigenvalues_sorted, eigenvectors_sorted


def diagonalize_all_Hk_matrices(Hk_matrices_all, num_processes=None):
    """
    Parallelizes the diagonalization of the Hamiltonian matrices using multiprocessing.

    Args:
        Hk_matrices_all: 3D numpy array (n_k_points, dim, dim).
        num_processes: Number of CPU cores to use. If None, uses all available cores.

    Returns:
          all_eigenvalues: 2D array (n_k_points, dim)
          all_eigenvectors: 3D array (n_k_points, dim, dim)

    """
    # num_k_points = Hk_matrices_all.shape[0]
    # 1. Determine number of processes
    if num_processes is None:
        num_processes = cpu_count()
    print(f"Parallelism={num_processes}")
    # 2. Split the data into chunks
    # np.array_split divides the array into sub-arrays along axis 0.
    # It handles cases where n_k_points is not perfectly divisible by num_processes.
    chunks = np.array_split(Hk_matrices_all, num_processes, axis=0)
    # 3. Create the Pool and map the worker function
    with Pool(processes=num_processes) as pool:
        # pool.map applies 'diagonalize_chunk' to each item in 'chunks'
        # results is a list of tuples: [(evals_1, evecs_1), (evals_2, evecs_2), ...]
        results = pool.map(diagonalize_chunk, chunks)
    # 4. Reassemble the results
    # zip(*results) unzips the list of tuples into two separate tuples:
    # one containing all eigenvalue chunks, one containing all eigenvector chunks.
    eigenvalues_list, eigenvectors_list = zip(*results)
    # Concatenate the chunks back into monolithic numpy arrays
    all_eigenvalues = np.concatenate(eigenvalues_list, axis=0)
    all_eigenvectors = np.concatenate(eigenvectors_list, axis=0)
    return all_eigenvalues, all_eigenvectors

num_processes=12
t_diag_start = datetime.now()
all_eigenvalues, all_eigenvectors = diagonalize_all_Hk_matrices(Hk_matrices_all, num_processes)
t_diag_end = datetime.now()
print("diagonalization time: ", t_diag_end-t_diag_start)

# Create a figure
plt.figure(figsize=(6, 8))
num_bands = all_eigenvalues.shape[1]
for i in range(0,num_bands):
    # Plot the i-th column (band) against the k-path distance
    plt.plot(all_distances, all_eigenvalues[:, i], color='blue', linewidth=1.5)

# Add vertical lines for high symmetry points
for index in high_symmetry_indices:
    # Map the index to the actual distance value
    # Check bounds to avoid errors if index is out of range
    if index < len(all_distances):
        plt.axvline(x=all_distances[index], color='black', linestyle='--', linewidth=0.8)


# Set x-ticks to be the high symmetry points
valid_indices = [i for i in high_symmetry_indices if i < len(all_distances)]
tick_locations = [all_distances[i] for i in valid_indices]

plt.xticks(tick_locations, high_symmetry_labels, fontsize=14)


# Limit x-axis to the range of the path
plt.xlim(all_distances[0], all_distances[-1])
plt.ylim(-0.5,0.5)
# plt.yticks(fontsize=20)

# Y-axis label size
plt.ylabel("Energy", fontsize=22)
# plt.yticks(ticks=plt.yticks()[0], labels=[])


# Title size
plt.title(f"{name}", fontsize=24)
plt.grid(alpha=0.3)
plt.tight_layout()


out_pic_file_name=str(f"{name}_band.png")
plt.savefig(out_pic_file_name, bbox_inches='tight')
plt.close()