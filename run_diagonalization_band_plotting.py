from datetime import datetime
import sys

import numpy as np
import sympy as sp
sp.init_printing(use_unicode=False, wrap_line=False)
#this script runs diagonalization for plotting band

from plot_energy_band.block_diagonalization import *


argErrCode = 20
if (len(sys.argv) != 2):
    print("wrong number of arguments")
    print("example: python preprocessing.py /path/to/mc.conf")
    exit(argErrCode)

confFileName = str(sys.argv[1])
num_processes=12
interpolate_point_num=25
verbose=True

all_coords, all_distances, high_symmetry_indices, high_symmetry_labels,quantum_numbers_k,processed_input_data=subroutine_get_interpolated_points_in_BZ_and_quantum_number_k(confFileName)
