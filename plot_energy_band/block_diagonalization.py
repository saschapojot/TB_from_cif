from multiprocessing import Pool, cpu_count
import sympy as sp
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import pickle


from plot_energy_band.load_path_in_Brillouin_zone import *


# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))