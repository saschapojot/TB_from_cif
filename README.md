
#this is for demonstration during symmetry analysis development
#TODO: f orbital representations must be rechecked
###########################
Part I, proprocessing
1. python init_from_cif.py ./path/to/xxx.cif, #the xxx.cif file is generated from material studio
2. #step 1 generates ./path/to/xxx.conf, needs to complete 
    #empty key values in xxx.conf file, then
   python ./path/to/xxx.conf
3. #deals with diagonalization of energy bands for plotting
    python run_diagonalization_band_plotting.py ./path/to/xxx.conf #same conf file as in step 1, 