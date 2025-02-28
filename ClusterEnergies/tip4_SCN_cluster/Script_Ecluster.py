#!/usr/bin/python

# Basics
import os
import json
import numpy as np
from glob import glob

import subprocess

# ASE
from ase import Atoms
from ase import io
from ase.visualize import view

# Trajectory reader
import MDAnalysis
from MDAnalysis.analysis.distances import distance_array

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt

#=======================================
# Parameter Section
#=======================================

# Cluster parameters:

# Number of cluster images per configuration
Ncluster = 50

# Cluster range from center
Rcluster = 5.5

# Cluster center residue with number of residues in cluster, see syst_reslab
Ccluster_dict = {
    'SCN': [
        [0, 0, 16, 0],
        [0, 0, 14, 1],
        [1, 0, 13, 0],
        [1, 0, 12, 1]
        ], 
    #'POT': [
        #[0, 0, 16, 0],
        #[0, 0, 14, 1],
        #[1, 0, 14, 0],
        #[1, 0, 12, 1],
        #]
    }

# Source directory
source = '../../tip4/t0/300/100_0/'

# Trajectory source
traj_dcdtag = 'dyna.*.dcd'
# first: split condition, second: index for irun
traj_dcdsplt = [['.', 1]]
traj_crdfile = 'step1.crd'
traj_psffile = 'step1.psf'

# System information
syst_reslab = ['SCN', 'ACEM', 'TIP4', 'POT']
syst_resnum = {
    'SCN': 3,
    'ACEM': 9,
    'TIP3': 3,
    'TIP4': 4,
    'POT': 1}
syst_rescnt = {
    'SCN': [0, 1, 2],
    'ACEM': [0, 1, 2, 3, 4 ,5, 6, 7, 8],
    'TIP3': [0, 1, 2],
    'TIP4': [0, 2, 3],
    'POT': 0}
syst_ressym = {
    'SCN': ['N', 'C', 'S'],
    'ACEM': ['C', 'C', 'N', 'H', 'H', 'O', 'H', 'H', 'H'],
    'TIP3': ['O', 'H', 'H'],
    'TIP4': ['O', 'X', 'H', 'H'],
    'POT': ['K']}
syst_reschr = {
    'SCN': -1.0,
    'ACEM': 0.0,
    'TIP3': 0.0,
    'TIP4': 0.0,
    'POT': 1.0}

# Cluster selection file
data_file = "data_cluster_{:s}_{:d}.json"

# Gaussian parameters:

# Computational setup
guassian_method = "M062X"
gaussian_basisset = "aug-cc-pVTZ"
gaussian_counterpoise = 2
gaussian_ncpus = 4
gaussian_memory = 500*gaussian_ncpus
gaussian_spin = 1

gaussian_gsub = "gsub16"

# Compution resultfile
results_file = "data_results_{:s}_{:d}.json"


# Template files
gaussian_template_input = "template_gaussian_input.com"
gaussian_template_runsh = "template_gaussian_runsh.sh"

# Working files format
gaussian_format_input = "gaussian_{:s}_{:s}_{:s}_{:d}_{:03d}.com"
gaussian_format_output = "gaussian_{:s}_{:s}_{:s}_{:d}_{:03d}.out"

# Working directories
gaussian_workdir = "gaussian_workdir"


#=======================================
# Cluster Preparation Section
#=======================================

# Get dcd files
dcdfiles = np.array(glob(os.path.join(source, traj_dcdtag)))
iruns = []
for dcdfile in dcdfiles:
    irun = dcdfile.split('/')[-1]
    for splt in traj_dcdsplt:
        irun = irun.split(splt[0])[splt[1]]
    iruns.append(int(irun))
iruns = np.array(iruns)
psffile = os.path.join(source, traj_psffile)

# Sort dcd files
dcdsort = np.argsort(iruns)
dcdfiles = dcdfiles[dcdsort]
iruns = iruns[dcdsort]

# DCD length
dcdsteps = np.zeros_like(iruns)
for ii, dcdfile in enumerate(dcdfiles):
    
    # Open dcd file
    dcd = MDAnalysis.Universe(psffile, dcdfile)
    
    # Get trajectory length
    dcdsteps[ii] = len(dcd.trajectory)

# Read residue information
listres = {}
numres = {}
for ires, residue in enumerate(syst_reslab):
    info_res = []
    with open(os.path.join(source, traj_crdfile), 'r') as f:
        for line in f:
            ls = line.split()
            if len(ls) > 2:
                if residue in ls[2]:
                    info_res.append(line)
    listres[residue] = info_res
    numres[residue] = int(len(info_res)/syst_resnum[residue])

# Get residue atom numbers
atomsint = {}
for ires, residue in enumerate(syst_reslab):
    atomsinfo = np.zeros(
        [numres[residue], syst_resnum[residue]], dtype=int)
    for ir in range(numres[residue]):
        ri = syst_resnum[residue]*ir
        for ia in range(syst_resnum[residue]):
            info = listres[residue][ri + ia].split()
            atomsinfo[ir, ia] = int(info[0]) - 1
    atomsint[residue] = atomsinfo


#=======================================
# Cluster Selection Section
#=======================================

# Set random seed
np.random.seed(42)

# Average compilation lists
ncoord_avg = np.zeros_like(syst_reslab, dtype=float)
ncoord_cnt = np.zeros_like(syst_reslab, dtype=int)
        
for Ccluster, compilation in Ccluster_dict.items():
    
    for ic, compi in enumerate(compilation):
        
        # Skip if cluster data file already exists
        data_file_i = data_file.format(Ccluster, ic)
        if os.path.exists(data_file_i):
            continue
        
        # Extract cluster conformations
        cluster_data = {}
        isel = 0
        itry = 0
        while isel < Ncluster:
            
            # Choose random time step and center cluster residue
            ifdcd = np.random.choice(iruns)
            itime = np.random.randint(dcdsteps[ifdcd])
            inres = np.random.randint(numres[Ccluster])
            
            # Initialize cluster data dictionary
            data = {}
            
            # Open selected dcd file
            dcd = MDAnalysis.Universe(psffile, dcdfiles[ifdcd])
            
            # Get positions at selected time step
            positions = np.array(dcd.trajectory[itime]._pos, dtype=float)
            
            # Get masses
            masses = dcd._topology.masses.values
            
            # Get cell information
            cell = dcd.trajectory[itime]._unitcell
                
            # Get center residue positions (pos_ccluster) 
            # and cluster center position (pos_ccenter)
            idx_ccluster = np.array(atomsint[Ccluster][inres], dtype=int)
            pos_ccluster = positions[idx_ccluster]
            if isinstance(syst_rescnt[Ccluster], (list, tuple)):
                idx_ccenter = idx_ccluster[
                    np.array(syst_rescnt[Ccluster], dtype=int)]
                totmass = np.sum(masses[idx_ccenter])
                pos_ccenter = np.sum([
                    positions[icenter]*masses[icenter] 
                    for icenter in idx_ccenter], axis=0)/totmass
            elif isinstance(syst_rescnt[Ccluster], int):
                pos_ccenter = positions[idx_ccluster[syst_rescnt[Ccluster]]]
            else:
                raise ValueError("Wrong cluster center definition")
            
            # Center residue charge
            chr_ccluster = syst_reschr[Ccluster] 
                
            # Select other cluster residue positions (pos_rcluster)
            pos_rcluster_all = []
            res_rcluster_all = []
            sys_rcluster_all = []
            chr_rcluster_all = 0.0
            valid = True
            for ir, residue in enumerate(syst_reslab):
                
                # Skip if residue not in system
                if not numres[residue]:
                    continue
                
                # Skip if selection also seen invalid
                if not valid:
                    continue
                
                # Get all residue center positions
                pos_rcenter_all = []
                for ii in range(numres[residue]):
                    
                    # Get residue positions (pos_rcluster) 
                    # and residue center position (pos_rcenter)
                    idx_rcluster = np.array(atomsint[residue][ii], dtype=int)
                    pos_rcluster = positions[idx_rcluster]
                    if isinstance(syst_rescnt[residue], (list, tuple)):
                        idx_rcenter = idx_rcluster[
                            np.array(syst_rescnt[residue], dtype=int)]
                        totmass = np.sum(masses[idx_rcenter])
                        pos_rcenter = np.sum([
                            positions[icenter]*masses[icenter] 
                            for icenter in idx_rcenter], axis=0)/totmass
                    elif isinstance(syst_rescnt[residue], int):
                        pos_rcenter = positions[
                            idx_rcluster[syst_rescnt[residue]]]
                    else:
                        raise ValueError("Wrong cluster center definition")
            
                    # Append residue center position to list
                    pos_rcenter_all.append(pos_rcenter)
                
                # Get cluster center - residue center distances
                pos_rcenter_all = np.array(pos_rcenter_all)
                distances = distance_array(
                    pos_ccenter, pos_rcenter_all, box=cell).reshape(-1)
                
                # Select cluster residues
                in_cluster_range = np.logical_and(
                    distances > 0.0,
                    distances < Rcluster)
                in_cluster_indices = np.where(in_cluster_range)[0]
                
                # Update cluster compilation average
                if ncoord_cnt[ir] == 0:
                    ncoord_avg[ir] = float(len(in_cluster_indices))
                else:
                    ncoord_avg[ir] = (
                        (
                            ncoord_avg[ir]*ncoord_cnt[ir] 
                            + float(len(in_cluster_indices))
                            )/float(ncoord_cnt[ir] + 1)
                        )
                ncoord_cnt[ir] += 1
                
                # Sort by center distance
                dist_sorted = np.argsort(distances[in_cluster_range])
                in_cluster_indices = in_cluster_indices[dist_sorted]
                distances = distances[in_cluster_range][dist_sorted]
                
                # Cut to selected compilation
                if compi[ir] and len(in_cluster_indices) < compi[ir]:
                    valid = False
                    continue
                elif not compi[ir] and len(in_cluster_indices):
                    valid = False
                    continue
                in_cluster_indices = in_cluster_indices[:compi[ir]]
                
                # Append to cluster position array
                N_in_cluster_range = len(in_cluster_indices)
                if N_in_cluster_range:
                    idx_rcluster = np.array(
                        atomsint[residue][in_cluster_indices], dtype=int)
                    pos_rcluster = positions[idx_rcluster].reshape(-1, 3)
                    pos_rcluster_all += [list(posi) for posi in pos_rcluster]
                
                # Append residue labels and residue atom symbols
                res_rcluster_all += [residue]*N_in_cluster_range
                sys_rcluster_all += [*syst_ressym[residue]]*N_in_cluster_range
                
                # Add residue charges
                chr_rcluster_all += syst_reschr[residue]*N_in_cluster_range
            
            # Skip if cluster is not valid
            if not valid:
                continue
            
            # Prepare data entries
            c_pos = [list(posi) for posi in pos_ccluster]
            c_sym = list(syst_ressym[Ccluster])
            c_chr = chr_ccluster
            r_pos = [list(posi) for posi in pos_rcluster_all]
            r_sym = list(sys_rcluster_all)
            r_chr = chr_rcluster_all
            
            # Wrap cluster positions
            positions = np.array((c_pos + r_pos), dtype=float)
            positions -= pos_ccenter
            for ip, posi in enumerate(positions):
                for jj in range(3):
                    positions[ip, jj] = (posi[jj] - cell[jj]/2.)%cell[jj]
            c_pos = [list(posi) for posi in positions[:len(c_sym)]]
            r_pos = [list(posi) for posi in positions[len(c_sym):]]
            
            cluster = Atoms(
                c_sym + r_sym, positions=(c_pos + r_pos), cell=cell, pbc=True)
            io.write(
                "cluster_{:s}_{:d}_{:03d}.xyz".format(
                    Ccluster, ic, isel), 
                cluster)
            
            # Add selection to data
            data['c_pos'] = c_pos
            data['c_sym'] = c_sym
            data['c_chr'] = c_chr
            data['r_pos'] = r_pos
            data['r_sym'] = r_sym
            data['r_chr'] = r_chr
            data['residues'] = list([Ccluster] + res_rcluster_all)
            cluster_data[isel] = data
            
            # Increment selection counter
            isel += 1
            
            # Check tries
            itry += 1
            if itry > Ncluster*100:
                break

            # Show average compilation
            print("Average Cluster compilation")
            for ir, residue in enumerate(syst_reslab):
                print("  ", residue, ncoord_avg[ir])

        # Save cluster data
        with open(data_file_i, 'w') as f:
            json.dump(cluster_data, f, indent="  ")



##=======================================
## Gaussian Computation Section
##=======================================

# Create input directory
if not os.path.exists(gaussian_workdir):
    os.makedirs(gaussian_workdir)

for Ccluster, compilation in Ccluster_dict.items():
    
    for ic, compi in enumerate(compilation):
        
        # Read cluster data
        data_file_i = data_file.format(Ccluster, ic)
        with open(data_file_i, 'r') as f:
            cluster_data = json.load(f)
        
        # Create input files
        for ii, data in cluster_data.items():
    
            # Working file names
            gaussian_input = gaussian_format_input.format(
                guassian_method, gaussian_basisset, Ccluster, ic, int(ii))
            gaussian_output = gaussian_format_output.format(
                guassian_method, gaussian_basisset, Ccluster, ic, int(ii))
            
            # Skip if output exists
            if os.path.exists(os.path.join(gaussian_workdir, gaussian_output)):
                continue
            
            # Prepare cluster data
            charge = round(data['c_chr'] + data['r_chr'])
            chargespin = '{:d} {:d} {:d} {:d} {:d} {:d}'.format(
                charge, gaussian_spin,
                round(data['c_chr']), gaussian_spin,
                round(data['r_chr']), gaussian_spin)
            atom_symbols = data['c_sym'] + data['r_sym']
            atom_fragment = [1]*len(data['c_sym']) + [2]*len(data['r_sym'])
            atom_positions = np.append(data['c_pos'], data['r_pos']).reshape(
                -1, 3)
            lines_positions = [
                "{:s}(Fragment={:d}) {:>20.15f} {:>20.15f} {:>20.15f}\n".format(
                    atom_symbol, atom_fragment[iatom], *atom_positions[iatom])
                for iatom, atom_symbol in enumerate(atom_symbols)
                if atom_symbol.lower() != 'X'.lower()]
            lines_positions = ("".join(lines_positions))[:-1]
            
            # Create Gaussian input file
            with open(gaussian_template_input, 'r') as f:
                ginput = f.read()
            ginput = ginput.replace("%MEM%", str(gaussian_memory))
            ginput = ginput.replace("%CPU%", str(gaussian_ncpus))
            ginput = ginput.replace("%METHOD%", str(guassian_method))
            ginput = ginput.replace("%BASIS%", str(gaussian_basisset))
            ginput = ginput.replace("%NFR%", str(gaussian_counterpoise))
            ginput = ginput.replace("%CHARGESPIN%", str(chargespin))
            ginput = ginput.replace("%XYZ%", str(lines_positions))
            if "K" in atom_symbols:
                with open("template_gaussian_basisK.com", 'r') as f:
                    basisK = f.read()
                ginput += basisK
            ginput += '\n'
            with open(os.path.join(gaussian_workdir, gaussian_input), 'w') as f:
                f.write(ginput)
            
            ## Submit Gaussian job
            #subprocess.run(
                #'cd {:s} ; {:s} {:s} -q infinite'.format(
                    #gaussian_workdir, gaussian_gsub, gaussian_input), 
                #shell=True)


#=======================================
# Result Collection Section
#=======================================

for Ccluster, compilation in Ccluster_dict.items():
    
    for ic, compi in enumerate(compilation):
        
        # Read cluster data
        data_file_i = data_file.format(Ccluster, ic)
        with open(data_file_i, 'r') as f:
            cluster_data = json.load(f)
        
        # Result dictionary
        results = {}

        # Iterate over results
        running_number = 0
        for ii, data in cluster_data.items():

            # Working file names
            gaussian_input = gaussian_format_input.format(
                guassian_method, gaussian_basisset, Ccluster, ic, int(ii))
            gaussian_output = gaussian_format_output.format(
                guassian_method, gaussian_basisset, Ccluster, ic, int(ii))
            
            # Skip if output exists
            if not os.path.exists(os.path.join(
                gaussian_workdir, gaussian_output)):
                continue
            
            # Read input file
            with open(
                os.path.join(gaussian_workdir, gaussian_output), 'r') as f:
                reslines = f.readlines()
            
            # Get counterpoise corrected interaction energy 
            scf_energy = []
            for line in reslines:
                if "SCF Done:" in line:
                    scf_energy.append(float(line.split()[4]))
            if not len(scf_energy) == 5:
                print(ii)
                continue
            dE_int = scf_energy[0] - scf_energy[1] - scf_energy[2]
            
            # Read input file
            with open(
                os.path.join(gaussian_workdir, gaussian_input), 'r') as f:
                inplines = f.readlines()
            
            # Collect residue compilation
            #symbols = []
            #residues = []
            #positions = []
            #for line in inplines:
                #if "Fragment" in line:
                    #linedata = line.split()
                    
                    #symbol = linedata[0].split("(")[0]
                    #symbols.append(symbol)
                    #if symbol == "S":
                        #residues.append("SCN")
                    #elif symbol == "O":
                        #residues.append("TIP4")
                    #elif symbol == "K":
                        #residues.append("POT")
                    
                    #position = np.array(linedata[-3:], dtype=float)
                    #positions.append(list(position))
            
            # Combine results
            symbols = data['c_sym'] + data['r_sym']
            positions = data['c_pos'] + data['r_pos']
            residues = data['residues']

            # Append result
            results[running_number] = {}
            results[running_number]['E_int'] = dE_int
            results[running_number]['residues'] = residues
            results[running_number]['positions'] = positions
            results[running_number]['symbols'] = symbols
            
            # Increment running number
            running_number += 1
                
        # Save results
        results_file_i = results_file.format(Ccluster, ic)
        with open(results_file_i, 'w') as f:
            json.dump(results, f, indent="  ")
