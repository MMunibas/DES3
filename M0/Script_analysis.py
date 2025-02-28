# Basics
import os
import sys
import time
import numpy as np
from itertools import product, groupby
from glob import glob

# Trajectory reader
import MDAnalysis
from MDAnalysis.analysis.distances import self_distance_array, distance_array

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

# ASE
from ase import Atoms
from ase.io import read, write
from ase.visualize import view

# Multiprocessing
from multiprocessing import Pool

#------------
# Parameters
#------------

# Parallel evaluation runs
tasks = 4

# Source directory
source = '.'

# Temperatures
temperatures = [300]

# Mixtures
mixtures = [0, 20, 30, 50, 70, 80, 90, 100]

# Runs
Nrun = 5

# Trajectory source
traj_dcdtag = 'dyna.*.dcd'
# first: split condition, second: index for irun
traj_dcdinfo = ['.', 1]
traj_crdfile = 'step1.crd'
traj_psffile = 'step1.psf'

# Result directory
res_maindir = 'results_analysis'
res_evaldir = 'evalfiles'

# Residue of interest
eval_residue = 'SCN'

# Distances from eval_residue atoms to eval_resreg atoms
evaluation = {
    #(0, 1, 2): {
        #'SCN': [0, 1, 2],
        #'ACEM': [2, 3, 4, 5],
        #'TIP3': [0, 1, 2],
        #'POT': [0]},
    0: {
        'SCN': [0, 1, 2],
        'ACEM': [2, 3, 4, 5],
        'TIP3': [0, 1, 2],
        'POT': [0]},
    1: {
        'SCN': [0, 1, 2],
        'ACEM': [2, 3, 4, 5],
        'TIP3': [0, 1, 2],
        'POT': [0]},
    2: {
        'SCN': [0, 1, 2],
        'ACEM': [2, 3, 4, 5],
        'TIP3': [0, 1, 2],
        'POT': [0]},
    }

# Regarding residues
eval_resreg = ['SCN', 'ACEM', 'TIP3', 'POT']
eval_resnum = {
    'SCN': 3,
    'ACEM': 9,
    'TIP3': 3,
    'POT': 1}
eval_ressym = {
    'SCN': ['N', 'C', 'S'],
    'ACEM': ['C', 'C', 'N', 'H', 'H', 'O', 'H', 'H', 'H'],
    'TIP3': ['O', 'H', 'H'],
    'POT': ['K']}
eval_reschr = {
    'SCN': [-0.46, -0.36, -0.18],
    'ACEM': [-0.27, 0.55, -0.62, 0.32, 0.30, -0.55, 0.09, 0.09, 0.09],
    'TIP3': [-0.84, 0.42, 0.42],
    'POT': [1.00]}
eval_resmss = {
    'SCN': [14.01, 12.01, 32.06],
    'ACEM': [12.01, 12.01, 14.01, 1.01, 1.01, 12.01, 1.01, 1.01, 1.01],
    'TIP3': [14.01, 1.01, 1.01],
    'POT': [39.10]}

# Van der Waals radii to evaluate close distances
eval_vdWradii = {'H': 1.1, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'K': 2.75}

# Time step for averaging in ps
eval_timestep = 10.0

# Maximum time to evaluate in ps
eval_maxtime = 5000.0

# Distance of interest to plot
plot_dint = 7.0

plot_residue = r'SCN$^{-}$'
plot_ressym = {
    'SCN': ['N', 'C', 'S', r'SCN$^{-}$'],
    'ACEM': [
        'C', 'C', 'N', r'H$_\mathrm{N}$', r'H$_\mathrm{N}$', 'O', 'H', 'H', 'H',
        'acetamide'],
    'TIP3': ['O', 'H', 'H', r'H$_2$O'],
    'POT': [r'K$^{+}$', r'K$^{+}$']}

#--------------
# Preparations
#--------------

# Iterate over systems
info_systems = np.array(list(product(temperatures, mixtures)))

# Get atoms and pair information
numres = {}
atomsint = {}

# Iterate over systems
for sys in info_systems:
    
    # Data directory
    temp = str(sys[0])
    mix = str(sys[1])
    datadir = os.path.join(source, temp, mix + "_0")
    
    # System tag
    tag = temp + '_' + mix
    
    # Check if all residues are considered
    all_residues = eval_resreg
    if eval_residue not in eval_resreg:
        all_residues.append(eval_residue)
    
    # Read residue information
    listres = {}
    numres[tag] = {}
    for ires, res in enumerate(all_residues):
        info_res = []
        with open(os.path.join(datadir, traj_crdfile), 'r') as f:
            for line in f:
                ls = line.split()
                if len(ls) > 2:
                    if res in ls[2]:
                        info_res.append(line)
        listres[res] = info_res
        numres[tag][res] = int(len(info_res)/eval_resnum[res])
    
    # Get residue atom numbers
    atomsint[tag] = {}
    for ires, res in enumerate(all_residues):
        atomsinfo = np.zeros(
            [numres[tag][res], eval_resnum[res]], dtype=int)
        for ir in range(numres[tag][res]):
            ri = eval_resnum[res]*ir
            for ia in range(eval_resnum[res]):
                info = listres[res][ri + ia].split()
                atomsinfo[ir, ia] = int(info[0]) - 1
        atomsint[tag][res] = atomsinfo
        
# Make result directory
if not os.path.exists(res_maindir):
    os.mkdir(res_maindir)
    
if not os.path.exists(os.path.join(res_maindir, res_evaldir)):
    os.mkdir(os.path.join(res_maindir, res_evaldir))

#---------------------
# Collect system data
#---------------------

# Iterate over systems and resids
info_systems = list(product(temperatures, mixtures, range(Nrun)))

def read_sys(i):
    
    # Begin timer
    start = time.time()
    
    # Data directory
    temp = str(info_systems[i][0])
    mix = str(info_systems[i][1])
    run = str(info_systems[i][2])
    
    #resid = info_systems_resids[i][3]
    
    datadir = os.path.join(source, temp, "{:s}_{:s}".format(mix, run))
    
    # System tag
    tag = temp + '_' + mix
    
    # Read dcd files and get atom distances
    #---------------------------------------
    
    # Get dcd files
    dcdfiles = np.array(glob(os.path.join(datadir, traj_dcdtag)))
    iruns = np.array([
        int(dcdfile.split('/')[-1].split(traj_dcdinfo[0])[traj_dcdinfo[1]])
        for dcdfile in dcdfiles])
    psffile = os.path.join(datadir, traj_psffile)
    
    # Sort dcd files
    dcdsort = np.argsort(iruns)
    dcdfiles = dcdfiles[dcdsort]
    iruns = iruns[dcdsort]
    
    # Distance file
    distfile = os.path.join(
        res_maindir, res_evaldir, 
        'dists_{:s}_{:s}_{:s}_{:s}.npy'.format(
            temp, mix, run, eval_residue))
    
    if not os.path.exists(distfile) or False:
        
        # Initialize distance dictionary
        # Final build (n-d: description):
        # 0: Evaluation residue 1 atom (or COM), 
        # 1: Target residue type,
        # 2: Target residue 2 atom (or COM),
        # 3: Time step
        # 4: Residue 1
        # 5: Residue 2
        # -> Distance between residue 1 atom and residue 2 atom (or COM)
        eval_dists = {}
        
        # Initialize trajectory time counter in ps
        traj_time_dcd = 0.0
        
        # Iterate over dcd files
        for idcd, dcdfile in enumerate(dcdfiles):
            
            print(
                "Temp = {:s}, Mix = {:s}, Run = {:s}, idcd = {:d}".format(
                    temp, mix, run, idcd))
            
            # Open dcd file
            dcd = MDAnalysis.Universe(psffile, dcdfile)
            
            # Get trajectory parameter
            Nframes = len(dcd.trajectory)
            Nskip = int(dcd.trajectory.skip_timestep)
            dt = np.round(
                float(dcd.trajectory._ts_kwargs['dt']), decimals=8)/Nskip
            
            # Get atom types
            atoms = np.array([ai for ai in dcd._topology.names.values])
            Natoms = len(atoms)
            
            # Evaluate structure at centre of eval_timestep
            timesteps = np.arange(Nframes)*dt*Nskip
            timewindows = []
            for istep, timestep in enumerate(timesteps):
                
                aux = timestep%eval_timestep
                if (aux < dt*Nskip/2.) or (eval_timestep - aux < dt*Nskip/2.):
                    
                    timewindows.append(istep)
                    
            if timesteps[-1] not in timewindows:
                
                timewindows.append(istep + 1)
                    
            timewindows = np.array(timewindows)
            timecenters = np.array([
                (timewindows[it] + timewindows[it - 1])//2 
                for it in range(1, len(timewindows))])
            Ntimes = len(timecenters)
            
            # Prepare position and cell array to read from trajectory
            # to compute distances just once for all timesteps
            positions = np.zeros(
                (Ntimes, Natoms, 3), dtype=np.float32)
            cell = np.zeros(
                (Ntimes, 6), dtype=np.float32)
            
            # Read trajectory
            for ic, tc in enumerate(timecenters):
                
                # Get positions
                positions[ic] = dcd.trajectory[tc]._pos
                
                # Get cell information
                cell[ic] = dcd.trajectory[tc]._unitcell
                
            # Residue positions
            eval_posres = positions[:, atomsint[tag][eval_residue], :]
            
            #print(Ntimes, Natoms)
            
            # Iterate over defined eval_residue atoms
            for ires, (resatomi, item) in enumerate(evaluation.items()):
                
                #print(ires, (resatomi, item))
                
                if isinstance(resatomi, tuple):
                    key_i = 'COM' + (len(resatomi)*'_{:d}').format(
                        *resatomi)
                    totmass = sum(
                        [eval_resmss[eval_residue][ai] for ai in resatomi])
                    eval_posres_i = np.sum(np.array([
                        positions[:, atomsint[tag][eval_residue][:, ai], :]
                        *eval_resmss[eval_residue][ai]
                        for ai in resatomi]), axis=0)/totmass
                else:
                    key_i = resatomi
                    eval_posres_i = positions[
                        :, atomsint[tag][eval_residue][:, key_i], :]
                
                # Add case to result distances dictionary
                if key_i not in eval_dists:
                    eval_dists[key_i] = {}
                
                #print(eval_posres_i.shape)
                
                # Iterate over defined target residues
                for jres, (resj, resatoms) in enumerate(item.items()):
                        
                    # Add case to result distances dictionary
                    if resj not in eval_dists[key_i]:
                        eval_dists[key_i][resj] = {}
                    
                    #print(jres, (resj, resatoms))
                    
                    # Iterate over target residue atoms
                    for ja, resatomj in enumerate(resatoms):
                        
                        if isinstance(resatomj, list):
                            key_j = 'COM' + (len(resatomj)*'_{:d}').format(
                                *resatomj)
                            totmass = sum([
                                eval_resmss[resj][aj]
                                for aj in resatomj])
                            eval_posres_j = np.sum(np.array([
                                positions[:, atomsint[tag][resj][:, aj], :]
                                *eval_resmss[resj][aj] 
                                for aj in resatomj]), axis=0)/totmass
                        else:
                            key_j = resatomj
                            eval_posres_j = positions[
                                :, atomsint[tag][resj][:, resatomj], :]
                        
                        # Add case to result distances dictionary
                        if key_j not in eval_dists[key_i][resj]:
                            eval_dists[key_i][resj][key_j] = []
                        
                        #print(ja, resatomj)
                        #print(eval_posres_j.shape)
                        
                        # Compute distances per selected frame
                        for ic, tc in enumerate(timecenters):
                            
                            distances_ij = distance_array(
                                eval_posres_i[ic], eval_posres_j[ic],
                                box=cell[ic])
                            
                            #print(eval_residue, resj, key_i, key_j)
                            #print(eval_residue==resj, key_i==key_j)
                            #print(distances_ij.shape)
                            
                            eval_dists[key_i][resj][key_j].append(
                                distances_ij)
                            
            # Set time
            traj_time = traj_time_dcd + Nframes*Nskip*dt
            
            # Check time
            if traj_time >= eval_maxtime:
                
                # Convert final list to numpy array
                for key_i in eval_dists.keys():
                    for resj in eval_dists[key_i].keys():
                        for key_j in eval_dists[key_i][resj].keys():
                            eval_dists[key_i][resj][key_j] = np.array(
                                eval_dists[key_i][resj][key_j])
                            #print(
                                #eval_residue, key_i, resj, key_j,
                                #eval_dists[key_i][resj][key_j].shape)
                            
                # Save result file of frames
                np.save(distfile, eval_dists, allow_pickle=True)
                
                # End timer
                end = time.time()
                print(
                    'System {:s}, {:s} for {:.3f}ps '.format(
                        temp, mix, traj_time)
                    + 'done in {:4.1f} s'.format(end - start))
                
                return
            
            traj_time_dcd = traj_time
        
        # If eval_maxtime is not reached before end of dcd files save anyways
        
        # Convert final list to numpy array
        for key_i in eval_dists.keys():
            for resj in eval_dists[key_i].keys():
                for key_j in eval_dists[key_i][resj].keys():
                    eval_dists[key_i][resj][key_j] = np.array(
                        eval_dists[key_i][resj][key_j])
        
        # Save result file of frames
        np.save(distfile, eval_dists, allow_pickle=True)
        
        # End timer
        end = time.time()
        print(
            'System {:s}, {:s} for just {:.3f}ps '.format(
                temp, mix, traj_time)
            + 'done in {:4.1f} s'.format(end - start))
                
        return
        
    else:
        
        print('System {:s}, {:s} already done'.format(
            temp, mix))
        return
    
#read_sys(0)
#exit()

if tasks==1:
    for i in range(0, len(info_systems)):
        read_sys(i)
else:    
    if __name__ == '__main__':
        pool = Pool(tasks)
        pool.imap(read_sys, range(0, len(info_systems)))
        pool.close()
        pool.join()


#----------------------------
# Paper Plot g(r) vs. mix V
#----------------------------

# Fontsize
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

dpi = 300


info_systems = list(product(temperatures, mixtures, range(Nrun)))

rad_lim = (1.00, 7.00)
rad_bins = np.linspace(rad_lim[0], rad_lim[1], num=121)
rad_dist = rad_bins[1] - rad_bins[0]
rad_cent = rad_bins[:-1] + rad_dist/2.

for isys, sysi in enumerate(info_systems):
    
    # Data directory
    temp = str(sysi[0])
    mix = str(sysi[1])
    run = str(sysi[2])
    datadir = os.path.join(source, temp, mix)
    
    # System tag
    tag = temp + '_' + mix
    
    # Distance file
    distfile = os.path.join(
        res_maindir, res_evaldir, 
        'dists_{:s}_{:s}_{:s}_{:s}.npy'.format(
            temp, mix, run, eval_residue))
    
    gfile = os.path.join(
        res_maindir, res_evaldir, 
        'g_histogramm_data_{:s}_{:s}_{:s}.npy'.format(
            temp, mix, run))
    nfile = os.path.join(
        res_maindir, res_evaldir, 
        'n_histogramm_data_{:s}_{:s}_{:s}.npy'.format(
            temp, mix, run))
    
    if not os.path.exists(gfile) or False:
        print(distfile)
        # Load distance file
        distsdata = np.load(distfile, allow_pickle=True).item(0)
        
        # Coordination number and radial distribution function results
        # eval_residue atom, target residue, target residue atom, ..
        # radial distances (cut or bin center)
        n_hist = {}
        g_hist = {}
        
        # Iterate over defined eval_residue atoms
        for ires, (resatomi, item) in enumerate(evaluation.items()):
            
            if isinstance(resatomi, tuple):
                key_i = 'COM' + (len(resatomi)*'_{:d}').format(
                    *resatomi)
            else:
                key_i = resatomi
        
            # Add case to result dictionary
            n_hist[key_i] = {}
            g_hist[key_i] = {}
            
            # Iterate over defined target residues
            for jres, (resj, resatoms) in enumerate(item.items()):
                
                # Add case to result dictionary
                n_hist[key_i][resj] = {}
                g_hist[key_i][resj] = {}
                
                # Iterate over target residue atoms
                for ja, resatomj in enumerate(resatoms):
                    
                    if isinstance(resatomj, list):
                        key_j = 'COM' + (len(resatomj)*'_{:d}').format(
                            *resatomj)
                    else:
                        key_j = resatomj
        
                    n_hist[key_i][resj][key_j] = np.zeros(
                        rad_cent.shape[0], dtype=np.float32)
                    g_hist[key_i][resj][key_j] = np.zeros(
                        rad_cent.shape[0], dtype=np.float32)
                    
                    # Get distances for atom pair (Nsteps, i, j)
                    dists = distsdata[key_i][resj][key_j]
                    Nsteps = dists.shape[0]
                    Nresid = float(dists.shape[1])
                    if eval_residue==resj and key_i==key_j:
                        Nresid *= 2.
                    
                    # Compute distance histogram
                    distslist = dists.reshape(-1)
                    nr = np.histogram(
                        distslist[distslist > 0.0], bins=rad_bins)[0]/Nsteps
                    print(eval_residue, resj, key_i, key_j, len(distslist), np.sum(nr), dists.shape)
                    
                    # Compute coordination number per cutoff radii
                    for ir, rc in enumerate(rad_cent):
                        
                        n_hist[key_i][resj][key_j][ir] += np.sum(nr[:ir])/Nresid
                    
                    # Compute rdf
                    V = 4./3.*np.pi*rad_lim[1]**3
                    N = np.sum(nr)
                    if N > 0.0:
                        gr = (V/N)*(nr/rad_dist)/(4.0*np.pi*rad_cent**2)
                    else:
                        gr = np.zeros_like(nr)
                    
                    g_hist[key_i][resj][key_j] = gr[:]
            
        np.save(nfile, n_hist, allow_pickle=True)
        np.save(gfile, g_hist, allow_pickle=True)
        
    else:
        
        n_hist = np.load(nfile, allow_pickle=True).item(0)
        g_hist = np.load(gfile, allow_pickle=True).item(0)

# Plot options

# Figure
figsize = (12, 6)
sfig = float(figsize[0])/float(figsize[1])
fig = plt.figure(figsize=figsize)

# Alignment
left = 0.10
bottom = 0.15
column = [0.38, 0.10]
row = [0.35, 0.03]

line_scheme = {
    '0': 'solid', 
    '20': 'dotted', 
    '30': 'dashed', 
    '50': 'dashdot', 
    '70': (0, (5, 1)), 
    '80': (0, (3, 1, 1, 1, 1, 1)), 
    '90': (0, (3, 1, 3, 1, 1, 1)), 
    '100': (0, (3, 1, 3, 1, 1, 1))}

color_scheme = {
    '0': 'b', 
    '20': 'r', 
    '30': 'brown', 
    '50': 'g', 
    '70': 'purple', 
    '80': 'orange', 
    '90': 'magenta', 
    '100': 'cyan'}

legl = {
    '0': '0%',
    '20': '20%',
    '30': '30%',
    '50': '50%',
    '70': '70%',
    '80': '80%',
    '90': '90%',
    '100': '100%'}

# Add axis
axs1 = fig.add_axes([
    left + 0*sum(column), bottom + 1*sum(row), column[0], row[0]])
axs2 = fig.add_axes([
    left + 0*sum(column), bottom + 0*sum(row), column[0], row[0]])
axs3 = fig.add_axes([
    left + 1*sum(column), bottom + 1*sum(row), column[0], row[0]])
axs4 = fig.add_axes([
    left + 1*sum(column), bottom + 0*sum(row), column[0], row[0]])

info_systems = np.array(
    list(product(temperatures, [0, 20, 30, 50, 70, 80, 90, 100])))

for isys, sysi in enumerate(info_systems):
        
    # Data directory
    temp = str(sysi[0])
    mix = str(sysi[1])
    
    g_hist1 = None
    g_hist2 = None
    g_hist3 = None
    g_hist4 = None
    
    # Read results per run
    for irun in range(Nrun):
        
        gfile = os.path.join(
            res_maindir, res_evaldir, 
            'g_histogramm_data_{:s}_{:s}_{:s}.npy'.format(
                temp, mix, str(run)))
        
        g_hist = np.load(gfile, allow_pickle=True).item(0)
        
        if g_hist1 is None:
            g_hist1 = g_hist[1]['SCN'][1]
        else:
            g_hist1 += g_hist[1]['SCN'][1]
        
        if g_hist2 is None:
            g_hist2 = g_hist[1]['POT'][0]
        else:
            g_hist2 += g_hist[1]['POT'][0]
        
        if g_hist3 is None:
            g_hist3 = g_hist[1]['TIP3'][0]
        else:
            g_hist3 += g_hist[1]['TIP3'][0]
        
        if g_hist4 is None:
            g_hist4 = g_hist[1]['ACEM'][2]
        else:
            g_hist4 += g_hist[1]['ACEM'][2]
    
    # Normalize radial distribution
    g_hist1 /= float(Nrun)
    g_hist2 /= float(Nrun)
    g_hist3 /= float(Nrun)
    g_hist4 /= float(Nrun)
    
    axs1.plot(
        rad_cent, g_hist1,
        color=color_scheme[mix],
        linestyle=line_scheme[mix],
        label=legl[mix],
        lw=2)
    
    axs2.plot(
        rad_cent, g_hist2,
        color=color_scheme[mix],
        linestyle=line_scheme[mix],
        label=legl[mix],
        lw=2)
    
    axs3.plot(
        rad_cent, g_hist3/g_hist3[-1],
        color=color_scheme[mix],
        linestyle=line_scheme[mix],
        label=legl[mix],
        lw=2)
    
    axs4.plot(
        rad_cent, g_hist4/g_hist4[-1],
        color=color_scheme[mix],
        linestyle=line_scheme[mix],
        label=legl[mix],
        lw=2)
    
    if isys==0:
            
        axs2.set_xlabel(r'Radius $r$ ($\mathrm{\AA}$)')
        axs2.get_xaxis().set_label_coords(0.50, -0.20)
        axs4.set_xlabel(r'Radius $r$ ($\mathrm{\AA}$)')
        axs4.get_xaxis().set_label_coords(0.50, -0.20)
        
        axs2.set_ylabel(r'$g(r)$')
        axs2.get_yaxis().set_label_coords(-0.12, 1.00)
        #axs3.set_ylabel(r'$g(r) \cdot \rho(\mathrm{H_2O})$')
        axs3.set_ylabel(r'$g(r)$')
        axs3.get_yaxis().set_label_coords(-0.12, 0.50)
        #axs4.set_ylabel(r'$g(r) \cdot \rho(\mathrm{acetamide})$')
        axs4.set_ylabel(r'$g(r)$')
        axs4.get_yaxis().set_label_coords(-0.12, 0.50)
        
        axs1.set_xlim(rad_lim)
        axs2.set_xlim(rad_lim)
        axs3.set_xlim(rad_lim)
        axs4.set_xlim(rad_lim)
        
        axs1.set_ylim([-4/20., 4 + 4/20])
        axs2.set_ylim([-11/20., 11 + 11/20])
        axs3.set_ylim([-3/20., 3 + 3/20])
        axs4.set_ylim([-2/20., 2 + 2/20])
        
        axs1.set_xticklabels([])
        axs3.set_xticklabels([])
        
        axs1.set_yticks([0, 1, 2, 3, 4])
        axs2.set_yticks([0, 2, 4, 6, 8, 10])
        axs3.set_yticks([0, 1, 2, 3])
        axs4.set_yticks([0, 1, 2])
        
        tbox = TextArea(
            'A', 
            textprops=dict(color='k', fontsize=18))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.02, 0.84),
            bbox_transform=axs1.transAxes, borderpad=0.)
        
        axs1.add_artist(anchored_tbox)
        
        tbox = TextArea(
            r'C$_\mathrm{SCN^-}-$C$_\mathrm{SCN^-}$', 
            textprops=dict(color='k', fontsize=MEDIUM_SIZE))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.02, 0.64),
            bbox_transform=axs1.transAxes, borderpad=0.)
        
        axs1.add_artist(anchored_tbox)
        
        tbox = TextArea(
            'B', 
            textprops=dict(color='k', fontsize=18))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.02, 0.84),
            bbox_transform=axs2.transAxes, borderpad=0.)
        
        axs2.add_artist(anchored_tbox)
        
        tbox = TextArea(
            r'C$_\mathrm{SCN^-}-$K$^\mathrm{+}$', 
            textprops=dict(color='k', fontsize=MEDIUM_SIZE))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.02, 0.64),
            bbox_transform=axs2.transAxes, borderpad=0.)
        
        axs2.add_artist(anchored_tbox)
        
        tbox = TextArea(
            'C', 
            textprops=dict(color='k', fontsize=18, ha='left'))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.02, 0.84),
            bbox_transform=axs3.transAxes, borderpad=0.)
        
        axs3.add_artist(anchored_tbox)
        
        tbox = TextArea(
            '\n' + r'C$_\mathrm{SCN^-}-$O$_\mathrm{H_2O}$', 
            textprops=dict(color='k', fontsize=MEDIUM_SIZE, ha='right'))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.98, 0.84),
            bbox_transform=axs3.transAxes, borderpad=0.)
        
        axs3.add_artist(anchored_tbox)
        
        tbox = TextArea(
            'D', 
            textprops=dict(color='k', fontsize=18, ha='left'))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.02, 0.84),
            bbox_transform=axs4.transAxes, borderpad=0.)
        
        axs4.add_artist(anchored_tbox)
        
        tbox = TextArea(
            '\n' + r'C$_\mathrm{SCN^-}-$N$_\mathrm{acetamide}$', 
            textprops=dict(color='k', fontsize=MEDIUM_SIZE, ha='right'))
    
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(0.98, 0.84),
            bbox_transform=axs4.transAxes, borderpad=0.)
        
        axs4.add_artist(anchored_tbox)
    
    if isys==(len(info_systems)-1):
        axs2.legend(
            loc=[0.72, 0.22], ncol=1, framealpha=1.0, fontsize=SMALL_SIZE-2)

plt.savefig(
    os.path.join(
        res_maindir, 'paper_gdist_300.png'),
    format='png', dpi=dpi)
