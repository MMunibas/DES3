# Basics
import os
import sys
import time
import numpy as np
from itertools import product
from glob import glob

# Trajectory reader
import MDAnalysis

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ASE
from ase import Atoms
from ase.io import read, write
from ase.visualize import view

# Statistics
from statsmodels.tsa.stattools import acovf
from scipy.optimize import curve_fit

#------------
# Parameters
#------------

# Source directory
source = '.'

# Temperatures
temperatures = [300]

# Mixtures
mixtures = [0, 20, 30, 50, 70, 80, 90, 100]

# Number of simulation samples
Nsmpl = 5

# Trajectory source
traj_dcdtag = 'dyna.*.dcd'

# first: split condition, second: index for irun
traj_dcdinfo = ['.', 1]
traj_crdfile = 'step1.crd'
traj_psffile = 'step1.psf'

# Frequency source
freq_resdir = 'results'
freq_resfiletag = 'freq_{:s}_{:d}_{:s}_{:d}_*.dat'
freq_ntasks = 5
freq_vibnum = -1
# Indices: mix, irun, residue, ires, ifile
freq_resfileinfo = ['_', [1, 2, 3, 4, 5]]

# Frequency range 
freq_warning = [1850, 2250]

# Residue of interest
eval_residue = 'SCN'

# Regarding residues
eval_resreg = ['SCN', 'ACEM', 'TIP3', 'POT']
eval_resnum = [3, 9, 3, 1]
eval_ressym = {
    'SCN': ['N', 'C', 'S'],
    'ACEM': ['C', 'C', 'N', 'H', 'H', 'O', 'H', 'H', 'H'],
    'TIP3': ['O', 'H', 'H'],
    'POT': ['K']}

# Writing step size in ps
eval_stepsize = 0.100
eval_maxtme = 2000.0

# Workdir label
worktag = ''

# Result directory
res_maindir = 'results_ffcf'

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
    for irun in range(Nsmpl):
        
        datadir = os.path.join(source, worktag + str(temp), "{:s}_{:d}".format(
            mix, irun))
        
        if os.path.exists(os.path.join(datadir, traj_crdfile)):
            break
    
    # System tag
    tag = '{:s}_{:s}'.format(temp, mix)
    
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
                if res in line:
                    info_res.append(line)
        listres[res] = info_res
        numres[tag][res] = int(len(info_res)/eval_resnum[ires])
        
    # Get residue atom numbers
    atomsint[tag] = {}
    for ires, res in enumerate(all_residues):
        atomsinfo = np.zeros(
            [numres[tag][res], eval_resnum[ires]], dtype=int)
        for ir in range(numres[tag][res]):
            ri = eval_resnum[ires]*ir
            for ia in range(eval_resnum[ires]):
                info = listres[res][ri + ia].split()
                atomsinfo[ir, ia] = int(info[0]) - 1
        atomsint[tag][res] = atomsinfo
        
# Make result directory
if not os.path.exists(res_maindir):
    os.mkdir(res_maindir)
    
#---------------------
# Collect system data
#---------------------

info_systems = np.array(
    list(
        product(
            temperatures, 
            mixtures, 
            list(range(Nsmpl))
            )
        )
    )

# Iterate over systems and resids
for sys in info_systems:
    
    # Data directory
    temp = str(sys[0])
    mix = str(sys[1])
    irun = sys[2]
    
    datadir = os.path.join(source, str(temp), "{:s}_{:d}".format(
        mix, irun))
    print("Start ", datadir)
    
    # System tag
    tag = '{:s}_{:s}'.format(temp, mix)
    
    # Read dcd files 
    #----------------
    
    # Output file
    freqsfile = os.path.join(
        res_maindir, 'freqs_{:s}_{:s}_{:d}_{:s}.npz'.format(
            temp, mix, irun, eval_residue))
    
    print("Already read: ", os.path.exists(freqsfile))
    if os.path.exists(freqsfile):
        continue
    
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
    
    # Prepare data number parameter
    Ndcd = len(iruns)
    Nfrm = None
    Nres = numres[tag][eval_residue]
    
    # Iterate over residues
    for resid in range(numres[tag][eval_residue]):
        
        # Frequency frame counter
        ifrm = 0
        
        # Iterate over dcd files
        for dcdfile in dcdfiles:
            
            idcd = int(
                dcdfile.split('/')[-1].split(traj_dcdinfo[0])[traj_dcdinfo[1]])
            
            # Open dcd file
            dcd = MDAnalysis.Universe(psffile, dcdfile)
            
            # Get trajectory parameter
            Nframes = len(dcd.trajectory)
            Nskip = int(dcd.trajectory.skip_timestep)
            dt = np.round(
                float(dcd.trajectory._ts_kwargs['dt']), decimals=8)/Nskip
            
            # Get frequency files
            eval_freqfiles = np.array(glob(os.path.join(
                datadir, freq_resdir,
                freq_resfiletag.format(mix, idcd, eval_residue, resid))))
            
            # Sort frequency files
            ifreqs = np.array([
                int(freqfile.split('/')[-1].split('.')[0].split('_')[-1]) 
                for freqfile in eval_freqfiles])
            freqsort = np.argsort(ifreqs)
            eval_freqfiles = eval_freqfiles[freqsort]
            ifreqs = ifreqs[freqsort]
            
            # Iterate over frequency files
            for freqfile in eval_freqfiles:
                print(freqfile)
                # Read frequencies of mode 'freq_vibnum'
                with open(freqfile, 'r') as f:
                    freqlines = f.readlines()

                freqs_mode = [
                    float(line.split()[freq_vibnum - 1]) 
                    for line in freqlines if len(line)]
                
                # If not done yet, get Number of frequency frames and 
                # initialize frequency array
                if Nfrm is None:
                    Nfrm = len(freqs_mode)
                    all_frequencies = np.zeros(
                        [Ndcd*Nfrm*len(eval_freqfiles), Nres])
                    dt_frequencies = dt*Nskip//(Nfrm*len(eval_freqfiles))
            
                # Add frequencies to array
                if len(freqs_mode) > Nfrm:
                    all_frequencies[ifrm:(ifrm + Nfrm), resid] = (
                        freqs_mode[(len(freqs_mode) - Nfrm):])
                else:
                    all_frequencies[ifrm:(ifrm + Nfrm), resid] = freqs_mode
                ifrm += Nfrm

    # Store frequencies
    np.savez(freqsfile, frequencies=all_frequencies, dt=dt_frequencies)



#-----------------------------
# Plot FFCF
#-----------------------------

# Fontsize
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE, weight='bold')  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

dpi = 200


color_scheme = [
    'b', 'r', 'g', 'purple', 'orange', 'magenta', 'brown', 'darkblue',
    'darkred', 'darkgreen', 'darkgrey', 'olive']


# Figure arrangement
figsize = (12, 8)
left = 0.12
bottom = 0.15
row = np.array([0.70, 0.00])
column = np.array([0.50, 0.10])


# Initialize figure and axes
fig = plt.figure(figsize=figsize)
axs1 = fig.add_axes([left, bottom, column[0], row[0]])

plt_log = False
pltt = 100.0
fitt = None
tstart = 1.00

tau_slow = []
tau_fast = []

# Iterate over systems
info_systems = np.array(
    list(
        product(
            temperatures, 
            mixtures
            )
        )
    )
for isys, sys in enumerate(info_systems):
    
    # Data directory
    temp = str(sys[0])
    mix = str(sys[1])
    
    # System tag    
    tag = '{:s}_{:s}'.format(temp, mix)
    
    avg_corr = None
    res_corr = []
    maxtime = 0
    
    # Iterate over sample runs
    for ismpl in range(Nsmpl):
    
        # Read frequencies
        freqsfile = os.path.join(
            res_maindir, 'freqs_{:s}_{:s}_{:d}_{:s}.npz'.format(
                temp, mix, ismpl, eval_residue))
        
        result = np.load(freqsfile)
        freqs = result['frequencies']
        #print(freqs[:5, 0])
        # Time step
        dt = result['dt']
        if dt is None or dt == 0.0:
            dt = eval_stepsize
        
        # Reduce to evaluation time
        time = np.arange(0.0, freqs.shape[0]*dt, dt)
        freqs = freqs[time <= eval_maxtme]
        
        # Max time steps
        if freqs.shape[0] > maxtime:
            maxtime = freqs.shape[0]
        
        # Compute correlation
        for ires in range(freqs.shape[1]):
            
            # Normalization
            mean_freq = np.mean(freqs[:,ires])
            denominator = np.mean((freqs[:,ires] - mean_freq)**2)
            
            corr = acovf(freqs[:,ires], fft=True)/denominator
            res_corr.append(corr)
       
    # Get average correlation
    avg_corr = np.zeros(maxtime)
    time = np.arange(0.0, maxtime*dt, dt)
    Ncorr = 0
    
    for corr in res_corr:
        #avg_corr[:len(corr)] += corr
        if len(corr)==maxtime:
            avg_corr += corr
            Ncorr += 1
    avg_corr /= Ncorr

    # Bi-exponential fit
    def func(x, a1, t1, t2, c):
        return a1 * np.exp(-x/t1) + (1.0 - a1) * np.exp(-x/t2) + c
    
    if fitt is None:
        izero = np.min(np.where(avg_corr < 0.0)[0])
        fitt_i = time[izero]
    else:
        fitt_i = fitt.copy()
    
    popt = [0.9, 0.3, 5.0, 0.0]
    log_nsteps = 10
    log_fitt = np.log(fitt_i)
    steps = np.exp(
        np.arange(
            (log_fitt - tstart)/log_nsteps, 
            (log_fitt - tstart) + (log_fitt - tstart)/log_nsteps/2., 
            (log_fitt - tstart)/log_nsteps) 
        + tstart)
    log_corr = np.zeros_like(avg_corr)
    log_corr[avg_corr > 0.0] = np.log(avg_corr[avg_corr > 0.0])
    log_corr[avg_corr <= 0.0] = np.nan
    
    for step in steps:
        
        select_fit = np.logical_and(
            np.logical_and(
                np.logical_not(np.isnan(avg_corr)),
                avg_corr < 0.3
            ),
            np.logical_and(
                time > tstart,
                time < step
            )
        )
        
        #sigma = (1. + avg_corr)
        #print(time[select_fit])
        try:
            
            popt, pcov = curve_fit(
                func, 
                time[select_fit], 
                avg_corr[select_fit], 
                p0=popt,
                #sigma=sigma[select_fit],
                bounds=[
                    [0.0, 0.0, 0.0, 0.0], 
                    [1.0, 100.0, 100.0, 0.001]])
            
        except (RuntimeError, ValueError):
            
            fit_complete = False
            
        else:
            
            fit_complete = True
    
    # Print fit results
    print(
        "T = {:s}K, Mix = {:s}, Fit complete: ".format(temp, mix), 
        fit_complete)
    print("Fit range: {:.3f} - {:.3f} ps".format(tstart, fitt_i))
    print(
        "A_i = [{:.2f}, {:.2f}]\n".format(popt[0], 1.0 - popt[0])
        + "tau_i = [{:.2f}, {:.2f}] ps\n".format(popt[1], popt[2])
        + "Delta = {:.2f}".format(popt[3])
        )
    tau = np.sort([popt[1], popt[2]])
    tau_slow.append(tau[-1])
    tau_fast.append(tau[0])
    
    label = (
        "T = {:s}K, Mix = {:s}\n".format(temp, mix)
        + r'$\tau_{1,2}$' + r'=({:.2f}, {:.2f}) ps'.format(*tau))
    
    if plt_log:
            
        axs1.plot(
            time, log_corr, '-', 
            color=color_scheme[isys], label=label)
    
    else:
        
        axs1.plot(
            time, avg_corr, '-', 
            color=color_scheme[isys], label=label)
    
        if fit_complete:
            
            select_plt = np.logical_and(
                np.logical_not(np.isnan(avg_corr)),
                time < pltt)
            
            fit_corr = func(time[select_plt], *popt)
            
            axs1.plot(
                time[select_plt], fit_corr, '--', 
                color=color_scheme[isys])
    
    axs1.set_xlim(0.0, pltt)
    axs1.set_ylim(0.0, 0.2)
    
    axs1.set_xlabel(r'Time (ps)', fontweight='bold')
    axs1.get_xaxis().set_label_coords(0.5, -0.1)
    if plt_log:
        ylabel = (
            r'ln(FFCF) (ln(ps$^{-1}$))')
    else:
        ylabel = (
            r'FFCF (ps$^{-1}$)')
    axs1.set_ylabel(ylabel, fontweight='bold')
    axs1.get_yaxis().set_label_coords(-0.08, 0.50)
    
axs1.legend(loc='upper right')

figtitle = 'ffcf.png'
plt.savefig(
    os.path.join(
        res_maindir, figtitle),
    format='png', dpi=dpi)
plt.close()
