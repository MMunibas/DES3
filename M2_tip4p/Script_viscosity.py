# Basics
import os
import sys
import time
import numpy as np
from itertools import product, groupby
from glob import glob

# Trajectory reader
import MDAnalysis

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

# Statistics
from statsmodels.tsa.stattools import acovf

# Miscellaneous
import ase.units as units

#------------
# Parameters
#------------

# Source directory
source = '.'

# Temperatures
temperatures = [300]

# Mixtures
mixtures = [0, 20, 30, 50, 70, 80, 90, 100]

# Runs
Nrun = 5

# Stress tensor files
strfiles_tag = 'str.*.dat'
strfiles_split = ['.', 1]

# DCD Trajectory files
dcdfiles_tag = 'nvt.*.dcd'
dcdfiles_split = ['.', 1]
psffile = 'step1.psf'

# Result directory
resdir = 'results_viscosity'
resevaldir = 'evalfiles'

# Result files
resfiles_tag = 'stress.{:s}.{:s}.{:s}.npz'

# Residue of interest
eval_residue = 'SCN'

# Regarding residues
eval_resreg = ['SCN', 'ACEM', 'TIP4', 'POT']
eval_resnum = {
    'SCN': 3,
    'ACEM': 9,
    'TIP3': 3,
    'TIP4': 4,
    'POT': 1}
eval_ressym = {
    'SCN': ['N', 'C', 'S'],
    'ACEM': ['C', 'C', 'N', 'H', 'H', 'O', 'H', 'H', 'H'],
    'TIP3': ['O', 'H', 'H'],
    'TIP4': ['O', 'X', 'H', 'H'],
    'POT': ['K']}
eval_reschr = {
    'SCN': [-0.46, -0.36, -0.18],
    'ACEM': [-0.27, 0.55, -0.62, 0.32, 0.30, -0.55, 0.09, 0.09, 0.09],
    'TIP3': [-0.84, 0.42, 0.42],
    'TIP4': [0.0, -1.04, 0.52, 0.52],
    'POT': [1.00]}
eval_resmss = {
    'SCN': [14.01, 12.01, 32.06],
    'ACEM': [12.01, 12.01, 14.01, 1.01, 1.01, 12.01, 1.01, 1.01, 1.01],
    'TIP3': [16.00, 1.01, 1.01],
    'TIP4': [16.00, 0.0, 1.01, 1.01],
    'POT': [39.10]}

plot_residue = r'SCN$^{-}$'
plot_ressym = {
    'SCN': ['N', 'C', 'S', r'SCN$^{-}$'],
    'ACEM': [
        'C', 'C', 'N', r'H$_\mathrm{N}$', r'H$_\mathrm{N}$', 'O', 'H', 'H', 'H',
        'acetamide'],
    'TIP3': ['O', 'H', 'H', r'H$_2$O'],
    'TIP4': ['O', 'X', 'H', 'H', r'H$_2$O'],
    'POT': [r'K$^{+}$', r'K$^{+}$']}

#---------------------
# Collect Data
#---------------------

# Iterate over systems and resids
info_systems = list(product(temperatures, mixtures, range(Nrun)))

# Prepare result directories
if not os.path.exists(resdir):
    os.makedirs(resdir)
    os.makedirs(os.path.join(resdir, resevaldir))
elif not os.path.exists(os.path.join(resdir, resevaldir)):
    os.makedirs(os.path.join(resdir, resevaldir))

for i in range(0, len(info_systems)):
    
    # System info
    temp = str(info_systems[i][0])
    mix = str(info_systems[i][1])
    run = str(info_systems[i][2])
    
    # Data directory
    datadir = os.path.join(source, temp, "{:s}_{:s}".format(mix, run))
    
    if os.path.exists(
        os.path.join(resdir, resevaldir, resfiles_tag.format(temp, mix, run))
    ):
        continue
    
    # System tag
    tag = temp + '_' + mix
   
    # Detect files
    #----------------------

    # Get stress tensor files
    strfiles = np.array(glob(os.path.join(datadir, strfiles_tag)))
    istrs = np.array([
        int(strfile.split('/')[-1].split(strfiles_split[0])[strfiles_split[1]])
        for strfile in strfiles])
    
    # Sort stress tensor files
    strsort = np.argsort(istrs)
    strfiles = strfiles[strsort]
    istrs = istrs[strsort]
    
    # Check stress tensor file sizes
    strsizes = [os.path.getsize(strfile) for strfile in strfiles]
    if len(strsizes) > 1 and strsizes[-1] < strsizes[0]:
        strfiles = strfiles[:-1]
        istrs = istrs[:-1]

    # Get dcd files
    dcdfiles = np.array(glob(os.path.join(datadir, dcdfiles_tag)))
    idcds = np.array([
        int(dcdfile.split('/')[-1].split(dcdfiles_split[0])[dcdfiles_split[1]])
        for dcdfile in dcdfiles])

    if not len(dcdfiles):
        continue

    # Sort dcd files
    dcdsort = np.argsort(idcds)
    dcdfiles = dcdfiles[dcdsort]
    idcds = idcds[dcdsort]
    
    # Read system volume
    #----------------------
    
    # Open dcd file
    dcd = MDAnalysis.Universe(
        os.path.join(datadir, psffile),
        dcdfiles[0])
    
    # Read cell
    data_volume = np.prod(dcd.trajectory[0]._unitcell[:3])

    # Read stress tensor
    #----------------------
    
    # Stress tensor and time list
    data_time = []
    data_stress = []
    
    # Iterate over stress tensor files
    iprint = None
    for ii, istr in enumerate(istrs):
        
        # Stress tensor file path
        strfile = strfiles[ii]
        print(f"Read stress tensor file '{strfile:s}'")
        # Open stress tensor file
        with open(strfile, 'r') as fstr:
            flines = fstr.readlines()

        # Convert to 3x3 stress tensor
        for fline in flines:
            sline = fline.split()
            data_time.append(float(sline[0]))
            data_stress.append(np.array(sline[1:], dtype=float).reshape(3, 3))

    # Store results
    #----------------------
    
    # Convert to numpy arrays
    data_time = np.array(data_time, dtype=float)
    data_stress = np.array(data_stress, dtype=float)
    
    # Set initial time to zero
    data_time -= data_time[0]
    
    # Save results
    np.savez(
        os.path.join(resdir, resevaldir, resfiles_tag.format(temp, mix, run)),
        data_volume=data_volume,
        data_time=data_time,
        data_stress=data_stress)

#---------------------
# Evaluate Data
#---------------------

def moving_average(data_set, npoints):
    if npoints == 0:
        npoints = 1
    weights = np.ones(npoints) / npoints
    return np.convolve(data_set, weights, 'same')

# Iterate over systems and resids
info_systems = list(product(temperatures, mixtures, range(Nrun)))

# Result dictionary
viscosity_results = {}

if not os.path.exists(os.path.join(resdir, 'viscosity_results.npz')):

    for i in range(0, len(info_systems)):
        
        # System info
        temp = str(info_systems[i][0])
        mix = str(info_systems[i][1])
        run = str(info_systems[i][2])

        # Prepare viscosity dictionary
        pair_tag = f"{temp:s}_{mix:s}"
        if pair_tag not in viscosity_results:
            viscosity_results[pair_tag] = []

        # Load results
        data_file = os.path.join(
            resdir, resevaldir, resfiles_tag.format(temp, mix, run))
        if os.path.exists(data_file):
            data = np.load(data_file)
        else:
            continue

        # Volume in m**3 (<- Ang**3)
        data_volume = data['data_volume']*1.e-30
        # time in s (<- ps)
        data_time = data['data_time']*1.e-12
        # Pressure in Pa (<- bar)
        data_stress = data['data_stress']*1.e5

        # Modify stress tensor
        mod_data_stress = data_stress.copy()
        mod_data_stress[:, 0, 0] = 0.5*(
            data_stress[:, 0, 0] - data_stress[:, 1, 1])
        mod_data_stress[:, 1, 1] = 0.5*(
            data_stress[:, 1, 1] - data_stress[:, 2, 2])
        mod_data_stress[:, 2, 2] = 0.5*(
            data_stress[:, 0, 0] - data_stress[:, 2, 2])

        # Compute stress tensor autocorrelation function
        triu_indices = np.triu_indices(3)
        mod_tril_stress = np.array([
            mod_data_stress_i[triu_indices] 
            for mod_data_stress_i in mod_data_stress])
        covf_strss = np.array([
            acovf(moving_average(mod_tril_stress_i, 10), fft=True) 
            for mod_tril_stress_i in mod_tril_stress.T])

        # Apply Green-Kubo relation
        kb_JK = units.kB/units.J
        int_covf_strss = []
        for covf_strss_i in covf_strss:
            izero = np.where(covf_strss_i <= 0.0)[0][0]
            int_covf_strss.append(
                np.trapz(covf_strss_i[:izero], x=data_time[:izero]))
        int_covf_strss = np.array(int_covf_strss)
        #int_covf_strss = np.array([
            #np.trapz(covf_strss_i, x=data_time)
            #for covf_strss_i in covf_strss])
        viscosity = (
            data_volume/(6*kb_JK*float(temp))
            * np.sum(int_covf_strss))

        # Store result in cPa
        viscosity_results[pair_tag].append(viscosity*1.e3)

        print(temp, mix, run, viscosity)
        #for covf in covf_strss:
            #plt.plot(data_time*1.e12, covf)
        #plt.xlim(0, 200)
        #plt.savefig(
            #os.path.join(
                #resdir, resevaldir, f"test_{i:d}.png"),
            #format='png')
        #plt.close()

    np.savez(
        os.path.join(resdir, 'viscosity_results.npz'),
        **viscosity_results)

else:
    
    viscosity_results = np.load(os.path.join(resdir, 'viscosity_results.npz'))
    

# Plot viscosity
    
# Fontsize
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE, weight='bold')  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

dpi = 200

# Plot options
color_scheme = {
    0: 'red',
    10: 'gray', 
    20: 'orange', 
    30: 'olive',
    40: 'gray',
    50: 'green',
    60: 'gray', 
    70: 'cyan', 
    80: 'royalblue', 
    90: 'indigo', 
    100: 'purple'}

label_mix = {
    0: '0%',
    10: '10%',
    20: '20%',
    30: '30%',
    40: '40%',
    50: '50%',
    60: '60%',
    70: '70%',
    80: '80%',
    90: '90%',
    100: '100%'}

# Figure arrangement
figsize = (6, 6)

left = 0.20
bottom = 0.15
row = np.array([0.70, 0.00])
column = np.array([0.70, 0.00])

# Initialize figure
fig = plt.figure(figsize=figsize)

# Initialize fit parameter axis
axs = fig.add_axes([
    left,
    bottom, 
    column[0],
    row[0]])

# Plot single results and compute average per mixture
avg_viscosity_results = []
avg_mixtures = []
max_viscosity = 0.0
for pair_tag, results in viscosity_results.items():

    # Get temperature and mixture
    temp, mix = np.array(pair_tag.split('_'), dtype=int)
    
    # Plot single run results
    axs.plot(
        [mix]*len(results), results, 'x', color=color_scheme[mix])

    # Append average result
    avg_viscosity_results.append(np.mean(results))
    avg_mixtures.append(mix)
    
    if np.max(results) > max_viscosity:
        max_viscosity = np.max(results)

# Plot average result
for mix, result in zip(avg_mixtures, avg_viscosity_results):
    axs.plot(
        mix, result, 'o', color=color_scheme[mix], ms=6, 
        label=label_mix[mix])

axs.set_xlim([-10, 110])
axs.set_ylim([0.0, max_viscosity*1.05])

axs.set_xlabel(r'Mixture', fontweight='bold')
axs.get_xaxis().set_label_coords(0.5, -0.10)
axs.set_ylabel(r'Viscosity $\eta$ (cPa or mPa$\cdot$s)', fontweight='bold')
axs.get_yaxis().set_label_coords(-0.15, 0.50)

axs.set_title('Setup TIP4', fontweight='bold')

axs.legend(loc='upper right', title='Mixture')

figtitle = os.path.join(resdir, 'viscosity.png')
plt.savefig(figtitle, format='png', dpi=dpi)
plt.close()




# Plot viscosity
    
# Fontsize
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE, weight='bold')  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

dpi = 200

# Plot options
color_scheme = {
    0: 'red',
    10: 'gray', 
    20: 'orange', 
    30: 'olive',
    40: 'gray',
    50: 'green',
    60: 'gray', 
    70: 'cyan', 
    80: 'royalblue', 
    90: 'indigo', 
    100: 'purple'}

label_mix = {
    0: '0%',
    10: '10%',
    20: '20%',
    30: '30%',
    40: '40%',
    50: '50%',
    60: '60%',
    70: '70%',
    80: '80%',
    90: '90%',
    100: '100%'}

# Figure arrangement
figsize = (6, 6)

left = 0.20
bottom = 0.15
row = np.array([0.75, 0.00])
column = np.array([0.75, 0.00])

# Initialize figure
fig = plt.figure(figsize=figsize)

# Initialize fit parameter axis
axs = fig.add_axes([
    left,
    bottom, 
    column[0],
    row[0]])

# Plot single results and compute average per mixture
avg_viscosity_results = []
avg_mixtures = []
max_viscosity = 0.0
for pair_tag, results in viscosity_results.items():

    # Get temperature and mixture
    temp, mix = np.array(pair_tag.split('_'), dtype=int)
    
    # Plot single run results
    axs.plot(
        [mix]*len(results), results, 'x', color=color_scheme[mix])

    # Append average result
    avg_viscosity_results.append(np.mean(results))
    avg_mixtures.append(mix)
    
    if np.max(results) > max_viscosity:
        max_viscosity = np.max(results)

# Plot average result
for mix, result in zip(avg_mixtures, avg_viscosity_results):
    
    label = r"$\tilde{\eta}$" + f"({label_mix[mix]:>4s})" + " = "
    if result >= 10.0:
        label += f"{result:3.1f}"
    else:
        label += f"{result:3.2f}"
    axs.plot(
        mix, result, 'o', color=color_scheme[mix], ms=6, 
        label=label)

axs.set_xlim([-10, 110])
max_viscosity = 115
axs.set_ylim([0.0, max_viscosity])

axs.set_xlabel(r'Mixture', fontweight='bold')
axs.get_xaxis().set_label_coords(0.5, -0.10)
axs.set_ylabel(r'Viscosity $\eta$ (cPa or mPa$\cdot$s)', fontweight='bold')
axs.get_yaxis().set_label_coords(-0.15, 0.50)

axs.legend(
    loc='upper right', 
    title=r'Mean Viscosity $\tilde{\eta}$ in cPa')

tbox = TextArea(
    'C', 
    textprops=dict(color='k', fontsize='24', ha='right'))
anchored_tbox = AnchoredOffsetbox(
    loc='lower left', child=tbox, pad=0., frameon=False,
    bbox_to_anchor=(-0.2, 1.0),
    bbox_transform=axs.transAxes, borderpad=0.)
#axs.add_artist(anchored_tbox)

figtitle = os.path.join(resdir, 'viscosity_paper_tip4.png')
plt.savefig(figtitle, format='png', dpi=dpi)
plt.close()
