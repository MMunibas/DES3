import os
import sys
import numpy as np

import subprocess

from shutil import copyfile, copytree
from itertools import product

#------------
# Parameters
#------------

# Number of cpus per task
Ncpu = 4
# Memory per CPU in MP
Nmem = 1000

# Temperatures
temperatures = [300]

# Mixtures
#mixtures = ['1_17', '1_40', '1_83']
mixtures = [0, 20, 30, 50, 70, 80, 90, 100]

# Residues - according to single residue pdb files in source directory
residues = ["scn", "k", "acem", "tip3"]

# System composition according to residues list - dictionary key is system label
composition = {
    '1_17': [50, 50, 0,  840],
    '1_40': [30, 30, 0, 1206],
    '1_83': [20, 20, 0, 1662],
    0:   [75, 75, 225,   0],
    10:  [75, 75, 217,  23],
    20:  [75, 75, 209,  50],
    30:  [75, 75, 198,  81],
    40:  [75, 75, 186, 119],
    50:  [75, 75, 171, 164],
    60:  [75, 75, 153, 219],
    70:  [75, 75, 130, 290],
    80:  [75, 75, 100, 381],
    90:  [75, 75,  59, 506],
    95:  [75, 75,  30, 570],
    100: [75, 75,   0, 685]}

# Initial simulation box size per system composition
boxsize = {
    '1_17': 30.96,
    '1_40': 33.89,
    '1_83': 37.27,
    0:   30.,
    10:  30.,
    20:  30.,
    30:  30.,
    40:  30.,
    50:  30.,
    60:  30.,
    70:  30.,
    80:  30.,
    90:  30.,
    95:  30.,
    100: 30.}

# Time step [ps]
dt = 0.001

# Propagation steps
# Number of simulation samples
Nsmpl = 5
# Heating steps
Nheat = 50000
# Equilibration steps
Nequi = 150000
# Production runs and dynamic steps
Nprod = 50
Ndyna = 100000

# Step size for storing coordinates
twrte = 0.010   # each 10 fs
Nwrte = int(np.round(twrte/dt))

# Instantaneous Normal Mode analysis
Nvtms = 20
Nvdly = 1 
Nvstp = 1

# Maximum number of tasks
Nmaxt = 5

# Number of tasks per dcd file
Ntpdf = 5

# Skip frames
tskpf = 0.100   # each 100 fs
Nskpf = int(np.round(tskpf/dt/Nwrte))

# Optimization steps
Nstps = 100

# Main directory
maindir = os.getcwd()

# Workdir label
worktag = ""

# Template directory
tempdir = "template"

# Source directory
sourdir = "source"

# Packmol directory - if None, packmol in $PATH
packdir = os.path.join(maindir, sourdir, "packmol")
packdir = None

# CHARMM directory and modules to load
chrmmod = "gcc/gcc-9.2.0"
chrmdir = {
    'default': os.path.join(maindir, "../dev-release-75rkhs-fmdcm-hotfix"),
    '1_17': os.path.join(maindir, "../dev-release-dcm-50rkhs"),
    '1_40': os.path.join(maindir, "../dev-release-dcm-30rkhs"),
    '1_83': os.path.join(maindir, "../dev-release-dcm-20rkhs"),
    }

# Additional files from source directory to working directory
addfile = [
    "toppar",
    "toppar.str",
    "crystal_image.str",
    "rkhs_SCN_rRz.csv",
    "ion_scn.lpun"]

#-----------------------------
# Preparations - Systems
#-----------------------------

# List all systems of different conditions
if Nsmpl is None:
    systems = list(product(temperatures, mixtures))
else:
    systems = list(product(temperatures, mixtures, np.arange(Nsmpl)))

#-----------------------------
# Preparations - Directories
#-----------------------------

# Iterate over systems
for sys in systems:
    
    # System data
    temp = sys[0]
    mix = sys[1]
    
    # Generate working directories
    if Nsmpl is None:
        
        workdir = os.path.join(maindir, worktag + str(temp), "{:s}".format(
            str(mix)))
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        
    else:
        
        ismpl = sys[2]
        
        workdir = os.path.join(maindir, worktag + str(temp), "{:s}_{:d}".format(
            str(mix), ismpl))
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        
#-----------------------------
# Preparations - Packmol
#-----------------------------

# Iterate over systems
for sys in systems:
    
    # System data
    temp = sys[0]
    mix = sys[1]
    if Nsmpl is None:
        workdir = os.path.join(maindir, worktag + str(temp), "{:s}".format(
            str(mix)))
    else:
        ismpl = sys[2]
        workdir = os.path.join(maindir, worktag + str(temp), "{:s}_{:d}".format(
            str(mix), ismpl))
    
    # Prepare input file - packmol.inp
    with open(os.path.join(maindir, tempdir, "packmol_temp.inp"), 'r') as f:
        inplines = f.read()
    
    # Set random seed
    inplines = inplines.replace("RRR", '{:d}'.format(np.random.randint(1e6)))
    
    # Generate system composition input
    inppack = ""
    for ir, residue in enumerate(residues):
        
        # Check if residue is in composition
        if composition[mix][ir]==0:
            continue
        
        inppack += "structure {:s}.pdb\n".format(
            os.path.join(maindir, sourdir, residue.lower()))
        inppack += "  number {:d}\n".format(composition[mix][ir])
        halfsize = boxsize[mix]/2.
        inppack += (
            "  inside box {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n".format(
                -halfsize, -halfsize, -halfsize, halfsize, halfsize, halfsize))
        inppack += "end structure\n"
        
    # Set system composition
    inplines = inplines.replace("SSS", inppack)
    
    # Write input file
    with open(os.path.join(workdir, "packmol.inp"), 'w') as f:
        f.write(inplines)
    
    # Execute packmol
    if packdir is None:
        packrun = "packmol"
    else:
        packrun = os.path.join(packdir, "packmol")
        
    os.chdir(workdir)
    pckprc = subprocess.Popen(
        '{:s} < packmol.inp'.format(packrun), shell=True)
    pckprc.communicate()
    os.chdir(maindir)
    
##-----------------------------
## Preparations - Step 1
##-----------------------------

# Iterate over systems
for sys in systems:
    
    # System data
    temp = sys[0]
    mix = sys[1]
    if Nsmpl is None:
        workdir = os.path.join(maindir, worktag + str(temp), "{:s}".format(
            str(mix)))
    else:
        ismpl = sys[2]
        workdir = os.path.join(maindir, worktag + str(temp), "{:s}_{:d}".format(
            str(mix), ismpl))
    
    if len(addfile):
        for f in addfile:
            
            src = os.path.join(maindir, sourdir, f)
            trg = os.path.join(maindir, workdir, f)
            
            if os.path.isfile(src) and not os.path.exists(trg):
                copyfile(src, trg)
            elif os.path.isdir(src) and not os.path.exists(trg):
                copytree(src, trg)
            elif os.path.exists(trg):
                print("File/Directory '{:s}' already exist".format(src))
                print("  Copying source '{:s}' omitted".format(src))
            else:
                print("Copying source '{:s}' omitted".format(src))
    
    # Get single pdb files
    with open(os.path.join(workdir, "init.pdb"), 'r') as f:
        syslines = f.readlines()
    
    # Separate residues
    residf = []
    reslab = []
    for ir, residue in enumerate(residues):
        
        # Get residue identifier
        with open("{:s}.pdb".format(
            os.path.join(maindir, sourdir, residue.lower())), 'r') as f:
            reslines = f.readlines()
        
        for line in reslines:
            if "ATOM" in line:
                residf.append(line.split()[3])
                reslab.append(line.split()[10])
                break
        
        # Check if residue is in composition
        if composition[mix][ir]==0:
            continue
        
        # Get single residue pdb lines
        pdbres = ""
        for line in syslines:
            if "ATOM" in line:
                if residf[ir] in line.split()[3]:
                    pdbres += line
            else:
                pdbres += line
        
        # Write single residue file
        with open(
            os.path.join(workdir, "init.{:s}.pdb").format(residue), 'w') as f:
            f.write(pdbres)
    
    # Prepare input file - step1.inp
    with open(os.path.join(maindir, tempdir, "step1_temp.inp"), 'r') as f:
        inplines = f.read()
    
    # Prepare generation input
    genlines = ""
    for ir, residue in enumerate(residues):
        
        # Check if residue is in composition
        if composition[mix][ir]==0:
            continue
        
        genlines += "! Read {:s}\n".format(residf[ir])
        genlines += "open read card unit 10 name init.{:s}.pdb\n".format(
            residue)
        genlines += "read sequence pdb unit 10\n"
        genlines += "generate {:s} setup warn first none last none\n\n".format(
            reslab[ir])
        genlines += "open read card unit 10 name init.{:s}.pdb\n".format(
            residue)
        genlines += "read coor pdb unit 10 resid\n\n"
        
    # Set system composition
    inplines = inplines.replace("SSS", genlines)
    
    # Write input file
    with open(os.path.join(workdir, "step1.inp"), 'w') as f:
        f.write(inplines)
        

##-----------------------------
## Preparations - Step 2
##-----------------------------

# Iterate over systems
for sys in systems:
    
    # System data
    temp = sys[0]
    mix = sys[1]
    if Nsmpl is None:
        workdir = os.path.join(maindir, worktag + str(temp), "{:s}".format(
            str(mix)))
    else:
        ismpl = sys[2]
        workdir = os.path.join(maindir, worktag + str(temp), "{:s}_{:d}".format(
            str(mix), ismpl))
    
    # Prepare input file - step2.inp
    with open(os.path.join(maindir, tempdir, "step2_temp.inp"), 'r') as f:
        inplines = f.read()
    
    # Prepare parameters
    # Number of SCN residues
    iscn = list(residues).index("scn")
    inplines = inplines.replace('FFF', '{:d}'.format(composition[mix][iscn]))
    # First SCN index
    inplines = inplines.replace('IND', '{:d}'.format(1))
    # Number of SCN residues
    inplines = inplines.replace('NMX', '{:d}'.format(composition[mix][0]))
    # Temperature
    inplines = inplines.replace('XXX', '{:d}'.format(temp))
    # Random seed generator
    inplines = inplines.replace('RRRHH1', str(np.random.randint(1000000)))
    inplines = inplines.replace('RRRHH2', str(np.random.randint(1000000)))
    inplines = inplines.replace('RRRHH3', str(np.random.randint(1000000)))
    inplines = inplines.replace('RRRHH4', str(np.random.randint(1000000)))
    # Step size - Heating
    inplines = inplines.replace('TTT1', '{:d}'.format(Nheat))
    # Step size - Equilibration
    inplines = inplines.replace('TTT2', '{:d}'.format(Nequi))
    # Step size - Production
    inplines = inplines.replace('SSS', '{:.4f}'.format(dt))
    # Step size - Production
    inplines = inplines.replace('TTT3', '{:d}'.format(Ndyna))
    # Step size written to dcd file
    inplines = inplines.replace('NSV', '{:d}'.format(Nwrte))
    # Production runs
    inplines = inplines.replace('NNN', '{:d}'.format(Nprod))
    
    # Write input file
    with open(os.path.join(workdir, "step2.inp"), 'w') as f:
        f.write(inplines)
        
##----------------------------
## Preparations - INM script
##----------------------------

# Iterate over systems
for sys in systems:
    
    # System data
    temp = sys[0]
    mix = sys[1]
    if Nsmpl is None:
        workdir = os.path.join(maindir, worktag + str(temp), "{:s}".format(
            str(mix)))
    else:
        ismpl = sys[2]
        workdir = os.path.join(maindir, worktag + str(temp), "{:s}_{:d}".format(
            str(mix), ismpl))
    
    # Prepare input file - inm_temp.py
    with open(os.path.join(maindir, tempdir, "inm_temp.py"), 'r') as f:
        inplines = f.read()
    
    # Prepare parameters
    # Mixture
    inplines = inplines.replace('%MIX%', '{:s}'.format(str(mix)))
    # DCD selection
    inplines = inplines.replace('%TDCD%', '{:d}'.format(Nvtms))
    inplines = inplines.replace('%DDCD%', '{:d}'.format(Nvdly))
    inplines = inplines.replace('%SDCD%', '{:d}'.format(Nvstp))
    # Production runs
    inplines = inplines.replace('%NPROD%', '{:d}'.format(Nprod))
    # Tasks management
    inplines = inplines.replace('%NMAXT%', '{:d}'.format(Nmaxt))
    inplines = inplines.replace('%NTPDF%', '{:d}'.format(Ntpdf))
    inplines = inplines.replace('%NSKPF%', '{:d}'.format(Nskpf))
    inplines = inplines.replace('%NSTPS%', '{:d}'.format(Nstps))
    # CHARMM version and modules
    if isinstance(chrmmod, (list, tuple, np.ndarray)):
        raise NotImplementedError
    inplines = inplines.replace('%CHMOD%', '{:s}'.format(chrmmod))
    inplines = inplines.replace('%CHDIR%', '{:s}'.format(
        chrmdir.get(mix, chrmdir['default'])))
    
    # Write input file
    with open(os.path.join(workdir, "inm.py"), 'w') as f:
        f.write(inplines)
    
    # Copy template files in working directory
    for f in ['step3_temp.inp', 'step3_temp.sh', 'step3_temp.py']:
        src = os.path.join(maindir, tempdir, f)
        trg = os.path.join(workdir, f)
        copyfile(src, trg)


#-----------------------------
# Preparations - Step 4
#-----------------------------

# Iterate over systems
for sys in systems:
    
    # System data
    temp = sys[0]
    mix = sys[1]
    if Nsmpl is None:
        workdir = os.path.join(maindir, worktag + str(temp), "{:s}".format(
            str(mix)))
    else:
        ismpl = sys[2]
        workdir = os.path.join(maindir, worktag + str(temp), "{:s}_{:d}".format(
            str(mix), ismpl))
    
    # Prepare input file - step2.inp
    with open(os.path.join(maindir, tempdir, "step4_temp.inp"), 'r') as f:
        inplines = f.read()
    
    # Prepare parameters
    # Last NPT production run
    inplines = inplines.replace('LLL', '{:d}'.format(Nprod - 1))
    # Temperature
    inplines = inplines.replace('XXX', '{:d}'.format(temp))
    # Step size - Production
    inplines = inplines.replace('SSS', '{:.4f}'.format(dt))
    # Step size - Production
    inplines = inplines.replace('TTT3', '{:d}'.format(Ndyna))
    # Step size written to dcd file
    inplines = inplines.replace('NSV', '{:d}'.format(Nwrte))
    # Step size stress tensor written to dat file
    inplines = inplines.replace('NSP', '{:d}'.format(Nwrte))
    # Production runs
    inplines = inplines.replace('NNN', '{:d}'.format(Nprod))
    
    # Write input file
    with open(os.path.join(workdir, "step4.inp"), 'w') as f:
        f.write(inplines)

#----------------------------
# Preparations - Run Script
#----------------------------

# Iterate over systems
scrlines = ""
for sys in systems:
    
    # System data
    temp = sys[0]
    mix = sys[1]
    if Nsmpl is None:
        workdir = os.path.join(maindir, worktag + str(temp), "{:s}".format(
            str(mix)))
        jobtag = "{:d}_{:s}".format(temp, str(mix))
    else:
        ismpl = sys[2]
        workdir = os.path.join(maindir, worktag + str(temp), "{:s}_{:d}".format(
            str(mix), ismpl))
        jobtag = "{:d}_{:s}_{:d}".format(temp, str(mix), ismpl)
    
    # Prepare script file
    with open(os.path.join(tempdir, "run_temp.sh"), 'r') as f:
        inplines = f.read()
    
    # Prepare parameters
    inplines = inplines.replace("JNME", jobtag)
    # CPUs
    inplines = inplines.replace('CNUM', '{:d}'.format(Ncpu))
    # Memory per CPU
    inplines = inplines.replace('CMEM', '{:d}'.format(Nmem))
    # Add modules
    if isinstance(chrmmod, (list, tuple, np.ndarray)):
        modlines = ""
        for mod in chrmmod:
            modlines += "module load {:s}\n".format(mod)
    else:
        modlines = "module load {:s}\n".format(chrmmod)
    inplines = inplines.replace('MDLD', modlines)
    # Add CHARMM directory
    inplines = inplines.replace('CHRMDIR', '{:s}'.format(
        chrmdir.get(mix, chrmdir['default'])))
    
    # Write script file
    with open(os.path.join(workdir, 'run.sh'), 'w') as f:
        f.write(inplines)

    # Prepare script file
    with open(os.path.join(tempdir, "nvt_temp.sh"), 'r') as f:
        inplines = f.read()
    
    # Prepare parameters
    inplines = inplines.replace("JNME", jobtag)
    # CPUs
    inplines = inplines.replace('CNUM', '{:d}'.format(Ncpu))
    # Memory per CPU
    inplines = inplines.replace('CMEM', '{:d}'.format(Nmem))
    # Add modules
    if isinstance(chrmmod, (list, tuple, np.ndarray)):
        modlines = ""
        for mod in chrmmod:
            modlines += "module load {:s}\n".format(mod)
    else:
        modlines = "module load {:s}\n".format(chrmmod)
    inplines = inplines.replace('MDLD', modlines)
    # Add CHARMM directory
    inplines = inplines.replace('CHRMDIR', '{:s}'.format(
        chrmdir.get(mix, chrmdir['default'])))
    
    # Write script file
    with open(os.path.join(workdir, 'nvt.sh'), 'w') as f:
        f.write(inplines)

    # Prepare observation file
    with open(os.path.join(tempdir, "observe_temp.py"), 'r') as f:
        inplines = f.read()
    
    # Prepare parameters
    # Script file
    inplines = inplines.replace('%RFILE%', 'run.sh')
    # Input file
    inplines = inplines.replace('%IFILE%', 'step2.inp')
    # output file
    inplines = inplines.replace('%OFILE%', 'step2.out')
    
    # Write input file
    with open(os.path.join(workdir, "observe.py"), 'w') as f:
        f.write(inplines)
    
    # Start lines
    scrlines += "cd {:s}\n".format(workdir)
    #scrlines += "sbatch run.sh\n"
    #scrlines += "python observe.py &\n"
    #scrlines += "python inm.py &\n"
    scrlines += "sbatch nvt.sh\n"
    #scrlines += "sleep 1\n"
    scrlines += "cd {:s}\n".format(maindir)

# Write start file
with open("start.sh", 'w') as f:
    f.write(scrlines)

    
