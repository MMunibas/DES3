
# Basics
import os
import sys
import numpy as np

# PyTi
sys.path.append("/data/toepfer/Project_Eutectic/FDCM_calc/tip4/HydrationFreeEnergy") 
sys.path.append("/home/toepfer/data/Project_Eutectic/FDCM_calc/tip4/HydrationFreeEnergy")
from pyti import PyTI

config = {
    "system_filedir": "../m2_scn_tip4p/",
    "system_resfile": "dyna.0.res",
    "system_topfile": "scn_tip4.top",
    "system_parfile": "scn_tip4.par",
    "system_pdbfile": "dyna.0.pdb",
    "system_segid_solute": "LIG",
    "system_segid_solvent": "WAT",
    "system_eletype": "FDCM",
    "system_elefile": "scn_fluc.dcm",
    "system_addfile": ["crystal_image.str", "rkhs_scn_drz.csv", "rkhs_scn_drz.kernel", "rkhs_SCN_rRz.csv"],
    "system_workdir": "",
    "lambda_start": 0.0,
    "lambda_end": 1.0,
    "lambda_dwindow": 0.05,
    "lambda_Nwindow": None,
    "lambda_windows": np.append(np.arange(0.0, 0.05, 0.01), np.arange(0.05, 1.01, 0.05)),
    "dynmcs_time_equi": 50.,
    "dynmcs_time_prod": 150.,
    "dynmcs_time_step": 0.001,
    "dynmcs_save_step": 100,
    "dynmcs_temperature": 300,
    "cmptns_Njobs": 0,
    "cmptns_Ntask": 1,
    "cmptns_cpu_task": 1,
    "cmptns_mem_cpu": 1200,
    "cmptns_system_type": "slurm",
    "cmptns_load_module": "gcc/gcc-9.2.0",
    "cmptns_charmm_dir": None,
    "cmptns_charmm_exe": "/home/toepfer/programs/c49b1-dev-1rkhs/build/cmake/charmm",
    "tmplts_ele_gas": "../templates/tmpl-gas-ele.inp",
    "tmplts_ele_solv": "../templates/tmpl-096-solv-ele.inp",
    "tmplts_vdw_gas": "../templates/tmpl-gas-vdw.inp",
    "tmplts_vdw_solv": "../templates/tmpl-096-solv-vdw.inp",
    }

dG_solver = PyTI(config)
dG_solver.check()
dG_solver.prepare()
dG_solver.run()
