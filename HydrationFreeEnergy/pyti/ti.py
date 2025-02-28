# TI 

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Basics
import os
import glob
import json
import time
import queue
import shutil
import subprocess
import numpy as np

# Matplotlib
import matplotlib.pyplot as plt

# MDAnalysis
import MDAnalysis as mda

class PyTI():
    """
    Thermodynamic integration scheme to compute solvation free energy.
    """
    
    def __init__(self, config, prepare=False):
        """
        Initialize TI computation
        """
        
        #=======================================================================
        # Read config dictionary or file
        #=======================================================================
        
        # If config is file, read as json format
        if isinstance(config, str):
            
            config_path = config
            if os.path.isfile(config_path):
                with open(config, 'r') as config_file:
                    config_load = json.load(config_file)
            else:
                raise FileNotFoundError("Config input does not point to a file")
            
        # If config is a dictionary, take data
        elif isinstance(config, dict):
            
            config_load = config
            
        else:
            
            raise ValueError(
                "Config input could not be interpreted as dictionary"
                + " or file path!")
        
        # Define default config data
        config_data = {
            "system_filedir": None,
            "system_resfile": None,
            "system_topfile": None,
            "system_parfile": None,
            "system_pdbfile": None,
            "system_segid_solute": None,
            "system_segid_solvent": None,
            "system_eletype": "PC",
            "system_elefile": None,
            "system_addfile": None,
            "system_workdir": None,
            "lambda_start": 0.0,
            "lambda_end": 1.0,
            "lambda_dwindow": 0.05,
            "lambda_Nwindow": None,
            "lambda_windows": None,
            "lambda_thrshld": 0.5,
            "dynmcs_time_equi": 50.,
            "dynmcs_time_prod": 150.,
            "dynmcs_time_step": 0.001,
            "dynmcs_save_step": 100,
            "dynmcs_temperature": 300,
            "cmptns_Njobs": 0,
            "cmptns_nodes": 1,
            "cmptns_Ntask": 1,
            "cmptns_cpu_task": 1,
            "cmptns_mem_cpu": 1000,
            "cmptns_wait_time": 30,
            "cmptns_system_type": "slurm",
            "cmptns_load_module": None,
            "cmptns_charmm_dir": None,
            "cmptns_charmm_exe": None,
            "tmplts_vdw_gas": None,
            "tmplts_vdw_solv": None,
            "tmplts_ele_gas": None,
            "tmplts_ele_solv": None,
            "tmplts_slurm": None,
            }
        
        # Update default config data with specified ones
        config_data.update(config_load)
        
        #=======================================================================
        # Read and check config parameter and input files
        #=======================================================================
        
        # Utilities
        dtype_lst = (tuple, list, np.ndarray)
        dtype_int = (int, np.int16, np.int32, np.int64)
        dtype_flt = (float, np.float16, np.float32, np.float64)
        dtype_num = dtype_int + dtype_flt
        dtype_str = (str)
        
        #-----------------------------------------------------------------------
        # System files
        #-----------------------------------------------------------------------
        
        # Check system files directory
        if config_data["system_filedir"] is not None:
            if not isinstance(config_data["system_filedir"], str):
                raise ValueError(
                    "system_filedir is not a file path!")
            system_filedir = config_data["system_filedir"]
            if not os.path.isdir(system_filedir):
                raise ValueError(
                    "system_filedir directory does not exist!")
        else:
            
            raise ValueError(
                "system_filedir path is not defined!")
        
        # Check system restart file
        if config_data["system_resfile"] is not None:
            if not isinstance(config_data["system_resfile"], str):
                raise ValueError(
                    "system_resfile is not a file path!")
            system_resfile = config_data["system_resfile"]
            if not os.path.isfile(os.path.join(
                config_data["system_filedir"], system_resfile)):
                raise ValueError(
                    "system_resfile file does not exist!")
        else:
            raise ValueError(
                "system_resfile file is not defined!")
        
        # Check system topology file
        if config_data["system_topfile"] is not None:
            if not isinstance(config_data["system_topfile"], str):
                raise ValueError(
                    "system_topfile is not a file path!")
            system_topfile = config_data["system_topfile"]
            if not os.path.isfile(os.path.join(
                config_data["system_filedir"], system_topfile)):
                raise ValueError(
                    "system_topfile file does not exist!")
        else:
            raise ValueError(
                "system_topfile file is not defined!")
        
        # Check system parameter file
        if config_data["system_parfile"] is not None:
            if not isinstance(config_data["system_parfile"], str):
                raise ValueError(
                    "system_parfile is not a file path!")
            system_parfile = config_data["system_parfile"]
            if not os.path.isfile(os.path.join(
                config_data["system_filedir"], system_parfile)):
                raise ValueError(
                    "system_parfile file does not exist!")
        else:
            raise ValueError(
                "system_parfile file is not defined!")
        
        #-----------------------------------------------------------------------
        # PDB files
        #-----------------------------------------------------------------------
        
        # If system pdb file is defined, check for solute and solvent
        if config_data["system_pdbfile"] is not None:
            
            # Check system_pdbfile file
            if isinstance(config_data["system_pdbfile"], str):
                system_pdbfile = os.path.join(
                    config_data["system_filedir"],
                    config_data["system_pdbfile"])
                if not os.path.isfile(system_pdbfile):
                    raise ValueError(
                        "system_pdbfile does not exist!")
            else:
                raise ValueError(
                    "system_pdbfile is not a file path!")
            
            # Load system_pdbfile with MDAnalysis
            mda_system = mda.Universe(system_pdbfile, format='pdb')
            
            # Check system_segid_solute and system_segid_solvent
            if isinstance(config_data["system_segid_solute"], str):
                system_segid_solute = config_data["system_segid_solute"] 
                if not system_segid_solute in mda_system.segments.segids:
                    raise ValueError(
                        "system_segid_solute is not a valid segid in"
                        + " system_pdbfile!")
            else:
                raise ValueError(
                    "system_segid_solute is not a string of a segid!")
            if isinstance(config_data["system_segid_solvent"], str):
                system_segid_solvent = [config_data["system_segid_solvent"]]
                if not system_segid_solvent[0] in mda_system.segments.segids:
                    raise ValueError(
                        "system_segid_solvent is not a valid segid in"
                        " system_pdbfile!")
            elif isinstance(config_data["system_segid_solvent"], dtype_lst):
                for segid in config_data["system_segid_solvent"]:
                    if not isinstance(segid, str):
                        if not segid in mda_system.segments.segids:
                            raise ValueError(
                                "system_segid_solvent does not contain valid"
                                + " segid in system_pdbfile!")
            else:
                raise ValueError(
                    "system_segid_solvent is not string or list of segids!")
            
            # Select pdb_solute
            mda_solute = mda_system.select_atoms(
                "segid {:s}".format(system_segid_solute))
            #mda_solute.write("solute.pdb")
            
            # Select pdb_solvent
            selection = "segid {:s} and "*len(system_segid_solvent)
            mda_solvent = mda_system.select_atoms(
                selection[:-5].format(*system_segid_solvent))
            #mda_solvent.write("solvent.pdb")
            
            # Correct resids
            system_solute_resids = mda_solute.residues.resids
            system_solvent_resids = mda_solvent.residues.resids
            system_solvent_resids += np.max(system_solute_resids)
            mda_solvent.residues.resids = system_solvent_resids
            
        else:
            
            raise ValueError(
                "System pdb file system_pdbfile is not defined!")
        
        #-----------------------------------------------------------------------
        # Electrostatics
        #-----------------------------------------------------------------------
        
        # Check electrostatic interaction type (PC, DCM, FDCM or MTPL)
        options_eletype = ["PC", "DCM", "FDCM", "MTPL"]
        if config_data["system_eletype"] is not None:
            
            if not isinstance(config_data["system_eletype"], str):
                raise ValueError(
                    "system_eletype is not a string!")
            system_eletype = config_data["system_eletype"].upper()
            if not system_eletype in options_eletype:
                raise ValueError(
                    "system_eletype is not a valid electrostatic option!")             
            
        else:
            
            system_eletype = "PC"
            
        # Check electrostatic file (if DCM, FDCM or MTPL)
        if config_data["system_elefile"] is not None:
            
            if isinstance(config_data["system_elefile"], str):
                system_elefile = config_data["system_elefile"]
                if not os.path.isfile(os.path.join(
                    config_data["system_filedir"], system_elefile)):
                    raise ValueError(
                        "system_elefile does not exist!")
            else:
                raise ValueError(
                    "system_elefile is not a file name!")
            
        else:
            
            if config_data["system_eletype"] == "PC":
                system_elefile = ""
            else:
                raise ValueError(
                    "system_elefile is not defined!")
            
        #-----------------------------------------------------------------------
        # Additional files
        #-----------------------------------------------------------------------
        
        # Check the list of additional files needed for simulations
        if config_data["system_addfile"] is not None:
            
            if isinstance(config_data["system_addfile"], str):
                system_addfile = [config_data["system_addfile"]]
            elif isinstance(config_data["system_addfile"], dtype_lst):
                system_addfile = [
                    addfile for addfile in config_data["system_addfile"]]
            else:
                raise ValueError(
                    "system_addfile is not a file or list of files!")
            
            # Check existence of files 
            for addfile in system_addfile:
                if not os.path.exists(os.path.join(
                    config_data["system_filedir"], addfile)):
                    print(
                        "WARNING: File {:s}".format(addfile)
                        + " does not exist!"
                        + " Copying will be omitted!")
            
        else:
            
            system_addfile = []
            
            
        #-----------------------------------------------------------------------
        # Working directory
        #-----------------------------------------------------------------------
        
        # Check system files directory
        if config_data["system_workdir"] is not None:
            if not isinstance(config_data["system_workdir"], str):
                raise ValueError(
                    "system_workdir is not a file path!")
            system_workdir = config_data["system_workdir"]
            if system_workdir == ".":
                system_workdir = ""
        else:
            system_workdir = "runs"
        
        if not os.path.exists(system_workdir) and len(system_workdir):
            os.makedirs(system_workdir)
        
        #-----------------------------------------------------------------------
        # Lambda
        #-----------------------------------------------------------------------
        
        # If lambda windows are defined, adjust other lambda values
        if config_data["lambda_windows"] is not None:
            
            # Read lambda_windows
            if isinstance(config_data["lambda_windows"], dtype_num):
                lambda_windows = np.array([config_data["lambda_windows"]])
            elif isinstance(config_data["lambda_windows"], dtype_lst):
                lambda_windows = np.array(config_data["lambda_windows"])
            else:
                raise ValueError(
                    "lambda_windows are defined, but not a list or array!")
            
            # Sort lambda_windows and assign start end and step number
            lambda_windows = np.sort(lambda_windows)
            lambda_centres = (lambda_windows[:-1] + lambda_windows[1:])/2.
            lambda_start = lambda_windows[0]
            lambda_end = lambda_windows[-1]
            lambda_dwindow = np.unique(
                lambda_windows[1:] - lambda_windows[:-1])
            lambda_Nwindow = len(lambda_windows)
            
        # If lambda windows are not specifically defined
        else:
            
            # Check lambda_start and lambda_end
            if not isinstance(config_data["lambda_start"], dtype_num):
                raise ValueError(
                    "lambda_start is not defined as numeric value!")
            else:
                lambda_start = float(config_data["lambda_start"])
            if not isinstance(config_data["lambda_end"], dtype_num):
                raise ValueError(
                    "lambda_end is not defined as numeric value!")
            else:
                lambda_end = float(config_data["lambda_end"])
            if lambda_end < lambda_start:
                raise ValueError(
                    "lambda_start is larger than lambda_end!")
                
            # If lambda Nwindow is defined, adjust other lambda_dwindow
            if config_data["lambda_Nwindow"] is not None:
                
                # Read lambda_Nwindow
                if isinstance(config_data["lambda_Nwindow"], dtype_int):
                    lambda_Nwindow = config_data["lambda_Nwindow"]
                elif isinstance(config_data["lambda_Nwindow"], dtype_flt):
                    lambda_Nwindow = int(config_data["lambda_Nwindow"])
                else:
                    raise ValueError(
                        "lambda_Nwindow is not defined as numeric value!")
                
                # Check lambda_Nwindow
                if not lambda_Nwindow > 0:
                    raise ValueError(
                        "lambda_Nwindow is not a positive integer!")
                
                # Adjust lambda_dwindow
                lambda_dwindow = (lambda_end - lambda_start)/lambda_Nwindow
                
                # Set lambda_windows
                lambda_windows = np.linspace(
                    lambda_start, lambda_end, num=lambda_Nwindow, endpoint=True)
                
                # Get lambda_centres
                lambda_centres = (lambda_windows[:-1] + lambda_windows[1:])/2.
                
            elif config_data["lambda_dwindow"] is not None:
                
                # Read lambda_dwindow
                if isinstance(config_data["lambda_dwindow"], dtype_flt):
                    lambda_dwindow = config_data["lambda_dwindow"]
                else:
                    raise ValueError(
                        "lambda_dwindow is not defined as float!")
                
                # Check lambda_dwindow
                if not lambda_dwindow > 0.0:
                    raise ValueError(
                        "lambda_dwindow is not a positive float!")
                elif lambda_dwindow > (lambda_end - lambda_start):
                    raise ValueError(
                        "lambda_dwindow is larger than the lambda range!")
                
                # Adjust lambda_Nwindow
                lambda_Nwindow = int(np.around(
                    (lambda_end - lambda_start)/lambda_dwindow)) + 1
                
                # Adjust lambda_dwindow
                lambda_dwindow = (lambda_end - lambda_start)/lambda_Nwindow
                
                # Set lambda_windows
                lambda_windows = np.linspace(
                    lambda_start, lambda_end, num=lambda_Nwindow, endpoint=True)
                
                # Get lambda_centres
                lambda_centres = (lambda_windows[:-1] + lambda_windows[1:])/2.
                
            else:
                
                raise ValueError(
                    "Lambda range could not be defined by input parameters!")
            
        # Energy variance convergence threshold
        if config_data["lambda_thrshld"] is not None:
            
            if isinstance(config_data["lambda_thrshld"], dtype_num):
                lambda_thrshld = float(config_data["lambda_thrshld"])
            else:
                raise ValueError(
                    "lambda_thrshld is not defined as numeric value!")
            
        else:
            
            raise ValueError(
                "lambda_thrshld could not be defined by input parameter!")
        
        
        #-----------------------------------------------------------------------
        # Dynamics options
        #-----------------------------------------------------------------------
        
        # Check equilibration time
        if config_data["dynmcs_time_equi"] is not None:
            
            if isinstance(config_data["dynmcs_time_equi"], dtype_num):
                dynmcs_time_equi = float(config_data["dynmcs_time_equi"])
            else:
                raise ValueError(
                    "dynmcs_time_equi is not defined as numeric value!")
            
        else:
            
            raise ValueError(
                "dynmcs_time_equi could not be defined by input parameter!")
        
        # Check production time
        if config_data["dynmcs_time_prod"] is not None:
            
            if isinstance(config_data["dynmcs_time_prod"], dtype_num):
                dynmcs_time_prod = float(config_data["dynmcs_time_prod"])
            else:
                raise ValueError(
                    "dynmcs_time_prod is not defined as numeric value!")
            
        else:
            
            raise ValueError(
                "dynmcs_time_prod could not be defined by input parameter!")
        
        # Check time step
        if config_data["dynmcs_time_step"] is not None:
            
            if isinstance(config_data["dynmcs_time_step"], dtype_num):
                dynmcs_time_step = float(config_data["dynmcs_time_step"])
            else:
                raise ValueError(
                    "dynmcs_time_step is not defined as numeric value!")
            
        else:
            
            raise ValueError(
                "dynmcs_time_step could not be defined by input parameter!")
        
        # Get number of equilibration and production steps
        dynmcs_Nequi = int(np.ceil(dynmcs_time_equi/dynmcs_time_step))
        dynmcs_Nprod = int(np.ceil(dynmcs_time_prod/dynmcs_time_step))
        
        # Check dynamic parameter
        dynmcs_Nprod_min = 1e5
        if dynmcs_Nprod < dynmcs_Nprod_min:
            print(
                "WARNING: short production run"
                + " ({:d} production steps)".format(dynmcs_Nprod)
                + " requested!"
                + " At least {:d} production steps recommended!".format(
                    dynmcs_Nprod_min))
        dynmcs_time_step_max = 0.002
        if dynmcs_time_step > dynmcs_time_step_max:
            print(
                "WARNING: very short time step"
                + " ({:.4f} ps)".format(dynmcs_time_step)
                + " requested!" 
                + " At least {:.4f} ps recommended!".format(
                    dynmcs_time_step_max))
            
        # Check step interval to save in trajectory
        if config_data["dynmcs_time_step"] is not None:
            
            if isinstance(config_data["dynmcs_save_step"], dtype_int):
                dynmcs_save_step = config_data["dynmcs_save_step"]
            else:
                raise ValueError(
                    "dynmcs_save_step is not an integer value!")
            
        else:
            
            raise ValueError(
                "dynmcs_save_step could not be defined by input parameter!")
            
        # Check temperature
        if config_data["dynmcs_temperature"] is not None:
            
            if isinstance(config_data["dynmcs_temperature"], dtype_num):
                dynmcs_temperature = config_data["dynmcs_temperature"]
            else:
                raise ValueError(
                    "dynmcs_temperature is not defined as numeric value!")
            
        else:
            
            raise ValueError(
                "dynmcs_temperature could not be defined by input parameter!")
        
        #-----------------------------------------------------------------------
        # Computational options
        #-----------------------------------------------------------------------
        
        # Check number of parallel jobs
        if config_data["cmptns_Njobs"] is not None:
            
            if isinstance(config_data["cmptns_Njobs"], dtype_num):
                cmptns_Njobs = int(config_data["cmptns_Njobs"])
            else:
                raise ValueError(
                    "cmptns_Njobs is not defined as integer!")
            
        else:
            
            raise ValueError(
                "cmptns_Njobs is not defined by input parameter!")
        
        # Check number of requested nodes
        if config_data["cmptns_nodes"] is not None:
            
            if isinstance(config_data["cmptns_nodes"], dtype_num):
                cmptns_nodes = int(config_data["cmptns_nodes"])
            else:
                raise ValueError(
                    "cmptns_nodes is not defined as integer!")
            
        else:
            
            cmptns_nodes = 1
        
        # Check number of parallel tasks
        if config_data["cmptns_Ntask"] is not None:
            
            if isinstance(config_data["cmptns_Ntask"], dtype_num):
                cmptns_Ntask = int(config_data["cmptns_Ntask"])
            else:
                raise ValueError(
                    "cmptns_Ntask is not defined as integer!")
            
        else:
            
            cmptns_Ntask = 1
        
        # Check number of cpus per task
        if config_data["cmptns_cpu_task"] is not None:
            
            if isinstance(config_data["cmptns_cpu_task"], dtype_num):
                cmptns_cpu_task = int(config_data["cmptns_cpu_task"])
            else:
                raise ValueError(
                    "cmptns_cpu_task is not defined as integer!")
            
        else:
            
            cmptns_cpu_task = 1
        
        # Check number of memory per cpu in MB
        if config_data["cmptns_mem_cpu"] is not None:
            
            if isinstance(config_data["cmptns_mem_cpu"], dtype_num):
                cmptns_mem_cpu = int(config_data["cmptns_mem_cpu"])
            else:
                raise ValueError(
                    "cmptns_mem_cpu is not defined as integer!")
            
        else:
            
            raise ValueError(
                "cmptns_mem_cpu is not defined by input parameter!")
        
        # Check waiting time to check job completeness
        if config_data["cmptns_wait_time"] is not None:
            
            if isinstance(config_data["cmptns_wait_time"], dtype_num):
                cmptns_wait_time = int(config_data["cmptns_wait_time"])
            else:
                raise ValueError(
                    "cmptns_wait_time is not defined as numeric value!")
            
        else:
            
            raise ValueError(
                "cmptns_wait_time is not defined by input parameter!")
        
        
        # Check computational job system
        options_cmptns = ["local", "slurm"]
        if config_data["cmptns_system_type"] is not None:
            
            if not isinstance(config_data["cmptns_system_type"], str):
                raise ValueError(
                    "cmptns_system_type is not a valid string type!")
            cmptns_system_type = config_data["cmptns_system_type"].lower()
            if not config_data["cmptns_system_type"].lower() in options_cmptns:
                raise ValueError(
                    "cmptns_system_type is not a valid job management system!")
            
        else:
            
            raise ValueError(
                "cmptns_system_type is not defined by input parameter!")
        
        # Check modules to load
        if config_data["cmptns_load_module"] is not None:
            
            if isinstance(config_data["cmptns_load_module"], dtype_str):
                cmptns_load_module = [config_data["cmptns_load_module"]]
            elif not isinstance(config_data["cmptns_load_module"], dtype_lst):
                raise ValueError(
                    "cmptns_load_module is not defined as list of modules!")
            
        else:
            
            cmptns_load_module = []
        
        # Check charmm path
        if config_data["cmptns_charmm_dir"] is not None:
            
            if isinstance(config_data["cmptns_charmm_dir"], dtype_str):
                cmptns_charmm_dir = config_data["cmptns_charmm_dir"]
            else:
                raise ValueError(
                    "cmptns_charmm_dir is not path!")
            
        else:
            
            cmptns_charmm_dir = ""
            
        # Check charmm file
        if config_data["cmptns_charmm_exe"] is not None:
            
            if isinstance(config_data["cmptns_charmm_exe"], dtype_str):
                cmptns_charmm_exe = config_data["cmptns_charmm_exe"]
            else:
                raise ValueError(
                    "cmptns_charmm_exe is not a file!")
            
        else:
            
            cmptns_charmm_exe = "charmm"
            
        # Check charmm executable
        cmptns_charmm = os.path.join(cmptns_charmm_dir, cmptns_charmm_exe)
        
        # TODO Don't know how to check aliases in linux with python
        #if not os.path.isfile(cmptns_charmm):
        
        #=======================================================================
        # Finalize initialization
        #=======================================================================
        
        # Add files to class variables
        self.config_data = config_data
        self.system_filedir = system_filedir
        self.system_resfile = system_resfile
        self.system_topfile = system_topfile
        self.system_parfile = system_parfile
        self.system_pdbfile = system_pdbfile
        self.system_segid_solute = system_segid_solute
        self.system_segid_solvent = system_segid_solvent
        self.mda_system = mda_system
        self.mda_solute = mda_solute
        self.mda_solvent = mda_solvent
        self.system_eletype = system_eletype
        self.system_elefile = system_elefile
        self.system_addfile = system_addfile
        self.lambda_windows = lambda_windows
        self.lambda_centres = lambda_centres
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.lambda_dwindow = lambda_dwindow
        self.lambda_Nwindow = lambda_Nwindow
        self.lambda_thrshld = lambda_thrshld
        self.dynmcs_time_equi = dynmcs_time_equi
        self.dynmcs_time_prod = dynmcs_time_prod
        self.dynmcs_time_step = dynmcs_time_step
        self.dynmcs_Nequi = dynmcs_Nequi
        self.dynmcs_Nprod = dynmcs_Nprod
        self.dynmcs_save_step = dynmcs_save_step
        self.dynmcs_temperature = dynmcs_temperature
        self.cmptns_Njobs = cmptns_Njobs
        self.cmptns_nodes = cmptns_nodes
        self.cmptns_Ntask = cmptns_Ntask
        self.cmptns_cpu_task = cmptns_cpu_task
        self.cmptns_mem_cpu = cmptns_mem_cpu
        self.cmptns_wait_time = cmptns_wait_time
        self.cmptns_system_type = cmptns_system_type
        self.cmptns_load_module = cmptns_load_module
        self.cmptns_charmm_dir = cmptns_charmm_dir
        self.cmptns_charmm_exe = cmptns_charmm_exe
        self.cmptns_charmm = cmptns_charmm
        
        # Prepare already working directories if requested
        self.prepared = False
        if prepare:
            self.prepare()
        
        #=======================================================================
        # Program parameters
        #=======================================================================
        
        # Working directory
        self.workdir = os.path.abspath(system_workdir)
        
        # Simulation directories name tag {vdw/ele}_{gas/solv}_lambda_{1.4f}
        self.sdirtag = "{:s}_{:s}_lambda_{:1.2e}"
        
        # This modules directory
        self.thisdir = os.path.dirname(os.path.abspath(__file__))
        
        # Template file directory
        self.tmpldir = os.path.join(self.thisdir, "templates")
        
        # Not converged simulations directory
        self.ncnvdir = os.path.join(self.workdir, "not_converged")
        
        # CHARMM template files
        if config_data["tmplts_vdw_gas"] is None:
            tmpl_vdw_gas = os.path.join(self.tmpldir, "tmpl-gas-vdw.inp")
        else:
            if not isinstance(config_data["tmplts_vdw_gas"], str):
                raise ValueError(
                    "Custom template file vdw_gas is not a file path!")
            tmpl_vdw_gas = config_data["tmplts_vdw_gas"]
            if not os.path.isfile(tmpl_vdw_gas):
                raise ValueError(
                    "Custom template file vdw_gas does not exists!")
        
        if config_data["tmplts_vdw_solv"] is None:
            tmpl_vdw_solv = os.path.join(self.tmpldir, "tmpl-solv-vdw.inp")
        else:
            if not isinstance(config_data["tmplts_vdw_solv"], str):
                raise ValueError(
                    "Custom template file vdw_solv is not a file path!")
            tmpl_vdw_solv = config_data["tmplts_vdw_solv"]
            if not os.path.isfile(tmpl_vdw_solv):
                raise ValueError(
                    "Custom template file vdw_solv does not exists!")
        
        if config_data["tmplts_ele_gas"] is None:
            tmpl_ele_gas = os.path.join(self.tmpldir, "tmpl-gas-ele.inp")
        else:
            if not isinstance(config_data["tmplts_ele_gas"], str):
                raise ValueError(
                    "Custom template file ele_gas is not a file path!")
            tmpl_ele_gas = config_data["tmplts_ele_gas"]
            if not os.path.isfile(tmpl_ele_gas):
                raise ValueError(
                    "Custom template file ele_gas does not exists!")
        
        if config_data["tmplts_ele_solv"] is None:
            tmpl_ele_solv = os.path.join(self.tmpldir, "tmpl-solv-ele.inp")
        else:
            if not isinstance(config_data["tmplts_ele_solv"], str):
                raise ValueError(
                    "Custom template file ele_solv is not a file path!")
            tmpl_ele_solv = config_data["tmplts_ele_solv"]
            if not os.path.isfile(tmpl_ele_solv):
                raise ValueError(
                    "Custom template file ele_solv does not exists!")
        
        self.ftmp = {
            "vdw": {
                "gas": tmpl_vdw_gas,
                "solv": tmpl_vdw_solv},
            "ele": {
                "gas": tmpl_ele_gas,
                "solv": tmpl_ele_solv}
            }
        
        # CHARMM input files
        self.finp = {
            "vdw": {
                "gas": "gas-vdw.inp",
                "solv": "solv-vdw.inp"},
            "ele": {
                "gas": "gas-ele.inp",
                "solv": "solv-ele.inp"}
            }
        
        # CHARMM output files
        self.fout = {
            "vdw": {
                "gas": "gas-vdw.out",
                "solv": "solv-vdw.out"},
            "ele": {
                "gas": "gas-ele.out",
                "solv": "solv-ele.out"}
            }
        
        # Result data files
        self.fres = {
            "vdw": {
                "gas": "results-gas-vdw_{:1.3f}.dat",
                "solv": "results-solv-vdw_{:1.3f}.dat"},
            "ele": {
                "gas": "results-gas-ele_{:1.3f}.dat",
                "solv": "results-solv-ele_{:1.3f}.dat"}
            }
        
        # Slurm run script template file
        if config_data["tmplts_slurm"] is None:
            self.tmplts_slurm = os.path.join(self.tmpldir, "tmpl-run.sh")
        else:
            if not isinstance(config_data["tmplts_slurm"], str):
                raise ValueError(
                    "Custom template file tmplts_slurm is not a file path!")
            self.tmplts_slurm = config_data["tmplts_slurm"]
            if not os.path.isfile(self.tmplts_slurm):
                raise ValueError(
                    "Custom template file tmplts_slurm does not exists!")
            
        # Slurm run file
        self.frun_slurm = "run.sh"
        
    def check(self):
        """
        Return all important input and related parameter
        """
        
        msg = f"""
        System files directory: {self.system_filedir}
        System restart file: {self.system_resfile}
        System topology file: {self.system_topfile}
        System parameter file: {self.system_parfile}
        System pdb file: {self.system_pdbfile}
        System electrostatic: {self.system_eletype}
        System electrostatic file: {self.system_elefile}
        System additional files: {self.system_addfile}
        
        Solute segid tag: {self.system_segid_solute}
        Solvent segid tags: {self.system_segid_solvent}
        
        Lambda start: {self.lambda_start}
        Lambda end: {self.lambda_end}
        Initial Lambda windows
          Lambda window number: {self.lambda_Nwindow}
          Lambda windows: {self.lambda_windows}
          Lambda window centres: {self.lambda_centres}
        
        Equilibration run time: {self.dynmcs_time_equi} ps
        Production run time: {self.dynmcs_time_prod} ps
        Temperature: {self.dynmcs_temperature} K
        
        Job management system: {self.cmptns_system_type}
        Number of requested nodes: {self.cmptns_nodes}
        Number of parallel tasks: {self.cmptns_Ntask}
        Number of cpus per task: {self.cmptns_cpu_task}
        Number of memory per cpu: {self.cmptns_mem_cpu} MB
        
        Modules to load: {self.cmptns_load_module}
        Charmm executable: {self.cmptns_charmm}
        
        Template files:
          gas_vdw: {self.ftmp["vdw"]["gas"]}
          solv_vdw: {self.ftmp["vdw"]["solv"]}
          gas_ele: {self.ftmp["ele"]["gas"]}
          solv_ele: {self.ftmp["ele"]["solv"]}
        """
        
        print(msg)
        
    def prepare(
        self, lambda_windows=None, interaction_type=None, system_type=None):
        """
        Prepare working directories for dynamic simulations
        """
        
        # Check input
        lambda_windows, interaction_type, system_type = self.check_input(
            lambda_windows, interaction_type, system_type)
        
        # Get all preparation tasks
        allprps = [
            (lambda_w0, lambda_w1, ti, si) 
            for (lambda_w0, lambda_w1) in np.stack(
                (lambda_windows[:-1], lambda_windows[1:])).T
            for ti in interaction_type
            for si in system_type]
            
        # Make working directory
        if not os.path.isdir(self.workdir):
            os.mkdir(self.workdir)
        
        # Iterate over preparations
        for preps in allprps:
            
            # Allocate variables
            lambda_w0 = preps[0]
            lambda_w1 = preps[1]
            lambda_i = (lambda_w0 + lambda_w1)/2.
            ti = preps[2]
            si = preps[3]
            
            # Make simulation directory
            tagi = self.sdirtag.format(ti, si, lambda_i)
            diri = os.path.join(
                self.workdir, tagi)
            if not os.path.isdir(diri):
                os.mkdir(diri)
            
            # Copy files to simulation directory
            for f in [
                self.system_resfile, 
                self.system_topfile, 
                self.system_parfile,
                self.system_elefile,
                *self.system_addfile]:
                
                # Skip if file name is empty
                if not len(f):
                    continue
                
                # Source file
                ifile = os.path.join(self.system_filedir, f)
                # Target file
                jfile = os.path.join(diri, f)
                
                if os.path.isfile(ifile) and not os.path.exists(jfile):
                    shutil.copyfile(ifile, jfile)
                elif os.path.isdir(ifile) and not os.path.exists(jfile):
                    shutil.copytree(ifile, jfile)
                else:
                    print(
                        "Copying source '{:s}' omitted.".format(jfile)
                        + " Target already exists!")
            
            # Save solute pdb file
            system_solute_pdbfile = "solute.pdb"
            self.mda_solute.write(os.path.join(diri, system_solute_pdbfile))
            
            # Save solvate pdb file
            if si == "solv":
                system_solvent_pdbfile = "solvent.pdb"
                self.mda_solvent.write(
                    os.path.join(diri, system_solvent_pdbfile))
            
            # Prepare PERT input
            if ti == "vdw":
                pstart = self.dynmcs_Nequi
                pstop = self.dynmcs_Nequi + self.dynmcs_Nprod
                pertline = (
                    "LSTART {:8.6f} LAMBDA {:8.6f} LSTOP {:8.6f} PSTART {:7d}"
                    + " -\n"
                    + " PSTOP {:7d} PSLOW LINCR {:8.6f} -\n").format(
                        lambda_w0, lambda_i, lambda_w1, 
                        pstart, pstop, lambda_w1 - lambda_w0)
            
            # Read CHARMM input template file
            with open(self.ftmp[ti][si], 'r') as f:
                tmplines = f.read()
            
            # Replace parameters
            # Topology file
            tmplines = tmplines.replace("{FTOP}", "{:s}".format(
                self.system_topfile))
            # Parameter file
            tmplines = tmplines.replace("{FPAR}", "{:s}".format(
                self.system_parfile))
            # Solute PDB file
            tmplines = tmplines.replace("{FPDB_SOLU}", "{:s}".format(
                system_solute_pdbfile))
            # Solvate PDB file
            if si == "solv":
                tmplines = tmplines.replace("{FPDB_SOLV}", "{:s}".format(
                    system_solvent_pdbfile))
            # Electrostatic scheme
            if self.system_eletype == "PC":
                tmplines = tmplines.replace("{ELST}", "")
            elif self.system_eletype == "MTPL":
                mtplines = (
                    "OPEN UNIT 40 CARD READ NAME {:s}\n".format(
                        self.system_elefile)
                    + "MTPL MTPUNIT 40\n"
                    + "CLOSE UNIT 40\n")
                tmplines = tmplines.replace("{ELST}", mtplines)
            elif self.system_eletype == "DCM":
                dcmlines = (
                    "OPEN UNIT 40 CARD READ NAME {:s}\n".format(
                        self.system_elefile)
                    + "DCM IUDCM 40 TSHIFT\n"
                    + "CLOSE UNIT 40\n")
                tmplines = tmplines.replace("{ELST}", dcmlines)
            elif self.system_eletype == "FDCM":
                dcmlines = (
                    "OPEN UNIT 40 CARD READ NAME {:s}\n".format(
                        self.system_elefile)
                    + "DCM FLUX 10 IUDCM 40 TSHIFT\n"
                    + "CLOSE UNIT 40\n")
                tmplines = tmplines.replace("{ELST}", dcmlines)
            # Lambda value
            if ti == "ele":
                tmplines = tmplines.replace("{VLAM}", "{:7.6f}".format(
                    lambda_i))
            # Restart file
            tmplines = tmplines.replace("{FRES}", "{:s}".format(
                self.system_resfile))
            # Time step
            tmplines = tmplines.replace("{TSTEP}", "{:.4f}".format(
                self.dynmcs_time_step))
            # Bath temperature
            tmplines = tmplines.replace("{TBATH}", "{:.2f}".format(
                self.dynmcs_temperature))
            # First temperature
            tmplines = tmplines.replace("{TFRST}", "{:.2f}".format(
                4.*self.dynmcs_temperature/5.))
            # Result data table
            fres = self.fres[ti][si].format(lambda_i)
            tmplines = tmplines.replace("{FDAT}", "{:s}".format(
                fres))
            
            if ti == "vdw":
                # Combined number of simulation steps
                tmplines = tmplines.replace("{NBOTH}", "{:d}".format(
                    self.dynmcs_Nequi + self.dynmcs_Nprod))
                # PERT options
                tmplines = tmplines.replace("{OPERT}", "{:s}".format(
                    pertline))
            elif ti == "ele":
                # Number of equilibration simulation steps
                tmplines = tmplines.replace("{NEQUI}", "{:d}".format(
                    self.dynmcs_Nequi))
                # Number of production simulation steps
                tmplines = tmplines.replace("{NPROD}", "{:d}".format(
                    self.dynmcs_Nprod))
                # Frequency of images stored in trajectory
                tmplines = tmplines.replace("{NSAVE}", "{:d}".format(
                    self.dynmcs_save_step))
                tmplines = tmplines.replace("{NTRAJ}", "{:d}".format(
                    self.dynmcs_Nprod//self.dynmcs_save_step))
        
            # Write CHARMM input file
            with open(os.path.join(diri, self.finp[ti][si]), 'w') as f:
                f.write(tmplines)
            
            # Prepare slurm script if necessary
            if self.cmptns_system_type=="slurm":
                
                # Read slurm template file
                with open(self.tmplts_slurm, 'r') as f:
                    tmplines = f.read()
                    
                # Replace parameters
                # Job name
                tmplines = tmplines.replace("{JNME}", "{:s}".format(
                    tagi))
                # Number of requested nodes
                tmplines = tmplines.replace("{NNDS}", "{:d}".format(
                    self.cmptns_nodes))
                # Number of tasks
                tmplines = tmplines.replace("{NTSK}", "{:d}".format(
                    self.cmptns_Ntask))
                # Number of CPUs per task
                tmplines = tmplines.replace("{NCPU}", "{:d}".format(
                    self.cmptns_cpu_task))
                # Number of requested memory per CPU
                tmplines = tmplines.replace("{MEMB}", "{:d}".format(
                    self.cmptns_mem_cpu))
                # Modules to load
                tmplines = tmplines.replace(
                    "{MODL}", 
                    len(self.cmptns_load_module)*"{:s} ".format(
                        *self.cmptns_load_module))
                # CHARMM executable
                tmplines = tmplines.replace("{CHRM}", "{:s}".format(
                    self.cmptns_charmm))
                # CHARMM input file
                tmplines = tmplines.replace("{FINP}", "{:s}".format(
                    self.finp[ti][si]))
                # CHARMM output file
                tmplines = tmplines.replace("{FOUT}", "{:s}".format(
                    self.fout[ti][si]))
                
                # Write slurm run file
                frun_slurm = os.path.join(diri, self.frun_slurm)
                with open(frun_slurm, 'w') as f:
                    f.write(tmplines)
            
        return
    
    def run(
        self, lambda_windows=None, interaction_type=None, system_type=None,
        converge=False, threshold=None, output=True):
        """
        Run dynamic simulations in working directories and gather results
        """
        
        # Check input
        lambda_windows, interaction_type, system_type = self.check_input(
            lambda_windows, interaction_type, system_type)
        
        # Check convergence threshold
        if converge:
            if threshold is not None:
                if not isinstance(threshold, dtype_num):
                    raise ValueError(
                        "Energy convergence threshold is not a numeric value!")
            else:
                threshold = self.lambda_thrshld
        
        # Initialize the job queue
        jobqueue = queue.Queue()
        
        # Initialize result arrays: 
        # interaction - system - lambda range, mean, variance
        results = {
            "vdw": {
                "gas": [[], [], []],
                "solv": [[], [], []]},
            "ele": {
                "gas": [[], [], []],
                "solv": [[], [], []]}
            }
        
        
        # List all requested computations
        alljobs = [
            (lambda_w0, lambda_w1, ti, si) 
            for (lambda_w0, lambda_w1) in np.stack(
                (lambda_windows[:-1], lambda_windows[1:])).T
            for ti in interaction_type
            for si in system_type]
        for job in alljobs:
            jobqueue.put(job)
            
        # Start jobs depending on system
        if self.cmptns_system_type=="slurm":
            
            # Slurm jobs id list
            slrmids = []
            slrmjob = []
            
            # Iterate over jobs
            done = False
            while not done:
                
                # Check submitted jobs are still running
                squelst = subprocess.run(["squeue"], capture_output=True)
                squeids = [
                    int(job.split()[0])
                    for job in squelst.stdout.decode().split('\n')[1:-1]]
                fnshids = []
                for ii, idx in enumerate(slrmids):
                    if not idx in squeids:
                        fnshids.append(ii)
                        is_converged, mean, variance = self.check_result(
                            jobqueue, slrmjob[ii], converge, threshold)
                        if output:
                            jtag = self.sdirtag.format(
                                slrmjob[ii][2], slrmjob[ii][3], 
                                (slrmjob[ii][0] + slrmjob[ii][1])/2.)
                            print(
                                "Simulation {:d} in {:s} finished!".format(
                                    idx, os.path.join(self.workdir, jtag)))
                        if is_converged:
                            results[slrmjob[ii][2]][slrmjob[ii][3]][0].append(
                                (slrmjob[ii][0], slrmjob[ii][1]))
                            results[slrmjob[ii][2]][slrmjob[ii][3]][1].append(
                                mean)
                            results[slrmjob[ii][2]][slrmjob[ii][3]][2].append(
                                variance)
                        else:
                            if output:
                                jtag = self.sdirtag.format(
                                slrmjob[ii][2], slrmjob[ii][3], 
                                (slrmjob[ii][0] + slrmjob[ii][1])/2.)
                                print(
                                    "Energy convergence for simulation "
                                    + "{:d} in {:s} ".format(
                                        idx, os.path.join(self.workdir, jtag))
                                    + "not achieved!")
                
                # Delete from job list if jobs are finished
                if len(fnshids):
                    slrmids = [
                        task for it, task in enumerate(slrmids) 
                        if not it in fnshids]
                    slrmjob = [
                        jobi for ij, jobi in enumerate(slrmjob) 
                        if not ij in fnshids]
                    
                # Check if 
                # (i)   maximum submission number is reached
                # (ii)  jobs queue is empty but simulation are still running
                # (iii) jobs queue is empty and all simulation are done,
                #       evaluated and no refinement needed
                # -> (i) and (ii) wait and continue, (iii) all done
                if len(slrmids) >= self.cmptns_Njobs and self.cmptns_Njobs != 0:
                    time.sleep(self.cmptns_wait_time)
                    continue
                elif jobqueue.empty() and len(slrmids) > 0:
                    time.sleep(self.cmptns_wait_time)
                    continue
                elif jobqueue.empty() and len(slrmids) == 0:
                    done = True
                    continue
                # (iv)  if jobs still in queue and capacity is there, start job
                
                # Get job grom queue
                job = jobqueue.get()
                
                # Allocate variables
                lambda_w0 = job[0]
                lambda_w1 = job[1]
                lambda_i = (lambda_w0 + lambda_w1)/2.
                ti = job[2]
                si = job[3]
                
                # Working directory
                tagi = self.sdirtag.format(ti, si, lambda_i)
                diri = os.path.join(self.workdir, tagi)
                
                # Input, output and result file
                frun = os.path.join(diri, self.frun_slurm)
                finp = os.path.join(diri, self.finp[ti][si])
                fout = os.path.join(diri, self.fout[ti][si])
                fres = os.path.join(diri, self.fres[ti][si].format(lambda_i))
                
                # Check if result already exist and not empty
                if os.path.exists(fres) and os.stat(fres).st_size != 0:
                    is_converged, mean, variance = self.check_result(
                        jobqueue, job, converge, threshold)
                    if output:
                        print(
                            "Simulation in {:s} already done!".format(
                                diri))
                    if is_converged:
                        results[ti][si][0].append((lambda_w0, lambda_w1))
                        results[ti][si][1].append(mean)
                        results[ti][si][2].append(variance)
                    else:
                        if output:
                            print(
                                "Energy convergence for simulation "
                                + "in {:s} not achieved!".format(diri))
                    continue
                
                # Submit job
                task = subprocess.run(
                    ["sbatch", frun], capture_output=True, cwd=diri)
                slrmids.append(int(task.stdout.decode().split()[-1]))
                slrmjob.append(job)
                
        elif self.cmptns_system_type=="local":
            
            # Slurm jobs id list
            locltsk = []
            locljob = []
            
            # Adept parallel jobs number
            if self.cmptns_Njobs == 0:
                cmptns_Njobs = os.cpu_count()
            else:
                cmptns_Njobs = self.cmptns_Njobs
            
            # Iterate over jobs
            done = False
            while not done:
                
                # Check if tasks are still running
                fnshtsk = []
                for ii, task in enumerate(locltsk):
                    if task.poll() is not None:
                        fnshtsk.append(ii)
                        is_converged, mean, variance = self.check_result(
                            jobqueue, locljob[ii], converge, threshold)
                        if output:
                            jtag = self.sdirtag.format(
                                locljob[ii][2], locljob[ii][3], 
                                (locljob[ii][0] + locljob[ii][1])/2.)
                            print(
                                "Simulation in {:s} finished!".format(
                                    os.path.join(self.workdir, jtag)))
                        if is_converged:
                            results[locljob[ii][2]][locljob[ii][3]][0].append(
                                (locljob[ii][0], locljob[ii][1]))
                            results[locljob[ii][2]][locljob[ii][3]][1].append(
                                mean)
                            results[locljob[ii][2]][locljob[ii][3]][2].append(
                                variance)
                        else:
                            if output:
                                jtag = self.sdirtag.format(
                                    locljob[ii][2], locljob[ii][3], 
                                    (locljob[ii][0] + locljob[ii][1])/2.)
                                print(
                                    "Energy convergence in {:s} ".format(
                                        os.path.join(self.workdir, jtag))
                                    + "not achieved!")
                
                # Delete from job list if jobs are finished
                if len(fnshtsk):
                    locltsk = [
                        task for it, task in enumerate(locltsk) 
                        if not it in fnshtsk]
                    locljob = [
                        jobi for ij, jobi in enumerate(locljob) 
                        if not ij in fnshtsk]
                
                # Check if 
                # (i)   maximum submission number is reached
                # (ii)  jobs queue is empty but simulation are still running
                # (iii) jobs queue is empty and all simulation are done,
                #       evaluated and no refinement needed
                # -> (i) and (ii) wait and continue, (iii) all done
                if len(locltsk) >= cmptns_Njobs:
                    time.sleep(self.cmptns_wait_time)
                    continue
                elif jobqueue.empty() and len(locltsk) > 0:
                    time.sleep(self.cmptns_wait_time)
                    continue
                elif jobqueue.empty() and len(locltsk) == 0:
                    done = True
                    continue
                # (iv)  if jobs still in queue and capacity is there, start job
                
                # Get job from queue
                job = jobqueue.get()
                
                # Allocate variables
                lambda_w0 = job[0]
                lambda_w1 = job[1]
                lambda_i = (lambda_w0 + lambda_w1)/2.
                ti = job[2]
                si = job[3]
                
                # Working directory
                tagi = self.sdirtag.format(ti, si, lambda_i)
                diri = os.path.join(self.workdir, tagi)
                
                # Input, output and result file
                finp = os.path.join(diri, self.finp[ti][si])
                fout = os.path.join(diri, self.fout[ti][si])
                fres = os.path.join(diri, self.fres[ti][si].format(lambda_i))
                
                # Check if result already exist and not empty
                if os.path.exists(fres) and os.stat(fres).st_size != 0:
                    is_converged, mean, variance = self.check_result(
                        jobqueue, job, converge, threshold)
                    if output:
                        print(
                            "Simulation in {:s} already done!".format(
                                diri))
                    if is_converged:
                        results[ti][si][0].append((lambda_w0, lambda_w1))
                        results[ti][si][1].append(mean)
                        results[ti][si][2].append(variance)
                    else:
                        if output:
                            print(
                                "Energy convergence in {:s} ".format(
                                    diri)
                                + "not achieved!")
                    continue
                
                # Run job script
                task = subprocess.Popen(
                    [self.cmptns_charmm, "-i", finp, "-o", fout], cwd=diri)
                locltsk.append(task)
                locljob.append(job)
                
        # Evaluate results
        contributions = [False, False, False, False]
        lambda_integral = {
            "vdw": {
                "gas": 0.0,
                "solv": 0.0},
            "ele": {
                "gas": 0.0,
                "solv": 0.0}
            }
        solvation_free_energy = 0.0
        for ti in interaction_type:
            for si in system_type:
                
                # Gather lambda windows, free energies and variance
                lambda_windows = np.array(results[ti][si][0])
                lambda_energy = results[ti][si][1]
                lambda_variance = results[ti][si][2]
                lambda_centres = (
                    (lambda_windows[:,0] + lambda_windows[:,1])/2.)
                
                # Summation over lambda windows
                for iw, (lw1, lw2) in enumerate(lambda_windows):
                    lambda_integral[ti][si] += lambda_energy[iw]*(lw2 - lw1)
                #lambda_integral[ti][si] = np.trapz(
                    #lambda_energy, x=lambda_centres)
                
                # Print result
                resline = (
                    "Interaction type '{:s}', system type '{:s}': \n".format(
                        ti, si)
                    + " Free Energy: {:.3f} kcal/mol\n".format(
                        lambda_integral[ti][si])
                    + " Average (max) variance: "
                    + "{:.3f} ({:.3f}) kcal/mol".format(
                        np.mean(lambda_variance), np.max(lambda_variance)))
                print(resline)
                
                # Add to result value
                if si == "solv":
                    solvation_free_energy += lambda_integral[ti][si]
                    if ti == "vdw":
                        contributions[0] = True
                    elif ti == "ele":
                        contributions[1] = True
                elif si == "gas":
                    solvation_free_energy -= lambda_integral[ti][si]
                    if ti == "vdw":
                        contributions[2] = True
                    elif ti == "ele":
                        contributions[3] = True
        
        # Print solvation free energy if all contributions are included
        if np.all(contributions):
            
            # Solvent label
            if len(self.system_segid_solvent) > 1:
                system_segid_solvent = (
                    "(" + "{:s}, "*len(self.system_segid_solvent))
                system_segid_solvent = (
                    system_segid_solvent[:-2].format(
                        *self.system_segid_solvent)
                    + ")")
            else:
                system_segid_solvent = self.system_segid_solvent[0]
            
            print(
                "Solvation Free Energy for {:s} in {:s}:\n".format(
                    self.system_segid_solute, system_segid_solvent)
                + "{:.3f} kcal/mol".format(
                    solvation_free_energy))
                
        # Plot the results
        if np.all(contributions):
            
            # Fontsize
            SMALL_SIZE = 14
            MEDIUM_SIZE = 16
            BIGGER_SIZE = 18

            plt.rc('font', size=SMALL_SIZE, weight='bold')
            plt.rc('axes', titlesize=MEDIUM_SIZE)
            plt.rc('axes', labelsize=MEDIUM_SIZE)
            plt.rc('xtick', labelsize=SMALL_SIZE)
            plt.rc('ytick', labelsize=SMALL_SIZE)
            plt.rc('legend', fontsize=SMALL_SIZE)
            plt.rc('figure', titlesize=BIGGER_SIZE)
            
            # PLot resolution
            dpi = 300

            # Solvent label
            if len(self.system_segid_solvent) > 1:
                system_segid_solvent = (
                    "(" + "{:s}, "*len(self.system_segid_solvent))
                system_segid_solvent = (
                    system_segid_solvent[:-2].format(
                        *self.system_segid_solvent)
                    + ")")
            else:
                system_segid_solvent = self.system_segid_solvent[0]
            
            # Plot line style
            line_style = {
                "gas": '--',
                "solv": '-'}
            line_color = {
                "vdw": 'red',
                "ele": 'blue'}
            
            # Initialize figure for energy contributions along lambda
            figsize = (8, 8)
            left = 0.12
            bottom = 0.12
            row = np.array([0.75, 0.00])
            column = np.array([0.75, 0.10])
            
            fig = plt.figure(figsize=figsize)

            axs1 = fig.add_axes([left, bottom, column[0], row[0]])
            
            # Plot energy contributions
            for ti in interaction_type:
                for si in system_type:
                    
                    label = (
                        '{:s}, {:s}, {:.3f} kcal/mol'.format(
                            ti, si, lambda_integral[ti][si]))
                    
                    lambda_windows = np.array(results[ti][si][0])
                    lambda_energy = results[ti][si][1]
                    lambda_variance = results[ti][si][2]
                    lambda_centres = (
                        (lambda_windows[:,0] + lambda_windows[:,1])/2.)
                    
                    idx_sorted = np.argsort(lambda_centres)
                    lambda_centres = np.array(lambda_centres)[idx_sorted]
                    lambda_energy = np.array(lambda_energy)[idx_sorted]
                    
                    axs1.plot(
                        lambda_centres, lambda_energy, 
                        ls=line_style[si], color=line_color[ti],
                        label=label)
                    
            # Plot label
            axs1.set_title(
                "Solvation Free Energy for {:s} in {:s}:\n".format(
                    self.system_segid_solute, system_segid_solvent)
                + "{:.3f} kcal/mol".format(
                    solvation_free_energy), 
                fontweight='bold')
            
            axs1.set_xlabel(
                r'$\lambda$', fontweight='bold')
            axs1.get_xaxis().set_label_coords(0.5, -0.12)
            axs1.set_ylabel('Energy contribution (kcal/mol)', fontweight='bold')
            axs1.get_yaxis().set_label_coords(-0.12, 0.50)
            
            # Plot range
            axs1.set_xlim([0.0, 1.0])
            
            # Legend
            axs1.legend(loc='upper right')
            
            plt.savefig(
                os.path.join(
                    self.workdir, 'solvation_free_energy_contributions.png'),
                format='png', dpi=dpi)
            plt.close()
            
    def check_input(self, lambda_windows, interaction_type, system_type):
        """
        Check Job specification input 
        """
        
        # Check lambda centers
        if lambda_windows is None:
            lambda_windows = self.lambda_windows
        else:
            if isinstance(lambda_windows, dtype_num):
                lambda_windows = np.array([lambda_windows])
            elif isinstance(lambda_windows, dtype_lst):
                lambda_windows = np.array(lambda_windows)
            else:
                raise ValueError(
                    "lambda_windows is not defined as numeric, list or array!")
        lambda_windows = np.sort(lambda_windows)    
        
        # Check interaction and system type
        if interaction_type is not None:
            if isinstance(interaction_type, dtype_str):
                interaction_type = [interaction_type]
            for ti in interaction_type:
                if isinstance(interaction_type, dtype_str):
                   if not ti in ["vdw", "ele"]:
                       raise IOError(
                            "Unknown interaction type ('vdw', 'ele')!")
                else:
                    if not ti in ["vdw", "ele"]:
                       raise IOError(
                            "Invalid interaction type ('vdw', 'ele')!")
        else:
            interaction_type = ["vdw", "ele"]
        
        if system_type is not None:
            if isinstance(system_type, dtype_str):
                system_type = [system_type]
            for si in system_type:
                if isinstance(system_type, dtype_str):
                   if not si in ["gas", "solv"]:
                       raise IOError(
                            "Unknown system type ('gas', 'solv')!")
                else:
                    if not si in ["vdw", "ele"]:
                       raise IOError(
                            "Invalid system type ('gas', 'solv')!")
        else:
            system_type = ["gas", "solv"]
            
        return lambda_windows, interaction_type, system_type
        
    def check_result(self, jobqueue, job, converge, threshold):
        """
        Check convergence of the energy
        """
        
        # Allocate variables
        lambda_w0 = job[0]
        lambda_w1 = job[1]
        lambda_i = (lambda_w0 + lambda_w1)/2.
        ti = job[2]
        si = job[3]
        
        # Working directory
        tagi = self.sdirtag.format(ti, si, lambda_i)
        diri = os.path.join(self.workdir, tagi)
        
        # Input, output and result file
        fres = os.path.join(diri, self.fres[ti][si].format(lambda_i))
        
        # If not result file exist, wait a few seconds as it might still be
        # written
        nwait = 0
        while not os.path.exists(fres):
            time.sleep(5)
            nwait += 1
            if nwait > 10:
                break
        
        # Read result file
        with open(fres, 'r') as f:
            reslines = f.readlines()
        
        # Get variance
        if ti == "vdw":
            
            try:
                mean, variance = np.array(reslines[2].split(), dtype=float)
            except ValueError:
                print(
                    "Simulation did not finish correctly in {:s} !".format(
                    diri))
                return False, 0.0, 0.0
            except:
                raise IOError(
                    "Something else went wrong in {:s} !".format(
                    diri))
            
        elif ti == "ele":
            
            if len(reslines[2:]) == 0:
                print(
                    "Simulation did not finish correctly in {:s} !".format(
                    diri))
                return False, 0.0, 0.0
            
            N = 0
            mean = 0.0
            lvar = np.zeros(len(reslines[2:]), dtype=float)
            for resline in reslines[2:]:
                if len(resline):
                    try:
                        time_step, cele, iele = np.array(
                            resline.split(), dtype=float)
                    except ValueError:
                        print(
                            "Error in reading file {:s} !".format(
                            fres))
                        return False, 0.0, 0.0
                    except:
                        raise IOError(
                            "Something else went wrong in {:s} !".format(
                            diri))
                    lvar[N] = cele + iele
                    mean += lvar[N]
                    N += 1
            
            mean /= N
            variance = np.sum((lvar[:N] - mean)**2)/N
            
        # Decide if variance is lower than threshold else split job
        if not converge:
            
            return True, mean, variance
        
        else:
            
            if variance <= threshold:
                
                # Return results and convergence flag
                return True, mean, variance
            
            else:
                
                # Prepare new simulations
                self.prepare(
                    lambda_windows = [lambda_w0, lambda_i, lambda_w1], 
                    interaction_type=ti, 
                    system_type=si)
                
                # Add to job queue
                newjobs = [
                    (lambda_w0, lambda_i, ti, si),
                    (lambda_i, lambda_w1, ti, si)]
                for job in newjobs:
                    jobqueue.put(job)
                
                # Move working directory to the one for not converged ones
                if not os.path.exists(self.ncnvdir):
                    os.mkdir(self.ncnvdir)
                shutil.move(diri, os.path.join(self.ncnvdir, diri))
                
                # Return results and non-convergence flag
                return False, mean, variance
            
        
            
