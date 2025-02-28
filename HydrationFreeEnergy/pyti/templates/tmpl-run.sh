#!/bin/bash
#SBATCH --job-name={JNME}
#SBATCH --partition=long
#SBATCH --nodes={NNDS}
#SBATCH --ntasks={NTSK}
#SBATCH --cpus-per-task={NCPU}
#SBATCH --mem-per-cpu={MEMB}
#SBATCH --exclude=node[109-124]

#--------------------------------------
# Modules
#--------------------------------------

module load {MODL}

my_charmm={CHRM}

ulimit -s 10420

#--------------------------------------
# Prepare Run
#--------------------------------------

export SLURMFILE=slurm-$SLURM_JOBID.out

#--------------------------------------
# Run jobs
#--------------------------------------

srun $my_charmm -i {FINP} -o {FOUT}

# We succeeded, reset trap and clean up normally.
trap - EXIT
exit 0
