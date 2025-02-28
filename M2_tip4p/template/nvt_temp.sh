#!/bin/bash
#SBATCH --job-name=JNME
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=CNUM
#SBATCH --mem-per-cpu=CMEM

#---------
# Modules
#---------

MDLD
my_charmm="CHRMDIR/build/cmake/charmm"
ulimit -s 10420

#-----------
# Parameter
#-----------

## Your project goes here
export PROJECT=`pwd`

#--------------------------------------------
# Run equilibrium and production simulations
#--------------------------------------------

echo "Perform NVT simulation"
srun $my_charmm -i step4.inp -o step4.out

# We succeeded, reset trap and clean up normally.
trap - EXIT
exit 0
