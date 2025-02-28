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

if test -f "step1.out"; then
    out=$(grep "CHARMM" step1.out | tail -n 1)
    if grep -q "STOP" <<< "$out"; then
        echo "Initialization already done"
    else
        echo "Restart initialization"
        srun $my_charmm -i step1.inp -o step1.out
    fi
else
    echo "Start initialization"
    srun $my_charmm -i step1.inp -o step1.out
fi

echo "Perform simulation"
srun $my_charmm -i step2.inp -o step2.out

# We succeeded, reset trap and clean up normally.
trap - EXIT
exit 0
