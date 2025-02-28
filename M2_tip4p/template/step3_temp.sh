#!/bin/bash
#SBATCH --job-name=JBNM
#SBATCH --partition=vshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1400

#---------
# Modules
#---------

MDLD
my_charmm="CHRMDIR/build/cmake/charmm"
ulimit -s 10420

#------------
# Run CHARMM
#------------

srun $my_charmm -i INPF -o OUTF

#----------------
# Run Evaluation
#----------------

python EVAF OUTF RESF FLGF

rm -f INPF EVAF OUTF
