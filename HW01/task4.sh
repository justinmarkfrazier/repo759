#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:00:10
#SBATCH -c 2
#SBATCH -J FirstSlurm
#SBATCH -o FirstSlurm.out -e FirstSlurm.err

hostname