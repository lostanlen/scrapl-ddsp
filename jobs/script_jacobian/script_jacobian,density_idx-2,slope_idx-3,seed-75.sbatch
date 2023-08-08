#!/bin/bash
#SBATCH --job-name=script_jacobian,density_idx-2,slope_idx-3,seed-75
#SBATCH --output=jobs/script_jacobian/slurm/slurm_%j__script_jacobian,density_idx-2,slope_idx-3,seed-75.log
#SBATCH --time=4:00:00
#SBATCH --mem=32G


#########################
#
# Job: script_jacobian,density_idx-2,slope_idx-3,seed-75
# Args: 
# {'density_idx': 2,
#  'job_id': 'script_jacobian,density_idx-2,slope_idx-3,seed-75',
#  'sav_dir': '/scratch/vl1019/icassp2024_scrapl_data',
#  'seed': 75,
#  'slope_idx': 3}
#
#########################




###### ( hop into the singularity o_O )
singularity exec  \
    --overlay /scratch/vl1019/ic24_overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash << EOF
echo "@: entered singularity container"

[[ -f /ext3/env.sh ]] && source /ext3/env.sh
[[ -f /ext3/env ]] && source /ext3/env








echo "Computing JTFS Jacobians"

python script_jacobian.py \
    --density_idx=2 \
    --slope_idx=3 \
    --seed=75 \
    --sav_dir=/scratch/vl1019/icassp2024_scrapl_data \
    --job_id=script_jacobian,density_idx-2,slope_idx-3,seed-75


echo "Many thanks to Bea Steers, author of SLURMJOBS."



###### (escape from the singularity @o@ )
echo "@: exiting singularity container"
exit 0;
EOF

