#!/bin/bash
#SBATCH --job-name=script_jacobian,density_idx-0,slope_idx-0,seed-3
#SBATCH --output=jobs/script_jacobian/slurm/slurm_%j__script_jacobian,density_idx-0,slope_idx-0,seed-3.log
#SBATCH --time=12:00:00
#SBATCH --mem=32G


#########################
#
# Job: script_jacobian,density_idx-0,slope_idx-0,seed-3
# Args: 
# {'density_idx': 0,
#  'job_id': 'script_jacobian,density_idx-0,slope_idx-0,seed-3',
#  'sav_dir': '/scratch/vl1019/icassp2024_scrapl_data',
#  'seed': 3,
#  'slope_idx': 0}
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
    --density_idx=0 \
    --slope_idx=0 \
    --seed=3 \
    --sav_dir=/scratch/vl1019/icassp2024_scrapl_data \
    --job_id=script_jacobian,density_idx-0,slope_idx-0,seed-3


echo "Many thanks to Bea Steers, author of SLURMJOBS."



###### (escape from the singularity @o@ )
echo "@: exiting singularity container"
exit 0;
EOF

