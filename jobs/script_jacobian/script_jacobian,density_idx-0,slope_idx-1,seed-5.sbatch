#!/bin/bash
#SBATCH --job-name=script_jacobian,density_idx-0,slope_idx-1,seed-5
#SBATCH --output=jobs/script_jacobian/slurm/slurm_%j__script_jacobian,density_idx-0,slope_idx-1,seed-5.log
#SBATCH --time=2:00:00


#########################
#
# Job: script_jacobian,density_idx-0,slope_idx-1,seed-5
# Args: 
# {'density_idx': 0,
#  'job_id': 'script_jacobian,density_idx-0,slope_idx-1,seed-5',
#  'sav_dir': '/scratch/user/icassp2024_scrapl_data',
#  'seed': 5,
#  'slope_idx': 1}
#
#########################




###### ( hop into the singularity o_O )
singularity exec  \
    --overlay /scratch/user/ic24_overlay-15GB-500K.ext3:ro \
    cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash << EOF
echo "@: entered singularity container"

[[ -f /ext3/env.sh ]] && source /ext3/env.sh
[[ -f /ext3/env ]] && source /ext3/env








echo "Computing JTFS Jacobians"

python script_jacobian.py \
    --density_idx=0 \
    --slope_idx=1 \
    --seed=5 \
    --sav_dir=/scratch/user/icassp2024_scrapl_data \
    --job_id=script_jacobian,density_idx-0,slope_idx-1,seed-5


echo "Many thanks to Bea Steers, author of SLURMJOBS."



###### (escape from the singularity @o@ )
echo "@: exiting singularity container"
exit 0;
EOF

