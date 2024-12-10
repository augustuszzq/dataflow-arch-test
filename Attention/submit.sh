salloc --ntasks=7 --time=0-00:45 --cpus-per-task=28 --account=cis240050p --gres=cs:cerebras:1 \
#srun singularity exec --bind /home,/jet/home/zzhong2 /ocean/neocortex/cerebras/cbcore_latest.sif /bin/bash
srun --pty singularity exec --bind /home,/jet/home/zzhong2 /ocean/neocortex/cerebras/cbcore_latest.sif /bin/bash
# singularity shell --bind /home,/jet/home/zzhong2 /ocean/neocortex/cerebras/cbcore_latest.sif