#!/bin/bash
dnn="${dnn:-resnet20}"
source exp_configs/$dnn.conf
nworkers=4
nwpernode=4
nstepsupdate=1
#PY=/home/comp/csshshi/anaconda2/bin/python
#MPIPATH=/home/comp/csshshi/local/openmpi3.1.1
PY=/home/comp/qiangwang/anaconda2/bin/python
MPIPATH=/home/comp/qiangwang/software/openmpi-3.1.0
#MPIPATH=/home/shshi/local/openmpi3.1.1
#PY=/home/shshi/anaconda2/bin/python
$MPIPATH/bin/mpirun --prefix $MPIPATH -np $nworkers -hostfile cluster$nworkers -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    -x NCCL_P2P_DISABLE=1 \
    $PY horovod_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate --nwpernode $nwpernode 
