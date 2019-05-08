#!/bin/bash
job_num="${job_num:-4}"
size="${size:-1024}"
iter="${iter:-2000}"
logRoot=logs/job_n${job_num}_s${size}
mkdir -p $logRoot
for ((i=1;i<=$job_num;i++))
do
    date +"%T.%N"
    size=$size iter=$iter ./osu_mpi.sh 1>${logRoot}/job_$i.log 2>&1 &
done
wait
sleep 5
