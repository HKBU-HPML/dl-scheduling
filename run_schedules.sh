job_root=job_configs
job_set="${job_set:-job_set_1}"
current_time=$(date +%Y_%m_%d_%H_%M_%S)

job_list=$job_root/$job_set/job_*.json
i=0
for j in $job_list
do
  echo "running job $j: hostfile=$job_root/$job_set/cluster_j$i job_set=$job_set job_id=$i sh horovod_mpi.sh"
  global_sync=$current_time job_root=$job_root hostfile=$job_root/$job_set/cluster_j$i job_set=$job_set job_id=$i sh horovod_mpi.sh &
  i=$(expr $i + 1)
done
