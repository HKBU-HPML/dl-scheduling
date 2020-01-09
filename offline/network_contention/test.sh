#PY=/home/comp/qiangwang/anaconda2/bin/python
MPIPATH=/home/t716/blackjack/software/openmpi-3.1.0
hostfile=cluster2
$MPIPATH/bin/mpirun --prefix $MPIPATH -hostfile $hostfile -bind-to none -map-by slot \
        --mca btl_tcp_if_include eth0 \
        /home/t716/blackjack/dl_scheduling/network_contention/osu_allreduce

