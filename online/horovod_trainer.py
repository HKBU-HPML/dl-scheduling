# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import datetime, pause
import torch
import numpy as np
import argparse, os
import settings
import utils
import logging

from dl_trainer import DLTrainer
import horovod.torch as hvd
from dl_job import DLJob

from settings import logger, formatter

def parse_timestamp(time_str):
    year, month, day, hour, minute, second = [int(elem) for elem in time_str.split("_")]
    return datetime.datetime(year, month, day, hour, minute, second, 0) + datetime.timedelta(seconds=15)

def ssgd_with_horovod(job, mode, start_dt):

    rank = hvd.rank()
    gpu_id = -1
    if job.cuda:
        ngpus=1
        gpu_id = job.device_ids[rank]
        torch.cuda.set_device(gpu_id)
    else:
        ngpus=-1

    if rank != 0:
        pretrain = None

    trainer = DLTrainer(rank, dist=False, batch_size=job.batch_size, is_weak_scaling=True, ngpus=ngpus, data_dir=job.data_dir, dataset=job.dataset, dnn=job.dnn, lr=job.lr, nworkers=job.nworkers, prefix='allreduce')

    if mode == 'simulate':
        synt_model = torch.rand(4, job.model_size * (2**20) / 4 / 4)

    init_epoch = torch.ones(1) * trainer.get_train_epoch()
    init_iter = torch.ones(1) * trainer.get_train_iter()
    trainer.set_train_epoch(int(hvd.broadcast(init_epoch, root_rank=0)[0]))
    trainer.set_train_iter(int(hvd.broadcast(init_iter, root_rank=0)[0]))

    optimizer = hvd.DistributedOptimizer(trainer.optimizer, named_parameters=trainer.net.named_parameters())
    hvd.broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    trainer.update_optimizer(optimizer)
    #iters_per_epoch = trainer.get_num_of_training_samples() // (nworkers * batch_size * nsteps_update)

    times = []

    #display = 20 if iters_per_epoch > 20 else iters_per_epoch-1
    display = 1
    nsteps_update = job.nsteps_update

    # to integrate in LSTM code
    hidden = None
    dnn = job.dnn
    if dnn == 'lstm':
        hidden = trainer.net.init_hidden()

    logger.info("(Server %s, Job %d, Rank %d, GPU %d) Wait until start time: %s." % (settings.hostname, job.job_id, rank, gpu_id, start_dt))
    pause.until(start_dt)

    start_slot = time.time()
    for i in range(job.iters):

        s = time.time()
        optimizer.zero_grad()
        optimizer.local = False
        # forward 
        time.sleep(job.get_forward_schedule(rank, i) * 0.001)
        fw_start_slot=int((time.time() - start_slot) * 1000)
        if mode == 'simulate':
            time.sleep((job.fw_time) * 0.001)
        else:
            #optimizer.local = True
            if dnn == 'lstm':
                #print(hidden)
                #print(" =========== j : %d ===========", j)
                _, hidden = trainer.train_forward(1, hidden=hidden)
            else:
                trainer.train_forward(1)
                #trainer.train(1)
        fw_end_slot=int((time.time() - start_slot) * 1000)
        logger.info("(Server %s, Job %d, Rank %d, GPU %d) Forward task %d started at slot=%d, ended at slot=%d, duration=%d." % (settings.hostname, job.job_id, rank, gpu_id, i, fw_start_slot, fw_end_slot, fw_end_slot-fw_start_slot))

        # backward
        time.sleep(job.get_backward_schedule(rank, i) * 0.001)
        bw_start_slot=int((time.time() - start_slot) * 1000)
        if mode == 'simulate':
            time.sleep((job.bw_time) * 0.001)
        else:
            trainer.train_backward(1) 
            #trainer.train(1) 
            pass
        bw_end_slot=int((time.time() - start_slot) * 1000)
        logger.info("(Server %s, Job %d, Rank %d, GPU %d) Backward task %d started at slot=%d, ended at slot=%d, duration=%d." % (settings.hostname, job.job_id, rank, gpu_id, i, bw_start_slot, bw_end_slot, bw_end_slot-bw_start_slot))

        # communication
        time.sleep(job.get_communication_schedule(rank, i) * 0.001)
        comm_start_slot=int((time.time() - start_slot) * 1000)
        if mode == 'simulate':
            hvd.allreduce(synt_model)
            pass
        else:
            trainer.update_model()
        comm_end_slot=int((time.time() - start_slot) * 1000)
        logger.info("(Server %s, Job %d, Rank %d, GPU %d) Comm task %d started at slot=%d, ended at slot=%d, duration=%d." % (settings.hostname, job.job_id, rank, gpu_id, i, comm_start_slot, comm_end_slot, comm_end_slot-comm_start_slot))

        times.append(time.time()-s)
        #if i % display == 0 and i > 0: 
        #    time_per_iter = np.mean(times)
        #    logger.info('Time per iteration including communication: %f. Speed: %f images/s', time_per_iter, job.batch_size * nsteps_update / time_per_iter)
        #    times = []

    end_slot = time.time()
    logger.info("(Server %s, Job %d, Rank %d, GPU %d) Job ended. Total time is % s." % (settings.hostname, job.job_id, rank, gpu_id, int((end_slot - start_slot)*1000)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AllReduce trainer")
    #parser.add_argument('--batch-size', type=int, default=32)
    #parser.add_argument('--nsteps-update', type=int, default=1)
    #parser.add_argument('--nworkers', type=int, default=1, help='Just for experiments, and it cannot be used in production')
    #parser.add_argument('--nwpernode', type=int, default=1, help='Number of workers per node')
    #parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'cifar10', 'an4', 'ptb'], help='Specify the dataset for training')
    #parser.add_argument('--dnn', type=str, default='resnet50', choices=['resnet50', 'resnet20', 'vgg19', 'vgg16', 'alexnet', 'lstman4', 'lstm'], help='Specify the neural network for training')
    #parser.add_argument('--data-dir', type=str, default='./data', help='Specify the data root path')
    #parser.add_argument('--lr', type=float, default=0.1, help='Default learning rate')
    #parser.add_argument('--max-epochs', type=int, default=settings.MAX_EPOCHS, help='Default maximum epochs to train')
    #parser.add_argument('--pretrain', type=str, default=None, help='Specify the pretrain path')
    #parser.add_argument('--num-steps', type=int, default=35)
    #parser.set_defaults(compression=False)
    parser.add_argument('--job-root', type=str, default="job_configs", help='Specify the root of job sets')
    parser.add_argument('--job-set', type=str, default="job_set_1", help='Specify the job set')
    parser.add_argument('--job-id', type=int, default=0, help='Specify the job id')
    parser.add_argument('--mode', type=str, default='simulate', help='Specify the running mode', choices=['real', 'simulate'])
    parser.add_argument('--global-sync', type=str, default='2019_6_1_12_0_0_0', help='global synchronization timestamp')
    args = parser.parse_args()

    # create a job objective
    dl_job = DLJob(args.job_root, args.job_set, args.job_id)

    prefix = settings.PREFIX
    relative_path = './logs/%s/' % args.job_set
    utils.create_path(relative_path)
    rank = 0
    if dl_job.nworkers > 1:
        hvd.init()
        rank = hvd.rank()
    logfile = os.path.join(relative_path, str(args.job_id)+'-'+settings.hostname+'-'+str(rank)+'.log')
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', args)

    start_dt = parse_timestamp(args.global_sync)
    ssgd_with_horovod(dl_job, args.mode, start_dt)
