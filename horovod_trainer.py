# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import torch
import numpy as np
import argparse, os
import settings
import utils
import logging
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm.Set_errhandler(MPI.ERRORS_RETURN)

from dl_trainer import DLTrainer
import horovod.torch as hvd
from dl_job import DLJob

from settings import logger, formatter


def ssgd_with_horovod(job):
    rank = hvd.rank()
    torch.cuda.set_device(job.device_ids[rank])
    if rank != 0:
        pretrain = None

    trainer = DLTrainer(rank, dist=False, batch_size=job.batch_size, is_weak_scaling=True, ngpus=1, data_dir=job.data_dir, dataset=job.dataset, dnn=job.dnn, lr=job.lr, nworkers=job.nworkers, prefix='allreduce')

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
    # todo zhtang ============
    hidden = None
    dnn = job.dnn
    if dnn == 'lstm':
        hidden = trainer.net.init_hidden()
    for i in range(job.iters):
        s = time.time()
        optimizer.zero_grad()
        for j in range(nsteps_update):
            if j < nsteps_update - 1 and nsteps_update > 1:
                optimizer.local = True
            else:
                optimizer.local = False
            # todo zhtang ==========
            if dnn == 'lstm':
                #print(hidden)
                #print(" =========== j : %d ===========", j)
                _, hidden = trainer.train(1, hidden=hidden)
            else:
                trainer.train(1)
        trainer.update_model()
        times.append(time.time()-s)
        if i % display == 0 and i > 0: 
            time_per_iter = np.mean(times)
            logger.info('Time per iteration including communication: %f. Speed: %f images/s', time_per_iter, job.batch_size * nsteps_update / time_per_iter)
            times = []


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
    ssgd_with_horovod(dl_job)
