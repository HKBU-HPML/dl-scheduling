# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
import time
import psutil

import torch
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.utils.data.distributed
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as ct
import settings
import torch.backends.cudnn as cudnn
cudnn.benchmark = False
cudnn.deterministic = True
from settings import logger, formatter
import struct
import models
import logging
import utils
from LR import LRSchedule
#from encoding import huffman
from tensorboardX import SummaryWriter
from datasets import DatasetHDF5

import ptb_reader
import models.lstm as lstmpy
from torch.autograd import Variable
import json

torch.manual_seed(0)
torch.set_num_threads(1)
#writer = None

_support_dataset = ['imagenet', 'cifar10']
_support_cnns = ['resnet20', 'resnet50', 'vgg19', 'alexnet']

NUM_CPU_THREADS=1

process = psutil.Process(os.getpid())


def init_processes(rank, size, backend='tcp', master='gpu10'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = master 
    os.environ['MASTER_PORT'] = '5935'

    #master_ip = "gpu20"
    #master_mt = '%s://%s:%s' % (backend, master_ip, '5955')
    logger.info("initialized trainer rank: %d of %d......" % (rank, size))
    #dist.init_process_group(backend=backend, init_method=master_mt, rank=rank, world_size=size)
    dist.init_process_group(backend=backend, rank=rank, world_size=size)
    logger.info("finished trainer rank: %d......" % rank)

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.name = 'mnistnet'

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

def get_available_gpu_device_ids(ngpus):
    return range(0, ngpus)

def create_net(num_classes, dnn='resnet20', **kwargs):
    ext = None
    if dnn == 'resnet20':
        net = models.__dict__['resnet20'](num_classes=num_classes)
    elif dnn == 'resnet50':
        net = models.__dict__['resnet50'](num_classes=num_classes)
    elif dnn == 'mnistnet':
        net = MnistNet()
    elif dnn == 'vgg16':
        net = models.VGG(dnn.upper())
    elif dnn == 'alexnet':
        #net = models.AlexNet()
        net = torchvision.models.alexnet()
    elif dnn == 'lstman4':
        net, ext = models.LSTMAN4(datapath=kwargs['datapath'])
    elif dnn == 'lstm':
        # model = lstm(embedding_dim=args.hidden_size, num_steps=args.num_steps, batch_size=args.batch_size,
        #              vocab_size=vocab_size, num_layers=args.num_layers, dp_keep_prob=args.dp_keep_prob)
        net = lstmpy.lstm(vocab_size=kwargs['vocab_size'], batch_size=kwargs['batch_size'])

    else:
        errstr = 'Unsupport neural network %s' % dnn
        logger.error(errstr)
        raise errstr 
    return net, ext


class DLTrainer:

    # todo =================
    def __init__(self, rank, master='gpu10', dist=True, ngpus=1, batch_size=32, 
        is_weak_scaling=True, data_dir='./data', dataset='cifar10', dnn='resnet20', 
        lr=0.04, nworkers=1, prefix=None, sparsity=0.95, pretrain=None, num_steps=35):

        self.rank = rank
        #if self.rank == 0:
        #    writer = SummaryWriter()
        self.pretrain = pretrain
        self.dataset = dataset
        self.prefix=prefix
        self.num_steps = num_steps
        self.ngpus = ngpus
        if self.ngpus > 0:
            self.batch_size = batch_size * self.ngpus if is_weak_scaling else batch_size
        else:
            self.batch_size = batch_size
        self.num_batches_per_epoch = -1
        if self.dataset == 'cifar10' or self.dataset == 'mnist':
            self.num_classes = 10
        elif self.dataset == 'imagenet':
            self.num_classes = 1000
        elif self.dataset == 'an4':
            self.num_classes = 29 
        # todo zhtang ==============
        elif self.dataset == 'ptb':
            self.num_classes = 10
        self.nworkers = nworkers # just for easy comparison
        # TODO zhtang =============
        self.data_dir = data_dir
        if type(dnn) != str:
            self.net = dnn
            self.dnn = dnn.name
            self.ext = None # leave for further parameters
        else:
            self.dnn = dnn
            if data_dir is not None:
                self.data_prepare()
            # TODO: Refact these codes!
            if self.dnn == 'lstm':
                self.net, self.ext = create_net(self.num_classes, self.dnn, vocab_size = self.vocab_size, batch_size=self.batch_size)
            elif self.dnn == 'lstman4':
                self.net, self.ext = create_net(self.num_classes, self.dnn, datapath=self.data_dir)
            else:
                self.net, self.ext = create_net(self.num_classes, self.dnn)
        self.lr = lr
        self.base_lr = self.lr
        self.is_cuda = self.ngpus > 0
        if self.is_cuda:
            torch.cuda.manual_seed_all(3000)

        if self.is_cuda:
            if self.ngpus > 1:
                devices = get_available_gpu_device_ids(ngpus)
                self.net = torch.nn.DataParallel(self.net, device_ids=devices).cuda()
            else:
                self.net.cuda()
        self.net.share_memory()
        self.accuracy = 0
        self.loss = 0.0
        self.train_iter = 0
        self.recved_counter = 0
        self.master = master
        self.average_iter = 0
        if dist:
            init_processes(rank, nworkers, master=master)
        if self.dataset != 'an4':
            if self.is_cuda:
                self.criterion = nn.CrossEntropyLoss().cuda()
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            from warpctc_pytorch import CTCLoss
            self.criterion = CTCLoss()
        self.lr_scheduler = getattr(LRSchedule, 'linear')(lr_init=self.lr, epochs=settings.MAX_EPOCHS, extra=0)
        weight_decay = 1e-4
        #if self.dataset == 'imagenet':
        #    weight_decay = 5e-4
        self.m = 0.9 # momentum
        nesterov = False
        if self.dataset != 'an4':
            nesterov = True
        else:
            self.lstman4_lr_epoch_tag = 0
            weight_decay = 0.
        self.optimizer = optim.SGD(self.net.parameters(), 
                lr=self.lr,
                #nesterov=True,
                momentum=self.m, 
                #weight_decay=5e-4)
                weight_decay=weight_decay,
                nesterov=nesterov)

        self.train_epoch = 0

        if self.pretrain is not None and os.path.isfile(self.pretrain):
            self.load_model_from_file(self.pretrain)

        self.sparsities = []
        self.compression_ratios = []
        self.communication_sizes = []
        self.remainer = {}
        self.v = {} # 
        #self.target_sparsities = [0., 0.15, 0.3, 0.6, 0.75, 0.9375, 0.984375, 0.996, 0.999]
        #self.target_sparsities = [0., 0.15, 0.3, 0.6]
        #self.target_sparsities = [0., 0.3, 0.9, 0.95, 0.999]
        #self.target_sparsities = [0., 0.1, 0.15, 0.2, 0.3, 0.5, 0.9, 0.95, 1.]
        self.target_sparsities = [1.]
        self.sparsity = sparsity
        logger.info('target_sparsities: %s', self.target_sparsities)
        self.avg_loss_per_epoch = 0.0
        self.timer = 0.0
        self.iotime = 0.0
        self.epochs_info = []
        self.distributions = {}
        self.gpu_caches = {}
        self.delays = []
        self.num_of_updates_during_comm = 0 
        self.train_acc_top1 = []
        logger.info('num_batches_per_epoch: %d'% self.num_batches_per_epoch)

        # qiang's topk assistant variables
        # calculate the number of parameters
        self.model_size = 0
        for name, param in self.net.state_dict().items():
            self.model_size += param.numel()
        self.residuals = torch.zeros(self.model_size).cuda()  # record the residuals of weight changes
        self.remote_model = {}
        self.remote_indexes = {}
        self.indexes_marked = torch.zeros(self.model_size)
        self.zero_condition = []
        self.values = {}
        self.indexes = {}
        self.recv_indexes = []
    
    # the tensor has been flatten, the tensor here is the gradient
    def compress(self, model, name=None, sigma_scale=2.5, ratio=0.05):
        start = time.time()
        with torch.no_grad():
            #if name not in self.residuals:
            #    self.residuals[name] = torch.zeros_like(gradient.data)
            # add the saved residuals to gradients
            #gradient.add_(self.residuals[name].data)

            # top-k solution
            numel = self.residuals.numel()
            k = int(numel * ratio)
            #print("compress the origin gradients:", self.residuals.norm())
            #print("n:%d, k:%d, r:%d" % (numel, k, ratio))
            values, indexes = torch.topk(torch.abs(self.residuals), k=k)
            #values, indexes = torch.topk(torch.abs(self.residuals), k=numel) # test for the whole model
            #print("select the gradients:", values.norm())
            #values = gradient[indexes]
            #if name not in self.zero_condition:
            #    self.zero_condition[name] = torch.ones(numel, dtype=torch.float32, device=gradient.device) 
            if len(self.zero_condition) == 0:
                self.zero_condition = torch.ones(numel, dtype=torch.float32, device=model.device)

            indexes = indexes.to(torch.long)
            self.zero_condition.fill_(1.0)
            self.zero_condition[indexes] = 0.0

            self.residuals.data.fill_(0.)
            # substract those weight changes that have been selected this time
            self.residuals.data = self.residuals.data * self.zero_condition
            #tensor.sub_(self.residuals[name].data)

            values = model[indexes]
            #self.indexes_marked[indexes] = 1.0
            #logger.info("Marked is %d/%d.", self.indexes_marked.sum(), self.model_size)
            self.values[name] = values
            self.indexes[name] = indexes
            #print("compressed model and result:", indexes[:5], model.norm(), values.norm())
            #logger.info('residuals before: %f', torch.norm(TopKCompressor.residuals[name].data))
            #return values, indexes
            #print("compressed model norm:", model.norm())
	    return values, indexes 

    def get_residuals(self, name, like_tensor):
        if name not in TopKCompressor.residuals:
            TopKCompressor.residuals[name] = torch.zeros_like(like_tensor.data)
        return TopKCompressor.residuals[name]

    def add_residuals(self, included_indexes, name):
        with torch.no_grad():
            residuals = TopKCompressor.residuals[name]
            indexes_t = torch.from_numpy(included_indexes).cuda(residuals.device).long()
            values = TopKCompressor.values[name]
            values[indexes_t] = 0.0
            residuals.data[TopKCompressor.indexes[name]] += values.data
            #logger.info('residuals after: %f', torch.norm(TopKCompressor.residuals[name].data))

    def decompress(self, tensor, ctc, name=None):
        z = tensor 
        return z 


    def get_acc(self):
        return self.accuracy

    def get_loss(self):
        return self.loss

    def get_model_state(self):
        return self.net.state_dict()

    def get_data_shape(self):
        return self._input_shape, self._output_shape

    def get_train_epoch(self):
        return self.train_epoch

    def get_train_iter(self):
        return self.train_iter

    def set_train_epoch(self, epoch):
        self.train_epoch = epoch

    def set_train_iter(self, iteration):
        self.train_iter = iteration

    def load_model_from_file(self, filename):
        checkpoint = torch.load(filename)
        self.net.load_state_dict(checkpoint['state'])
        self.train_epoch = checkpoint['epoch']
        self.train_iter = checkpoint['iter']
        logger.info('Load pretrain model: %s, start from epoch %d and iter: %d', filename, self.train_epoch, self.train_iter)

    def get_num_of_training_samples(self):
        return len(self.trainset)

    def imagenet_prepare(self):
        # Data loading code
        traindir = os.path.join(self.data_dir, 'train')
        testdir = os.path.join(self.data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        #trainset = torchvision.datasets.ImageFolder(traindir, transforms.Compose([
        hdf5fn = os.path.join(self.data_dir, 'imagenet-shuffled.hdf5')
        image_size = 224
        self._input_shape = (self.batch_size, 3, image_size, image_size)
        self._output_shape = (self.batch_size, 1000)
        trainset = DatasetHDF5(hdf5fn, 'train', transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ]))
        self.trainset = trainset

        train_sampler = None
        shuffle = False
        if self.nworkers > 1: 
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size, shuffle=shuffle,
            num_workers=NUM_CPU_THREADS, pin_memory=True, sampler=train_sampler)
        #testset = torchvision.datasets.ImageFolder(testdir, transforms.Compose([
        testset = DatasetHDF5(hdf5fn, 'val', transforms.Compose([
                transforms.ToPILImage(),
        #        transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        self.testset = testset
        self.testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.batch_size, shuffle=False,
            num_workers=1, pin_memory=True)

    def cifar10_prepare(self):
        #transform = transforms.Compose(
        #    [transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        #train_transform = transform
        #test_transform = transform
        image_size = 32
        self._input_shape = (self.batch_size, 3, image_size, image_size)
        self._output_shape = (self.batch_size, 10)
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        test_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
        trainset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True,
                                                download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False,
                                               download=True, transform=test_transform)
        self.trainset = trainset
        self.testset = testset

        train_sampler = None
        shuffle = True
        if self.nworkers > 1: 
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=shuffle, num_workers=NUM_CPU_THREADS, sampler=train_sampler)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=1)
        self.classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def mnist_prepare(self):
        image_size = 28
        self._input_shape = (self.batch_size, 3, image_size, image_size)
        self._output_shape = (self.batch_size, 10)
        trainset = torchvision.datasets.MNIST(self.data_dir, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ]))
        self.trainset = trainset
        testset = torchvision.datasets.MNIST(self.data_dir, train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ]))
        self.testset = testset
        train_sampler = None
        shuffle = True
        if self.nworkers > 1: 
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler

        self.trainloader = torch.utils.data.DataLoader(trainset,
                batch_size=self.batch_size, shuffle=shuffle, num_workers=NUM_CPU_THREADS, sampler=train_sampler)
        self.testloader = torch.utils.data.DataLoader(
                testset,
                batch_size=self.batch_size, shuffle=False, num_workers=1)
    # todo zhtang =======
    def ptb_prepare(self):
        # Data loading code

        # =====================================
        # num_workers=NUM_CPU_THREADS num_workers=1
        # batch_size=self.batch_size
        # num_steps = 35
        # hidden_size = 1500

        # =================================
        raw_data = ptb_reader.ptb_raw_data(data_path=self.data_dir)
        train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
        self.vocab_size = len(word_to_id)


        self._input_shape = (self.batch_size, self.num_steps)
        self._output_shape = (self.batch_size, self.num_steps)

        print('Vocabluary size: {}'.format(self.vocab_size))

        print('load data')

        epoch_size = ((len(train_data) // self.batch_size) - 1) // self.num_steps

        train_set = ptb_reader.TrainDataset(train_data, self.batch_size, self.num_steps)
        self.trainset = train_set
        if self.nworkers > 1: 
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler
        self.trainloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size, shuffle=shuffle,
            num_workers=NUM_CPU_THREADS, pin_memory=True, sampler=train_sampler)


        test_set = ptb_reader.TestDataset(valid_data, self.batch_size, self.num_steps)
        self.testset = test_set
        self.testloader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.batch_size, shuffle=False,
            num_workers=1, pin_memory=True)
        print('=========****** finish getting ptb data===========')

    def an4_prepare(self):
        from audio_data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
        audio_conf = self.ext['audio_conf']
        labels = self.ext['labels']
        train_manifest = os.path.join(self.data_dir, 'an4_train_manifest.csv')
        val_manifest = os.path.join(self.data_dir, 'an4_val_manifest.csv')


        with open('labels.json') as label_file:
            labels = str(''.join(json.load(label_file)))
        trainset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=train_manifest, labels=labels, normalize=True, augment=True)
        self.trainset = trainset
        testset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=val_manifest, labels=labels, normalize=True, augment=False)
        self.testset = testset

        if self.nworkers > 1:
            train_sampler = DistributedBucketingSampler(self.trainset, batch_size=self.batch_size, num_replicas=self.nworkers, rank=self.rank)
        else:
            train_sampler = BucketingSampler(self.trainset, batch_size=self.batch_size)

        self.train_sampler = train_sampler
        trainloader = AudioDataLoader(self.trainset, num_workers=4, batch_sampler=self.train_sampler)
        testloader = AudioDataLoader(self.testset, batch_size=self.batch_size,
                                  num_workers=4)
        self.trainloader = trainloader
        self.testloader = testloader


    def data_prepare(self):
        if self.dataset == 'imagenet':
            self.imagenet_prepare()
        elif self.dataset == 'cifar10':
            self.cifar10_prepare()
        elif self.dataset == 'mnist':
            self.mnist_prepare()
        elif self.dataset == 'an4':
            self.an4_prepare()
        elif self.dataset == 'ptb':
            self.ptb_prepare()
        else:
            errstr = 'Unsupport dataset: %s' % self.dataset
            logger.error(errstr)
            raise errstr
        self.data_iterator = None #iter(self.trainloader)
        self.num_batches_per_epoch = (self.get_num_of_training_samples()+self.batch_size*self.nworkers-1)//(self.batch_size*self.nworkers)
        #self.num_batches_per_epoch = self.get_num_of_training_samples()/(self.batch_size*self.nworkers)

    def update_optimizer(self, optimizer):
        self.optimizer = optimizer
        #self.trainloader = dataloader
        #self.data_iterator = iter(self.trainloader)
        #self.num_batches_per_epoch = (self.get_num_of_training_samples()+self.batch_size*self.nworkers-1)/(self.batch_size*self.nworkers)
        ##self.num_batches_per_epoch = self.get_num_of_training_samples()/(self.batch_size*self.nworkers)
        #self.avg_loss_per_epoch = 0.0
        #self.timer = 0.0
        #self.epochs_info = []
        #logger.info('updated dataloader for SSGD, num_batches_per_epoch: %d'% self.num_batches_per_epoch)

    def update_nworker(self, nworkers, new_rank=-1):
        if new_rank >= 0:
            rank = new_rank
            self.nworkers = nworkers
        else:
            reduced_worker = self.nworkers - nworkers
            rank = self.rank
            if reduced_worker > 0 and self.rank >= reduced_worker:
                rank = self.rank - reduced_worker
        self.rank = rank
        # todo zhtang an4 ====================
        if self.dnn != 'lstman4':
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.trainset, num_replicas=nworkers, rank=rank)
            train_sampler.set_epoch(self.train_epoch)
            shuffle = False
            self.train_sampler = train_sampler
            self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                                      shuffle=shuffle, num_workers=NUM_CPU_THREADS, sampler=train_sampler)
            self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                                     shuffle=False, num_workers=1)
        self.nworkers = nworkers
        self.num_batches_per_epoch = (self.get_num_of_training_samples()+self.batch_size*self.nworkers-1)//(self.batch_size*self.nworkers)

    def data_iter(self):
        try:
            d = self.data_iterator.next()
        except:
            self.data_iterator = iter(self.trainloader)
            d = self.data_iterator.next()
        
        #print(d[0].size())
        #print(d[0].size()[-1], self.batch_size)
        if d[0].size()[0] != self.batch_size:
            return self.data_iter()
        return d

    def _adjust_learning_rate_lstman4(self, progress, optimizer):
        if settings.WARMUP and progress< 5:
            warmup_total_iters = self.num_batches_per_epoch * 5 
            min_lr = self.base_lr / self.nworkers
            lr_interval = (self.base_lr - min_lr) / warmup_total_iters
            self.lr = min_lr + lr_interval * self.train_iter
            #warmuplr = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
            #self.lr = warmuplr[progress]
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
            return 
        if self.lstman4_lr_epoch_tag != progress:
            self.lstman4_lr_epoch_tag = progress 
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 1.01 
            #optim_state = optimizer.state_dict()
            #optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / 1.01
            #optimizer.load_state_dict(optim_state)
            self.lr = self.lr / 1.01

    def _adjust_learning_rate_general(self, progress, optimizer):
        if settings.WARMUP and progress < 5:
            warmup_total_iters = self.num_batches_per_epoch * 5 
            min_lr = self.base_lr / self.nworkers
            lr_interval = (self.base_lr - min_lr) / warmup_total_iters
            self.lr = min_lr + lr_interval * self.train_iter
            #warmuplr = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
            #self.lr = warmuplr[progress]
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
            return self.lr
        first = 81
        second = 122
        third = 155
        if self.dataset == 'imagenet':
            first = 30
            second = 60
            third = 80
        if progress < first: #40:  30 for ResNet-50, 40 for ResNet-20
            lr = self.base_lr
        elif progress < second: #80: 70 for ResNet-50, 80 for ResNet-20
            lr = self.base_lr *0.1
        elif progress < third:
            lr = self.base_lr *0.01
        else:
            lr = self.base_lr *0.001
        #if self.train_iter % self.num_batches_per_epoch != 0:
        #    lr = lr - lr/(self.train_iter % self.num_batches_per_epoch+1)
        self.lr = lr
        if settings.ZHU:
            k = (self.train_iter+1)#*self.nworkers
            lr = 1.0/(np.sqrt(k) * np.log(k))
            max_lr = self.base_lr
            if lr > max_lr:
                lr = max_lr
            self.lr = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr 

    def _adjust_learning_rate_vgg16(self, progress, optimizer):
        if progress > 0 and progress % 25 == 0:
            self.lr = self.base_lr / (2**(progress/25))
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

    def adjust_learning_rate(self, progress, optimizer):
        #if self.dnn == 'vgg16':
        #    return self._adjust_learning_rate_vgg16(progress, optimizer)
        if self.dnn == 'lstman4':
           return self._adjust_learning_rate_lstman4(self.train_iter//self.num_batches_per_epoch, optimizer)        
        return self._adjust_learning_rate_general(progress, optimizer)

    def print_weight_gradient_ratio(self):
        #own_state = self.net.state_dict()
        #for name, param in own_state.items():
        # Tensorboard
        #if self.rank == 0:
        #    for name, param in self.net.named_parameters():
        #        #writer.add_histogram(name, param.clone().cpu().data.numpy(), self.train_iter)
        #        writer.add_histogram(name, param.grad.clone().cpu().data.numpy(), self.train_iter)
        return
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                wn = param.data.norm()
                gn = param.grad.norm() 
                logger.info('[%s] = %f, %f, %f', name, wn, gn, wn/gn)

    def finish(self):
        #writer.close()
        pass

    def cal_accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def train(self, num_of_iters=1, data=None, hidden=None):
        self.loss = 0.0
        s = time.time()
        # todo ============
        # zero the parameter gradients
        #self.optimizer.zero_grad()
        for i in range(num_of_iters):
                # get the input
            # todo zhtang an4 =================================================
            self.adjust_learning_rate(self.train_epoch, self.optimizer)
            if self.train_iter % self.num_batches_per_epoch == 0 and self.train_iter > 0:
                logger.info('train iter: %d, num_batches_per_epoch: %d', self.train_iter, self.num_batches_per_epoch)
                #self.adjust_learning_rate(self.train_epoch, self.optimizer)
                logger.info('Epoch %d, avg train acc: %f, lr: %f, avg loss: %f' % (self.train_iter//self.num_batches_per_epoch, np.mean(self.train_acc_top1), self.lr, self.avg_loss_per_epoch/self.num_batches_per_epoch))
                mean_s = np.mean(self.sparsities)
                if self.train_iter>0 and np.isnan(mean_s):
                    logger.error('NaN detected! sparsities:  %s' % self.sparsities)
                    #sys.exit('NaN detected!!')
                logger.info('Average Sparsity: %f, compression ratio: %f, communication size: %f', np.mean(self.sparsities), np.mean(self.compression_ratios), np.mean(self.communication_sizes))
                self.sparsities = []
                self.compression_ratios = []
                self.communication_sizes = []
                self.train_acc_top1 = []
                #self.test(self.train_epoch)
                self.epochs_info.append(self.avg_loss_per_epoch/self.num_batches_per_epoch)
                self.avg_loss_per_epoch = 0.0
                #self.data_iterator = iter(self.trainloader)
                #if self.train_iter > 0 and self.train_iter % 100 == 0:
                #    self.print_weight_gradient_ratio()
                # Save checkpoint
                if self.train_iter > 0 and self.train_epoch % 5 == 0 and self.rank == 0:
                    state = {'iter': self.train_iter, 'epoch': self.train_epoch, 'state': self.get_model_state()}
                    if self.prefix:
                        relative_path = './weights/%s/%s-n%d-bs%d-lr%.4f' % (self.prefix, self.dnn, self.nworkers, self.batch_size, self.base_lr)
                    else:
                        relative_path = './weights/%s-n%d-bs%d-lr%.4f' % (self.dnn, self.nworkers, self.batch_size, self.base_lr)
                    if settings.SPARSE:
                        relative_path += '-s%.5f' % self.sparsity
                    utils.create_path(relative_path)
                    filename = '%s-rank%d-epoch%d.pth'%(self.dnn, self.rank, self.train_epoch)
                    fn = os.path.join(relative_path, filename)
                    self.save_checkpoint(state, fn)
                    #self.remove_dict(state)
                self.train_epoch += 1
                # todo zhtang an4 ===========
                if self.train_sampler and (self.nworkers > 1):
                    # print(" In training :  self.train_sampler.set_epoch(self.train_epoch)  ")
                    self.train_sampler.set_epoch(self.train_epoch)

            ss = time.time()
            if data is None:
                data = self.data_iter()

            if self.dataset == 'an4':
                inputs, labels_cpu, input_percentages, target_sizes = data
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            else:
                inputs, labels_cpu = data
            if self.is_cuda:
                if self.dnn == 'lstm' :
                    inputs = Variable(inputs.transpose(0, 1).contiguous()).cuda()
                    labels = Variable(labels_cpu.transpose(0, 1).contiguous()).cuda()
                else:
                    inputs, labels = inputs.cuda(non_blocking=True), labels_cpu.cuda(non_blocking=True)
            else:
                labels = labels_cpu
                
            # wrap them in Variable
            #inputs, labels = Variable(inputs), Variable(labels)
            #logger.info('[%d] labels: %s', self.train_iter, labels_cpu)
            self.iotime += (time.time() - ss)
            
            if self.dnn == 'lstman4':
                out, output_sizes = self.net(inputs, input_sizes)
                out = out.transpose(0, 1)  # TxNxH
                loss = self.criterion(out, labels_cpu, output_sizes, target_sizes)
                loss = loss / inputs.size(0)  # average the loss by minibatch
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 400)
            elif self.dnn == 'lstm' :
                hidden = lstmpy.repackage_hidden(hidden)
                #print(inputs.size(), hidden[0].size(), hidden[1].size())
                outputs, hidden = self.net(inputs, hidden)
                tt = torch.squeeze(labels.view(-1, self.net.batch_size * self.net.num_steps))
                loss = self.criterion(outputs.view(-1, self.net.vocab_size), tt)
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.net.parameters(), 0.25)
                for p in self.net.parameters():
                    p.data.add_(-self.lr, p.grad.data)
            else:
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
            loss_value = loss.item()
            # logger.info statistics
            self.loss += loss_value 

            self.avg_loss_per_epoch += loss_value

            # todo zhtang an4 ==================
            if self.dnn not in ['lstm', 'lstman4']:
                acc1, = self.cal_accuracy(outputs, labels, topk=(1,))
                self.train_acc_top1.append(acc1)
                
            self.train_iter += 1
        self.num_of_updates_during_comm += 1
        self.loss /= num_of_iters 
        self.timer += time.time() - s 
        display = 100
        if self.train_iter % display == 0:
            logger.info('[%3d][%5d/%5d][rank:%d] loss: %.3f, average forward and backward time: %f, iotime: %f ' %
                  (self.train_epoch, self.train_iter, self.num_batches_per_epoch, self.rank,  self.loss, self.timer/display, self.iotime/display))
            mbytes = 1024.*1024
            logger.info('GPU memory usage memory_allocated: %d MBytes, max_memory_allocated: %d MBytes, memory_cached: %d MBytes, max_memory_cached: %d MBytes, CPU memory usage: %d MBytes', 
                    ct.memory_allocated()/mbytes, ct.max_memory_allocated()/mbytes, ct.memory_cached()/mbytes, ct.max_memory_cached()/mbytes, process.memory_info().rss/mbytes)
            self.timer = 0.0
            self.iotime = 0.0
            if len(self.delays) > 0:
                delay = int(np.mean(self.delays))
            else:
                delay = 0
            logger.info('Delay interval: %d, average delay: %d', self.num_of_updates_during_comm- self.average_iter, delay)
            self.delays = []
            if self.is_cuda:
                torch.cuda.empty_cache()
            self.print_weight_gradient_ratio()
            
        # todo zhtang====
        if self.dnn == 'lstm':
            return num_of_iters, hidden
        return num_of_iters

    def train_forward(self, num_of_iters=1, data=None, hidden=None):
        self.loss = 0.0
        self.fw_loss = None

        self.s = time.time()
        self.adjust_learning_rate(self.train_epoch, self.optimizer)

        if self.train_iter % self.num_batches_per_epoch == 0 and self.train_iter > 0:
            logger.info('train iter: %d, num_batches_per_epoch: %d', self.train_iter, self.num_batches_per_epoch)
            #self.adjust_learning_rate(self.train_epoch, self.optimizer)
            logger.info('Epoch %d, avg train acc: %f, lr: %f, avg loss: %f' % (self.train_iter//self.num_batches_per_epoch, np.mean(self.train_acc_top1), self.lr, self.avg_loss_per_epoch/self.num_batches_per_epoch))
            mean_s = np.mean(self.sparsities)
            if self.train_iter>0 and np.isnan(mean_s):
                logger.error('NaN detected! sparsities:  %s' % self.sparsities)
                #sys.exit('NaN detected!!')
            logger.info('Average Sparsity: %f, compression ratio: %f, communication size: %f', np.mean(self.sparsities), np.mean(self.compression_ratios), np.mean(self.communication_sizes))
            self.sparsities = []
            self.compression_ratios = []
            self.communication_sizes = []
            self.train_acc_top1 = []
            #self.test(self.train_epoch)
            self.epochs_info.append(self.avg_loss_per_epoch/self.num_batches_per_epoch)
            self.avg_loss_per_epoch = 0.0
            #self.data_iterator = iter(self.trainloader)
            #if self.train_iter > 0 and self.train_iter % 100 == 0:
            #    self.print_weight_gradient_ratio()

            ## Save checkpoint
            #if self.train_iter > 0 and self.train_epoch % 5 == 0 and self.rank == 0:
            #    state = {'iter': self.train_iter, 'epoch': self.train_epoch, 'state': self.get_model_state()}
            #    if self.prefix:
            #        relative_path = './weights/%s/%s-n%d-bs%d-lr%.4f' % (self.prefix, self.dnn, self.nworkers, self.batch_size, self.base_lr)
            #    else:
            #        relative_path = './weights/%s-n%d-bs%d-lr%.4f' % (self.dnn, self.nworkers, self.batch_size, self.base_lr)
            #    if settings.SPARSE:
            #        relative_path += '-s%.5f' % self.sparsity
            #    utils.create_path(relative_path)
            #    filename = '%s-rank%d-epoch%d.pth'%(self.dnn, self.rank, self.train_epoch)
            #    fn = os.path.join(relative_path, filename)
            #    self.save_checkpoint(state, fn)
            #    #self.remove_dict(state)

            self.train_epoch += 1
            # todo zhtang an4 ===========
            if self.train_sampler and (self.nworkers > 1):
                # print(" In training :  self.train_sampler.set_epoch(self.train_epoch)  ")
                self.train_sampler.set_epoch(self.train_epoch)

        ss = time.time()
        if data is None:
            data = self.data_iter()

        if self.dataset == 'an4':
            inputs, labels_cpu, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        else:
            inputs, labels_cpu = data
        if self.is_cuda:
            if self.dnn == 'lstm' :
                inputs = Variable(inputs.transpose(0, 1).contiguous()).cuda()
                labels = Variable(labels_cpu.transpose(0, 1).contiguous()).cuda()
            else:
                inputs, labels = inputs.cuda(non_blocking=True), labels_cpu.cuda(non_blocking=True)
        else:
            labels = labels_cpu
            
        # wrap them in Variable
        #inputs, labels = Variable(inputs), Variable(labels)
        #logger.info('[%d] labels: %s', self.train_iter, labels_cpu)
        self.iotime += (time.time() - ss)
        
        if self.dnn == 'lstman4':
            out, output_sizes = self.net(inputs, input_sizes)
            out = out.transpose(0, 1)  # TxNxH
            self.fw_loss= self.criterion(out, labels_cpu, output_sizes, target_sizes)
            self.fw_loss = self.fw_loss / inputs.size(0)  # average the loss by minibatch

        elif self.dnn == 'lstm' :
            hidden = lstmpy.repackage_hidden(hidden)
            #print(inputs.size(), hidden[0].size(), hidden[1].size())
            outputs, hidden = self.net(inputs, hidden)
            tt = torch.squeeze(labels.view(-1, self.net.batch_size * self.net.num_steps))
            self.fw_loss = self.criterion(outputs.view(-1, self.net.vocab_size), tt)

        else:
            # forward + backward + optimize
            outputs = self.net(inputs)
            self.fw_loss = self.criterion(outputs, labels)
         
        torch.cuda.synchronize()
        # todo zhtang====
        if self.dnn == 'lstm':
            return num_of_iters, hidden
        return num_of_iters

    def train_backward(self, num_of_iters=1, data=None):

        if self.dnn == 'lstman4':
            self.fw_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 400)

        elif self.dnn == 'lstm' :
            self.fw_loss.backward()
            torch.nn.utils.clip_grad_norm(self.net.parameters(), 0.25)
            for p in self.net.parameters():
                p.data.add_(-self.lr, p.grad.data)
        else:
            # forward + backward + optimize
            self.fw_loss.backward()

        torch.cuda.synchronize()
        loss_value = self.fw_loss.item()
        # logger.info statistics
        self.loss += loss_value 

        self.avg_loss_per_epoch += loss_value

        ## todo zhtang an4 ==================
        #if self.dnn not in ['lstm', 'lstman4']:
        #    acc1, = self.cal_accuracy(outputs, labels, topk=(1,))
        #    self.train_acc_top1.append(acc1)
                
        self.train_iter += 1
        self.num_of_updates_during_comm += 1
        self.loss /= num_of_iters 
        self.timer += time.time() - self.s
        display = 100
        if self.train_iter % display == 0:
            logger.info('[%3d][%5d/%5d][rank:%d] loss: %.3f, average forward and backward time: %f, iotime: %f ' %
                  (self.train_epoch, self.train_iter, self.num_batches_per_epoch, self.rank,  self.loss, self.timer/display, self.iotime/display))
            mbytes = 1024.*1024
            logger.info('GPU memory usage memory_allocated: %d MBytes, max_memory_allocated: %d MBytes, memory_cached: %d MBytes, max_memory_cached: %d MBytes, CPU memory usage: %d MBytes', 
                    ct.memory_allocated()/mbytes, ct.max_memory_allocated()/mbytes, ct.memory_cached()/mbytes, ct.max_memory_cached()/mbytes, process.memory_info().rss/mbytes)
            self.timer = 0.0
            self.iotime = 0.0
            if len(self.delays) > 0:
                delay = int(np.mean(self.delays))
            else:
                delay = 0
            logger.info('Delay interval: %d, average delay: %d', self.num_of_updates_during_comm- self.average_iter, delay)
            self.delays = []
            if self.is_cuda:
                torch.cuda.empty_cache()
            self.print_weight_gradient_ratio()
            
        return num_of_iters


    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            if self.is_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            #inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
        acc = float(correct)/total
        loss = float(test_loss)/total
        logger.info('Epoch %d, lr: %f, val loss: %f, val acc: %f' % (epoch, self.lr, loss, acc))
        self.net.train()

    def update_model(self):
        prev_model = {}
        diff_model = {}
        if settings.EXCHANGE_MODE == 'TOPK_MODEL':
            for name, param in self.net.state_dict().items():
                prev_model[name] = param.clone()
        self.optimizer.step()
        
        if settings.EXCHANGE_MODE == 'TOPK_MODEL':
            with torch.no_grad():
                for name, param in self.net.state_dict().items():
                    #print([name], param.norm(), prev_model[name].norm())
                    diff_model[name] = param - prev_model[name]
                diff_tensor = torch.zeros(self.model_size).cuda()
                offset = 0
                for name, param in diff_model.items():
                    diff_tensor[offset:offset+param.numel()] = param.data.view(param.numel())
                    offset += param.numel()
                self.residuals += diff_tensor
                #print("update residuals with diff_tensor:", self.residuals.norm().cpu())

        if settings.EXCHANGE_MODE == 'MODEL+GRAD':
            for name, parameter in self.net.named_parameters():
                #phok = np.sum([self.m ** i for i in range(1,self.train_iter-self.average_iter+1)])
                if name not in self.v:
                    self.v[name] = parameter.grad.clone()
                else:
                    self.v[name] = self.m * self.v[name] + parameter.grad

    def encode_param(self, param, name=None):
        if not settings.SPARSE:
            param = param.cpu().numpy()
            return param.tobytes()
        elif not name.find('weight') >= 0: # only compress weights
            param = param.cpu().numpy()
            return param.tobytes()
        #    if settings.SPARSE:
        #        return huffman.fullencode(param.tobytes())
        #    else:
        #        return param.tobytes()
        #alpha = 0.95; beta = 0.1
        #nnz = np.count_nonzero(param==0.)
        #if name in self.remainer:
        #    pass
        #    #residuals = self.remainer[name] != 0.
        #    #param[residuals] = (param[residuals] + self.remainer[name][residuals])/2.
        #    #param[residuals] = alpha*param[residuals] + (1-alpha)*self.remainer[name][residuals]
        #    #param[residuals] = param[residuals] + self.remainer[name][residuals]
        #    #self.remainer[name][residuals] = param[residuals] - self.remainer[name][residuals]
        #else:
        #    self.remainer[name] = np.zeros(param.shape, dtype=param.dtype)
        #logger.debug('nnz before: %d, size: %d', nnz, param.size)
        #vals = param.flatten()
        #mean = np.mean(vals)
        #std = np.std(vals)
        #num_epoch = self.train_iter / self.num_batches_per_epoch 
        #if num_epoch >= len(self.target_sparsities):
        #    num_epoch = len(self.target_sparsities) - 1 
        #s = self.target_sparsities[num_epoch]
        #thres = s*2*std 
        #zero_condition = np.abs(param-mean) < thres
        #self.remainer[name][zero_condition] = self.remainer[name][zero_condition] * beta + param[zero_condition]
        #self.remainer[name][zero_condition] += param[zero_condition]
        #param[zero_condition] = 0.
        #sampling = np.random.randint(2, size=param[zero_condition].size)
        try:
            if name not in self.distributions:
                if self.is_cuda:
                    z = torch.zeros(param.shape, dtype=torch.float32, device=torch.cuda.current_device())
                else:
                    z = torch.zeros(param.shape, dtype=torch.float32)
                sparsity = self.sparsity
                if name.find('fc') >= 0:
                    sparsity = 0.999
                p = 1-sparsity
                z += p
                self.distributions[name] = z
            else:
                z = self.distributions[name]
            sampling = torch.bernoulli(z)
            zero_param = param * sampling
            param = zero_param.cpu().numpy()
            #sampling = np.random.binomial(1, 1-0.9, size=param.size)
            #sampling = sampling.reshape(param.shape)
            #param *= sampling
        except Exception as e:
            logger.error('Exception: %s', e)
        #mean_of_zeros = np.mean(param[zero_condition])
        #std_of_zeros = np.std(param[zero_condition])
        #param[zero_condition] = 0. #np.random.normal(mean_of_zeros, std_of_zeros, param[zero_condition].shape)
        #param[zero_condition] = np.sign(param[zero_condition]) * thres/2. 
        nnz = np.count_nonzero(param)
        real_s = (param.size-nnz)*1.0/param.size
        if np.isnan(real_s):
            logger.error('NaN detected! nnz: %d, size: %d' % (nnz, param.size))
        self.sparsities.append(real_s)
        #logger.debug('nnz after: %d, sparsity: %f', nnz, nnz*1.0/param.size)
        dumps = param.tobytes()
        original_size = len(dumps) 
        if settings.SPARSE:
            #dumps = huffman.fullencode(dumps)
            #dumps = huffman.encode_with_indexs(param)
            self.compression_ratios.append(original_size*1.0/len(dumps))
        return dumps

    def decode_param(self, data, name):
        if settings.SPARSE:
            if name.find('weight') >= 0:
                #data = huffman.fulldecode(data)
                if self.is_cuda and settings.GPU_CONSTRUCTION:
                    gpu_mem = self.gpu_caches.get(name, None)
                else:
                    gpu_mem = None
                #data = huffman.decode_with_indexs(data, gpu_mem)
                if self.is_cuda and gpu_mem is None and settings.GPU_CONSTRUCTION:
                    data = torch.from_numpy(data)
                    self.gpu_caches[name] = data.cuda()
                elif not settings.GPU_CONSTRUCTION:
                    data = torch.from_numpy(data)
                return data
        dumps = data
        #arr = np.frombuffer(dumps, dtype=np.float16).astype(np.float32)
        arr = np.frombuffer(dumps, dtype=np.float32)
        arr = torch.from_numpy(arr)
        if self.is_cuda:
            arr = arr.cuda()
        return arr

    def encode_model(self, model):
        if settings.EXCHANGE_MODE == 'TOPK_MODEL':
            # encode the whole model into a tensor
            #gradient_tensor = torch.zeros(self.model_size).cuda()
            model_tensor = torch.zeros(self.model_size).cuda()
            # flatten gradient_tensor
            #offset = 0
            #for name, param in model.items():
            #    if "gradient" in name:
            #        gradient_tensor[offset:offset+param.numel()] = param.view(param.numel())
            #        offset += param.numel()
            # flatten model_tensor
            offset = 0
            for name, param in model.items():
                #if not "gradient" in name:
                model_tensor[offset:offset+param.data.numel()] = param.view(param.data.numel())
                offset += param.data.numel()
            tensor, indexes = self.compress(model_tensor, name="net", sigma_scale=2.5, ratio=0.05)
         
            serialized = []
            tensor_bytes = tensor.cpu().numpy().tobytes()
            indexes_bytes = indexes.cpu().numpy().tobytes()
            serialized.append(struct.pack('i', len(tensor_bytes)))
            #logger.debug('encode name l: %d', len(name))
            serialized.append(tensor_bytes)
            serialized.append(struct.pack('i', len(indexes_bytes)))
            #logger.debug('encode model l: %d', len(byteparam))
            serialized.append(indexes_bytes)
            
            serialized = b''.join(serialized)
            return serialized 

        # origin code, will merge the above with it.
        s = time.time()
        serialized = []
        serialized.append(struct.pack('i', len(model.keys())))
        pciet = 0.
        for name, param in model.items():
            tmpt = time.time()
            #ny = param.cpu().numpy()
            ny = param
            pciet += tmpt-time.time()
            #ny = self.ternarize(param).cpu().numpy()
            #ny = param.cpu().numpy().astype(np.float16)
            byteparam = self.encode_param(ny, name)
            serialized.append(struct.pack('i', len(name)))
            #logger.debug('encode name l: %d', len(name))
            serialized.append(name)
            serialized.append(struct.pack('i', len(byteparam)))
            #logger.debug('encode model l: %d', len(byteparam))
            serialized.append(byteparam)
            #total_size += ny.size()
        #logger.debug('model total size: %d, --get model time used: %f, pcie time: %f', total_size * 4, time.time()-s, tmpt)
        serialized = b''.join(serialized)
        return serialized 

    def decode_model(self, serialized, remote_name = None):
        model_tensor = torch.zeros(self.model_size).cuda()
        if settings.EXCHANGE_MODE == 'TOPK_MODEL':
            # decode topk model
            offset = 0
            sparse_model_size = struct.unpack('i', serialized[offset:offset+4])[0]
            offset += 4
            sparse_model = serialized[offset:offset+sparse_model_size]
            offset += sparse_model_size
            indexes_size = struct.unpack('i', serialized[offset:offset+4])[0]
            offset += 4
            indexes = serialized[offset:offset+indexes_size]

            sparse_model = np.frombuffer(sparse_model, dtype=np.float32)
            sparse_model = torch.from_numpy(sparse_model)
            if self.is_cuda:
                sparse_model = sparse_model.cuda()
            indexes = np.frombuffer(indexes, dtype=np.int64)
            indexes = torch.from_numpy(indexes)
            #print(indexes.size())
            indexes = indexes.to(torch.long)
            if self.is_cuda:
                indexes = indexes.cuda()

            #print("receive model norm:", sparse_model.norm().cpu())
            #if not remote_name in self.remote_indexes:
            #    self.remote_indexes[remote_name] = torch.zeros(self.model_size).cuda()
            #    self.remote_model[remote_name] = torch.zeros(self.model_size).cuda()
            #self.remote_indexes[remote_name][indexes] = 1.0
            #self.remote_model[remote_name][indexes] = sparse_model
            model_tensor[indexes] = sparse_model
            self.recv_indexes = indexes
            
            #offset = 0
            #self.recv_indexes = indexes
            #for name, param in self.net.named_parameters():
            #    own_state[name] = model_tensor[offset:offset+param.data.numel()].view(param.size())
            #    offset += param.data.numel()
                
            return model_tensor
            #return self.remote_model[remote_name]

        # origin code, will merge the above with it.
        own_state = {}
        offset = 0
        num_item = struct.unpack('i', serialized[offset:offset+4])[0]
        offset += 4
        for i in range(num_item):
            l = struct.unpack('i', serialized[offset:offset+4])[0]
            #logger.debug('decode name l: %d', l)
            offset += 4
            name = serialized[offset:offset+l]
            offset += l
            l = struct.unpack('i', serialized[offset:offset+4])[0]
            #logger.debug('decode model l: %d', l)
            offset += 4
            param = serialized[offset:offset+l]
            offset += l
            own_state[name] = param
        return own_state

    def _get_original_params(self, mode=settings.EXCHANGE_MODE):
        if mode == 'MODEL' or mode == 'TOPK_MODEL':
            own_state = self.net.state_dict()
            return own_state
        #elif mode == 'TOPK_MODEL':
        #    model_of_params = {}
        #    for name, parameter in self.net.named_parameters():
        #        model_of_params[name] = parameter.data
        #    return model_of_params
        elif mode == 'GRAD':
            grad_of_params = {}
            for name, parameter in self.net.named_parameters():
                grad_of_params[name] = parameter.grad
            return grad_of_params
        elif mode == 'MODEL+GRAD':
            model_and_grad = {}
            for name, parameter in self.net.named_parameters():
                model_and_grad[name] = parameter.data
                model_and_grad[name+b'_gradient'] = parameter.grad
            return model_and_grad 

    def get_model(self, mode=settings.EXCHANGE_MODE):
        grad_of_params = self._get_original_params(mode)
        encoded_model = self.encode_model(grad_of_params)
        self.communication_sizes.append(len(encoded_model))
        return encoded_model

    def ternarize(self, params):
        """
        Paper: TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning, W. Wen et al., 2017
        """
        c = 2.5
        std = torch.std(params)
        params = torch.clamp(params, min=-c*torch.abs(std), max=c*torch.abs(std))
        st = torch.max(torch.abs(params))
        propabilities = torch.abs(params) / st
        distribution = torch.distributions.Bernoulli(propabilities)
        b = distribution.sample()
        tern = st * torch.sign(params)  * b 
        #tern = torch.sign(params)  * b
        logger.debug('Tern norm: %f', torch.norm(tern, 2))
        return tern

    def param_average(self, a, b, ratio, is_asked, v=None, name=None):
        """
        b should be a Tensor
        """
        if self.is_cuda:
            a_tensor = a
            if b.is_cuda:
                b_tensor = b
            else:
                b_tensor = b.cuda()
        else:
            a_tensor = a.cpu() 
            b_tensor = b
        #if is_asked:
        #new_param = (a_tensor+b_tensor.view(a_tensor.size()))/2.0
        b_tensor = b_tensor.view(a_tensor.size())
        if settings.SPARSE:
            condition = b_tensor == 0.
            if name.find('weight') >= 0:
                new_param = (a_tensor + torch.where(condition, a_tensor, b_tensor))/2.
            else:
                new_param = (a_tensor+b_tensor)/2. 
            #new_param[condition] = new_param[condition] - self.lr * self.net.named_parameters()[name].grad[condition]
            #for n, parameter in self.net.named_parameters():
            #    if name == n:
            #        new_param[condition] = new_param[condition] - 2*self.lr * parameter.grad[condition]
            #        break
            #new_param = (1+self.lr) * a_tensor + self.lr * b_tensor
            #new_param = (a_tensor + b_tensor)/2.
            # clamp
            #std = torch.std(new_param)
            #new_param[torch.abs(new_param)>=3*std] = 0.
            #new_param = (a_tensor + b_tensor)/2.
            #std = torch.std(new_param)
            #new_param[torch.abs(new_param)>=8*std] = torch.mean(new_param)
        else:
            if settings.EXCHANGE_MODE == 'MODEL+GRAD':
                v_tensor = torch.from_numpy(v).cuda()
                v_tensor = v_tensor.view(a_tensor.size())
                #new_param = ((b_tensor - self.lr * v_tensor) + a_tensor ) / 2. # Works
                #new_param = (b_tensor - self.lr * (v_tensor + 0.5*v_tensor * v_tensor * (a_tensor - b_tensor)) + a_tensor ) / 2. # Works
                #new_param = (b_tensor - self.lr * (v_tensor + 0.9*v_tensor * v_tensor * (a_tensor - b_tensor)) + a_tensor ) / 2.
                #interval = self.train_iter - self.average_iter
                #sump = 0.0
                #for j in range(1, interval+1):
                #    phok = float(np.sum([self.m ** i for i in range(1,j+1)]))
                #    sump+=phok
                #phok = float(np.sum([self.m ** i for i in range(1,interval+1)]))
                #square = v_tensor * v_tensor
                #g_dc = (v_tensor - square * b_tensor) * sump + square * self.v[name] + v_tensor * phok
                if name in self.v:
                    g_dc = self.v[name] * self.m + v_tensor
                else:
                    g_dc = v_tensor
                b_tensor_dc = b_tensor - self.lr * g_dc
                new_param = (a_tensor + b_tensor_dc) / 2.
                if name in self.v:
                    self.v[name][self.v[name]!=0.] = 0.
            else:
                new_param = (a_tensor + b_tensor)/2.
        del b_tensor
        del a_tensor
        return new_param

    def replace_model(self, model):
        recv_state = self.decode_model(model)
        own_state = self.net.state_dict()
        for name, param in own_state.items():
            if name in recv_state:
                a_tensor = own_state[name]
                b_tensor = self.decode_param(recv_state[name], name)
                b_tensor = b_tensor.view(a_tensor.size())
                own_state[name] = b_tensor 
        self.net.load_state_dict(own_state)

    def average_model(self, model, recved_loss, is_asked, remote_name):
        #logger.info("receive model from %s, try to average...", remote_name)
        #own_state = self.net.state_dict()
        #time.sleep(8)
        if settings.EXCHANGE_MODE == 'MODEL+GRAD':
            own_state = {}
            for name, parameter in self.net.named_parameters():
                own_state[name] = parameter.data
        else:
            own_state = self._get_original_params()
        s = time.time()
        loss = self.get_loss()
        average_ratio = 1.0
        if recved_loss > 0:
            r = (recved_loss - loss) / recved_loss
            if r > 0.2:
                #average_ratio = 1.0001#1+self.lr 
                average_ratio = 1.001#1+self.lr 
            elif r < -0.2:
                #average_ratio = 0.9999#1-self.lr 
                average_ratio = 0.999#1-self.lr 
            else:
                average_ratio = 1
        recv_state = self.decode_model(model, remote_name)

        if settings.EXCHANGE_MODE == 'TOPK_MODEL':
            model_tensor = torch.zeros(self.model_size).cuda()
            offset = 0
            for name, param in self.net.state_dict().items():
                model_tensor[offset:offset + param.numel()] = param.data.view(param.numel())
                offset += param.numel()

            #print("before average model:", model_tensor.norm().cpu(), recv_state.norm().cpu())
            #logger.info("check some indexes: %s", self.recv_indexes[:20].cpu())
            model_tensor[self.recv_indexes] = (recv_state[self.recv_indexes] + model_tensor[self.recv_indexes]) / 2.0
            #cur_indexes = (self.remote_indexes[remote_name] == 1).nonzero()
            #cur_indexes = cur_indexes.view(cur_indexes.numel())
            #model_tensor[cur_indexes] = (self.remote_model[remote_name][cur_indexes] + model_tensor[cur_indexes]) / 2.0
            #model_tensor = (recv_state + model_tensor) / 2.0
            #print("after average model:", model_tensor.norm().cpu())
            offset = 0
            own_state = {}
            for name, param in self.net.state_dict().items():
                #print(name)
                own_state[name] = model_tensor[offset:offset + param.numel()].view(param.size())
                offset += param.numel()
            
            self.net.load_state_dict(own_state)
            #for name, param in self.net.named_parameters():
            #    param.data = own_state[name]
            return
            
            
        #model_tensor = torch.zeros(self.model_size).cuda()
        #remote_tensor = torch.zeros(self.model_size).cuda()
        #local_tensor = torch.zeros(self.model_size).cuda()
        #offset = 0
        for name, param in own_state.items():
            #if name not in recv_state:
            #    continue
            remote_param = self.decode_param(recv_state[name], name)  
            v = None
            if settings.EXCHANGE_MODE == 'MODEL+GRAD':
                remote_gradients = self.decode_param(recv_state[name+b'_gradient'])
                #if settings.DELAY > 0:
                #    remote_param = remote_param-self.lr*remote_gradients * (self.train_iter-self.average_iter-1)
                #else:
                #remote_param = remote_param+0.25*self.lr*remote_gradients * (self.train_iter-self.average_iter)
                #remote_param = remote_param-self.lr*remote_gradients * (self.train_iter-self.average_iter-1)
                #correction = 0.0
                #for j in range(self.train_iter - self.average_iter+1):
                #    for k in range(j+1):
                #        correction += (1-0.9)**k

                #if name in self.v:
                #    v = 0.9 * self.v[name] + remote_gradients
                #    self.v[name] = v
                #else:
                #    v = remote_gradients
                #    self.v[name] = v
                v = remote_gradients

                #remote_param = remote_param - self.lr * v 
                #remote_param = remote_param - self.lr * v * v * (param - remote_param)
                #remote_param = remote_param-self.lr*remote_gradients*correction # momentum correction
            new_param = self.param_average(param, remote_param, average_ratio, is_asked, v, name)
            #model_tensor[offset:offset+new_param.numel()] = new_param.view(new_param.numel())
            #remote_tensor[offset:offset+remote_param.numel()] = remote_param.view(remote_param.numel())
            #local_tensor[offset:offset+param.numel()] = param.view(param.numel())
            #offset += new_param.numel()
            del param
            del remote_param
            own_state[name] = new_param
        if settings.EXCHANGE_MODE == 'MODEL':
            #print("before average norm:", local_tensor.norm().cpu(), remote_tensor.norm().cpu())
            #print("model average norm:", model_tensor.norm().cpu())
            self.net.load_state_dict(own_state)
        elif settings.EXCHANGE_MODE == 'GRAD':
            for name, parameter in self.net.named_parameters():
                parameter.grad.data = own_state[name].data
        elif settings.EXCHANGE_MODE == 'MODEL+GRAD':
            for name, parameter in self.net.named_parameters():
                parameter.data = own_state[name]
            #self.net.load_state_dict(own_state)
        logger.debug('====model average time: %f', time.time()-s)
        self.delays.append(self.num_of_updates_during_comm - self.average_iter)
        self.average_iter = self.num_of_updates_during_comm
        self.remove_dict(own_state)

    def update_with_remote_gradients(self, recved_gradients):
        recv_state = self.decode_model(recved_gradients)
        for name, parameter in self.net.named_parameters():
            recved = recv_state.get(name, None)
            if recved:
                b_tensor = self.decode_param(recved, name)
                b_tensor = b_tensor.view(parameter.size())
                parameter.grad = b_tensor 
        self.update_model()

    def remove_dict(self, dictionary):
        # keys = dictionary.keys()
        # for k in keys:
        #     del dictionary[k]
        #　todo zhtang an4 ========
        dictionary.clear()

    def send_model(self, rank):
        own_state = self.net.state_dict()
        for name, param in own_state.items():
            dist.send(tensor=param, dst=rank)
        #logger.info("finished %s layer..." % name)

    def recv_model(self, rank):
        own_state = self.net.state_dict()
        for name, param in own_state.items():
            dist.recv(tensor=param, src=rank)
            own_state[name] = (own_state[name] + param) / 2.0
        #logger.info("finished %s layer..." % name)
        self.net.load_state_dict(own_state)

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)

    def _step(self, closure=None):
        """Performs a single optimization step.
            Arguments:
                closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
    
        for group in self.optimizer.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
    
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)
        return loss

    def zero_grad(self):
        self.optimizer.zero_grad()


def train_with_single(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, num_steps=1):
    torch.cuda.set_device(0)
    trainer = DLTrainer(0, nworkers, dist=False, batch_size=batch_size, 
        is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, 
        dnn=dnn, lr=lr, nworkers=nworkers, prefix='allreduce', num_steps = num_steps)
    iters_per_epoch = trainer.get_num_of_training_samples() // (nworkers * batch_size * nsteps_update)

    times = []
    display = 100 if iters_per_epoch > 100 else iters_per_epoch-1
    for epoch in range(max_epochs):
        # todo zhtang ==========
        if dnn == 'lstm':
            hidden = trainer.net.init_hidden()
        for i in range(iters_per_epoch):
            s = time.time()
            trainer.optimizer.zero_grad()
            for j in range(nsteps_update):
                # todo zhtang ========
                if dnn == 'lstm':
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)
            trainer.update_model()
            times.append(time.time()-s)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f. Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []
# todo

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single trainer")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--nsteps-update', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'cifar10', 'an4', 'ptb'], help='Specify the dataset for training')
    parser.add_argument('--dnn', type=str, default='resnet50', choices=['resnet50', 'resnet20', 'vgg19', 'vgg16', 'alexnet', 'lstman4', 'lstm'], help='Specify the neural network for training')
    parser.add_argument('--data-dir', type=str, default='./data', help='Specify the data root path')
    parser.add_argument('--lr', type=float, default=0.1, help='Default learning rate')
    parser.add_argument('--max-epochs', type=int, default=settings.MAX_EPOCHS, help='Default maximum epochs to train')
    parser.add_argument('--num-steps', type=int, default=35)
    args = parser.parse_args()
    batch_size = args.batch_size * args.nsteps_update
    prefix = settings.PREFIX
    relative_path = './logs/singlegpu-%s/%s-n%d-bs%d-lr%.4f-ns%d' % (prefix, args.dnn, 1, batch_size, args.lr, args.nsteps_update)
    utils.create_path(relative_path)
    logfile = os.path.join(relative_path, settings.hostname+'.log')
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', args)
    # todo zhtang===========================
    train_with_single(args.dnn, args.dataset, args.data_dir, 1, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.num_steps)
