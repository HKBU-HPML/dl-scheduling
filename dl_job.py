import numpy as np
import time
import json, yaml
import logging
from settings import logger
import os

class DLJob:

    def __init__(self, job_root='job_configs', job_set='job_set_1', job_id=0):

        self.job_id = job_id
        self.job_config=os.path.join(job_root, job_set, "job_%d.json"%job_id)
        self.load_job_config(self.job_config)

    def load_job_config(self, job_config):

        with open(job_config, 'r') as f:
            #self.job_json = json.load(f)
            self.job_json = yaml.safe_load(f)

        # network training setting
        self.dnn = self.job_json["dnn"]
        self.lr = self.job_json["lr"]
        self.batch_size = self.job_json["batch_size"]
        self.dataset = self.job_json["dataset"]
        self.data_dir = self.job_json["data_dir"]
        self.nworkers = self.job_json["nworkers"]
        self.nsteps_update = self.job_json["nsteps_update"]

        # job setting
        self.sync_tt = self.job_json["sync_tt"]
        self.iters = self.job_json['iters']
        self.device_ids = self.job_json['gpus']
        self.ngpus = len(self.device_ids)
        self.hostfile = self.job_json['hostfile']
        self.schedule = self.job_json['schedule']

    def get_forward_schedule(self, rank, iters):
        return self.schedule['rank_'%rank]['forward'][iters]

    def get_backward_schedule(self, rank, iters):
        return self.schedule['rank_'%rank]['backward'][iters]

    def get_communication_schedule(self, rank, iters):
        return self.schedule['rank_'%rank]['comm'][iters]

    def get_device(self, rank):
        return self.device_ids[rank]

    
    
