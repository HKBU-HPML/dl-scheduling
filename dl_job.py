import pandas as pd
import numpy as np
import time

class DLJob:

    def __init__(self, job_idx, sync_tt, ngpus=1, batch_size=32, data_dir='./data', dataset='cifar10', dnn='resnet20', lr=0.04, nworkers=1):

    # network training setting
    self.batch_size = batch_size
    self.data_dir = data_dir
    self.dataset = dataset
    self.dnn = dnn
    self.lr = lr
    self.nworkers = nworkers

    # job setting
    self.job_idx = job_idx
    self.job_config = "jobs/%s.conf" % job_idx
    self.sync_tt = sync_tt
    self.f_timetable = []
    self.b_timetable = []
    self.c_timetable = []

    def load_job_config(self):
        self.job_schedule = pd.read_csv(self.job_config, header = 0)
        self.iters = self.job_schedule.shape[0]
        self.f_timetable = self.job_schedule.forward.tolist()
        self.b_timetable = self.job_schedule.backward.tolist()
        self.c_timetable = self.job_schedule.communicate.tolist()

    def run(self):
        for i in range(self.iters):
            time.sleep(self.f_timetable[i])
            # do forward
            time.sleep(self.b_timetable[i])
            # do backward
            time.sleep(self.c_timetable[i])
            # do communicate




    
    
