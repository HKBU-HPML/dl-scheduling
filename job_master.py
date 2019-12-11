import os, glob
from random import sample, randint
import json, yaml
from cluster import *
from job import dag_job
import numpy as np
import random

#SUPPORT_NETS = ["resnet20", "lstm", "lstman4", "resnet50"]
SUPPORT_NETS = ["resnet50", "googlenet", "alexnet"]
TEMPLATES = {
             #"resnet20": {"model_size":100, "lr":0.1, "dataset":"cifar10", "data_dir":"data", "fw_time":100, "bw_time":100, "batch_size":32},
             #"lstm": {"model_size":100, "lr":0.1, "dataset":"ptb", "data_dir":"data", "fw_time":100, "bw_time":100, "batch_size":20},
             #"lstman4": {"model_size":100, "lr":0.1, "dataset":"an4", "data_dir":"data", "fw_time":100, "bw_time":100, "batch_size":20},
             "resnet50": {"model_size":97.7, "lr":0.1, "dataset":"imagenet", "data_dir":"data", "fw_time":38, "bw_time":67, "batch_size":16},
             "googlenet": {"model_size":26.7, "lr":0.1, "dataset":"imagenet", "data_dir":"data", "fw_time":60, "bw_time":92, "batch_size":64},
             "alexnet": {"model_size":235, "lr":0.1, "dataset":"imagenet", "data_dir":"data", "fw_time":45, "bw_time":85, "batch_size":256},
             #"resnet50": {"model_size":97.7, "lr":0.1, "dataset":"imagenet", "data_dir":"data", "fw_time":19, "bw_time":33, "batch_size":4},
             #"googlenet": {"model_size":26.7, "lr":0.1, "dataset":"imagenet", "data_dir":"data", "fw_time":30, "bw_time":46, "batch_size":16},
             #"alexnet": {"model_size":235, "lr":0.1, "dataset":"imagenet", "data_dir":"data", "fw_time":22, "bw_time":42, "batch_size":64},
}

CLUSTER = {"num_node": 16, "num_gpu":4, "gpu_mem":8192, "cpu_mem":16384, "network_speed": 128} # unit is MB. 

class job_generator:

    def __init__(self, set_name, num_jobs):
        self.set_name = set_name
        self.num_jobs = num_jobs

    def random_generate(self):
        self.job_root = "job_configs/%s" % self.set_name
        if not os.path.exists(self.job_root):
            os.makedirs(self.job_root)

        for i in range(self.num_jobs):
            job_json = {}
            job_json["job_id"] = i
            job_json["job_name"] = "j%d" % i
            job_json["dnn"] = sample(SUPPORT_NETS, 1)[0]
            #job_json["dnn"] = "resnet20"
            job_json.update(TEMPLATES[job_json["dnn"]])

            job_json["nworkers"] = 2 ** randint(0, 4)
            job_json["nsteps_update"] = 1
            job_json["cuda_enabled"] = 1
            job_json["iters"] = randint(5, 50)

            with open(os.path.join(self.job_root, "job_%d.json"%i), "w") as f:
                yaml.safe_dump(job_json, f)
            
    def microsoft_generate(self):

        step = self.num_jobs / 80
        job_dist = [40, 47, 60, 75, 79, 80]
        #job_dist = [20, 27, 40, 65, 75, 80] 
        #job_dist = [50, 57, 65, 75, 79, 80]
        job_dist = [i * step for i in job_dist]

        self.job_root = "job_configs/%s" % self.set_name
        if not os.path.exists(self.job_root):
            os.makedirs(self.job_root)

        min_iter = 50
        max_iter = 300
        # 1-GPU jobs
        for i in range(job_dist[0]):
            job_json = {}
            job_json["job_id"] = i
            job_json["job_name"] = "j%d" % i
            job_json["dnn"] = sample(SUPPORT_NETS, 1)[0]
            #job_json["dnn"] = "resnet20"
            job_json.update(TEMPLATES[job_json["dnn"]])

            job_json["nworkers"] = 1
            job_json["nsteps_update"] = 1
            job_json["cuda_enabled"] = 1
            job_json["iters"] = randint(min_iter, max_iter) * 2

            with open(os.path.join(self.job_root, "job_%d.json"%i), "w") as f:
                yaml.safe_dump(job_json, f)

        # 2-GPU jobs
        for i in range(job_dist[0], job_dist[1]):
            job_json = {}
            job_json["job_id"] = i
            job_json["job_name"] = "j%d" % i
            job_json["dnn"] = sample(SUPPORT_NETS, 1)[0]
            #job_json["dnn"] = "resnet20"
            job_json.update(TEMPLATES[job_json["dnn"]])

            job_json["nworkers"] = 2
            job_json["nsteps_update"] = 1
            job_json["cuda_enabled"] = 1
            job_json["iters"] = randint(min_iter, max_iter) * 2

            with open(os.path.join(self.job_root, "job_%d.json"%i), "w") as f:
                yaml.safe_dump(job_json, f)

        # 4-GPU jobs
        for i in range(job_dist[1], job_dist[2]):
            job_json = {}
            job_json["job_id"] = i
            job_json["job_name"] = "j%d" % i
            job_json["dnn"] = sample(SUPPORT_NETS, 1)[0]
            #job_json["dnn"] = "resnet20"
            job_json.update(TEMPLATES[job_json["dnn"]])

            job_json["nworkers"] = 4
            job_json["nsteps_update"] = 1
            job_json["cuda_enabled"] = 1
            job_json["iters"] = randint(min_iter, max_iter) * 2 

            with open(os.path.join(self.job_root, "job_%d.json"%i), "w") as f:
                yaml.safe_dump(job_json, f)

        # 8-GPU jobs
        for i in range(job_dist[2], job_dist[3]):
            job_json = {}
            job_json["job_id"] = i
            job_json["job_name"] = "j%d" % i
            job_json["dnn"] = sample(SUPPORT_NETS, 1)[0]
            #job_json["dnn"] = "resnet20"
            job_json.update(TEMPLATES[job_json["dnn"]])

            job_json["nworkers"] = 8
            job_json["nsteps_update"] = 1
            job_json["cuda_enabled"] = 1
            job_json["iters"] = randint(min_iter, max_iter)

            with open(os.path.join(self.job_root, "job_%d.json"%i), "w") as f:
                yaml.safe_dump(job_json, f)

        # 16-GPU jobs
        for i in range(job_dist[3], job_dist[4]):
            job_json = {}
            job_json["job_id"] = i
            job_json["job_name"] = "j%d" % i
            job_json["dnn"] = sample(SUPPORT_NETS, 1)[0]
            #job_json["dnn"] = "resnet20"
            job_json.update(TEMPLATES[job_json["dnn"]])

            job_json["nworkers"] = 16
            job_json["nsteps_update"] = 1
            job_json["cuda_enabled"] = 1
            job_json["iters"] = randint(min_iter, max_iter)

            with open(os.path.join(self.job_root, "job_%d.json"%i), "w") as f:
                yaml.safe_dump(job_json, f)

        # 32-GPU jobs
        for i in range(job_dist[4], job_dist[5]):
            job_json = {}
            job_json["job_id"] = i
            job_json["job_name"] = "j%d" % i
            job_json["dnn"] = sample(SUPPORT_NETS, 1)[0]
            #job_json["dnn"] = "resnet20"
            job_json.update(TEMPLATES[job_json["dnn"]])

            job_json["nworkers"] = 32
            job_json["nsteps_update"] = 1
            job_json["cuda_enabled"] = 1
            job_json["iters"] = randint(min_iter, max_iter)

            with open(os.path.join(self.job_root, "job_%d.json"%i), "w") as f:
                yaml.safe_dump(job_json, f)

            
class job_scheduler:

    def __init__(self, set_name):
        self.job_root = "job_configs/%s" % set_name
        self.job_files = glob.glob(r'job_configs/%s/job*.json' % set_name)
        self.num_jobs = len(self.job_files)

        self.job_set = []
        self.json_set = []
        self.load_job_set()

        self.clust = cluster(CLUSTER)

        # statistical variable
        self.total_time = 0

    def print_jobs(self):

        def job_format(job):
            return "job %d:\t%s-%s, %d-gpu, %d-iter." % (job.job_id, job.dnn, job.dataset, job.nworkers, job.iters)

        for job in self.job_set:
            print job_format(job)
        print ""

    def load_job_set(self):
        for jf in self.job_files:
            with open(jf, 'r') as f:
                job_json = yaml.safe_load(f)
                self.job_set.append(dag_job(job_json))
                self.json_set.append(job_json)

    def allocate(self, algo="blf", pick_thres=32): # "ssf", "slf", "bsf", "blf", "random" 
 
        print "Placement Algorithm:", algo

        self.job_set.sort(key=lambda x:x.nworkers, reverse=True)
        #self.print_jobs()

        def get_node_compute_workload(node):
            return node.get_gpus_workload()

        #print ["job-%d-w%d-%d" % (j.job_id, j.nworkers, j.compute_duration) for j in self.job_set]
        # sort the job by some rules
        #self.job_set.sort(key=lambda x:x.gpu_compute_cumul_duration, reverse=True) # by GPU Time
        #print ["job-%d-%d" % (j.job_id, j.gpu_compute_cumul_duration) for j in self.job_set]

        #self.job_set.sort(key=lambda x:x.compute_duration) # by job duration
        #print ["job-%d-%d" % (j.job_id, j.compute_duration) for j in self.job_set]

        #self.job_set.sort(key=lambda x:x.nworkers, reverse=True) # by the GPU number
        #print ["job-%d-%d" % (j.job_id, j.nworkers) for j in self.job_set]

        if algo == "ssf":
            self.job_set = sorted(self.job_set, key=lambda x: (x.nworkers, x.compute_duration))
        elif algo == "slf":
            self.job_set = sorted(self.job_set, key=lambda x: (x.nworkers, -x.compute_duration))
        elif algo == "bsf":
            self.job_set = sorted(self.job_set, key=lambda x: (-x.nworkers, x.compute_duration))
        elif algo == "blf":
            self.job_set = sorted(self.job_set, key=lambda x: (-x.nworkers, -x.compute_duration))
        elif algo == "random":
            random.shuffle(self.job_set)
            
        #print ["job-%d-w%d-%d" % (j.job_id, j.nworkers, j.compute_duration) for j in self.job_set]

        def pick_gpus(job):

            if job.nworkers <= pick_thres:
                candidate_gpus = self.clust.gpu_list
                # select the gpu with the minimum makespan
                self.clust.gpu_list = sorted(self.clust.gpu_list, key=lambda x: (x.makespan, x.node_id))
                #print ["n%d-g%d-%d" % (g.node_id, g.gpu_id, g.makespan) for g in self.clust.gpu_list]
                return self.clust.gpu_list[:job.nworkers]

            # select nodes with least makespan
            for host_node in self.clust.node_list:
                host_node.update_makespan()
            self.clust.node_list = sorted(self.clust.node_list, key=lambda x: (x.makespan))

            ngpus = job.nworkers
            if ngpus <= 4:
                gpu_first_node = self.clust.node_list[0].gpu_list
                gpu_first_node = sorted(gpu_first_node, key=lambda x:(x.makespan))
                return gpu_first_node[:ngpus]
            else:
                gpu_list = []
                for i in range(ngpus / 4):
                    gpu_list.extend(self.clust.node_list[i].gpu_list)
                return gpu_list

        for job in self.job_set:

            finished = False
            allocated_worker = 0

            # pick gpus
            selected_gpus = pick_gpus(job)
            #print ["job-%d-%d to n-%d g-%d" % (job.job_id, job.compute_duration, g.node_id, g.gpu_id) for g in selected_gpus]

            # add the job to nodes and update the comm overhead
            nodes = [g.host_node for g in selected_gpus]
            selected_node_ids = []
            for n in nodes:
                if not n.node_id in selected_node_ids:
                    n.add_job(job)
                    selected_node_ids.append(n.node_id)
             
            comm_overhead = 0
            if len(selected_node_ids) > 1:
                comm_overhead = job.model_size * 7.8125 * job.iters

            # add the job to gpus and update the makespan
            max_makespan = max([g.makespan for g in selected_gpus])   
            for i, g in enumerate(selected_gpus):
                g.add_job(job, i, max_makespan + comm_overhead)

        self.clust.gpu_list = sorted(self.clust.gpu_list, key=lambda x: (x.makespan, x.node_id))
        print ["n%d-g%d-%d" % (g.node_id, g.gpu_id, g.makespan) for g in self.clust.gpu_list]
                
        #for node in self.clust.node_list:
        #    node.print_jobs()

    def check_finished(self):
        finished_ids = []
        for job in self.job_set:
            if job.is_finished:
                finished_ids.append(job.job_id)

        return finished_ids

    def schedule(self, comm_thres=1, adaDual=True):

        for job in self.job_set:
            job.initial_task()

        cur_time = 0
        
        job_id_pool = [job.job_id for job in self.job_set]

        time = 0
        num_finished_jobs = 0
        while len(job_id_pool) != 0:

            print_log = ""

            # update compute/comm task list, check whether some have ended
            for node in self.clust.node_list:
                node.update_status(time)

            finished_job_ids = self.check_finished()
            if len(finished_job_ids) > num_finished_jobs:
                num_finished_jobs = len(finished_job_ids)
                print_log += "finished: %s\n" % finished_job_ids
                job_id_pool = [job_id for job_id in job_id_pool if job_id not in finished_job_ids]

            comm_jobs = []
            for job in self.job_set:
                if not job.is_finished and job.ready_for_comm():
                    if len(job.nodes) == 1:
                        job.sync_comm(time)
                    else:
                        comm_jobs.append(job)

            #comm_jobs = sorted(comm_jobs, key=lambda x: (-x.model_size))
            comm_jobs = sorted(comm_jobs, key=lambda x: (-x.nworkers, -x.model_size))
            for job in comm_jobs:
                if job.is_all_nodes_free(time, thres=comm_thres, adaDual):
                    comm_task = job.get_comm()
                    for node in job.nodes:
                        node.add_run(comm_task, time)
                    job.lock_comm()
                    print_log += "job-%d running comm iter-%d %s.\n" % (job.job_id, job.available_tasks[0]["iter"], job.get_nodes())

            # update comm speed
            for node in self.clust.node_list:
                node.update_comm()

            # print the speed of those comm jobs
            for job in self.job_set:
                if job.is_comm_stall:
                    slowest_node = job.get_slowest_node()
                    print_log += job.comm_task.update_comm_time(time, slowest_node)
            
            for node in self.clust.node_list:
                # schedule available compute tasks
                for gpu in node.gpu_list:

                    # select a job
                    #candidate_jobs = gpu.get_ready_jobs()
                    #chosen_job = max(candidate_jobs, key=lambda x:x.nworkers)
                    if gpu.is_busy:
                        continue

                    candidate_compute = gpu.get_avail_compute()
                    #print candidate_jobs
                    if len(candidate_compute) == 0:
                        continue

                    #chosen_compute = max(candidate_compute, key=lambda x:x.nworkers)
                    chosen_compute = sorted(candidate_compute, key=lambda x:(-x.nworkers, -x.compute_duration))[0]

                    # run the chosen_job
                    job_id, cur_task = gpu.add_run(chosen_compute, time)
                    #if cur_task["type"] == "comm":
                    #    node.add_comm(chosen_compute)

                    print_log += "node %d-gpu %d: running job-%d %s\n" % (node.node_id, gpu.gpu_id, job_id, cur_task)

                ## choose communication task to execute
                #if not node.is_comm():
                #    candidate_comm = node.get_avail_comm()
                #    # only schedule one comm task
                #    if len(candidate_comm) != 0:
                #        comm = candidate_comm[0]
                #        job_id, cur_task = node.add_run(comm, time)
                #        print_log += "job-%d running comm %s.\n" % (job_id, comm.get_nodes())

            if print_log != "":
                print "Time: %d\n%s" % (time, print_log)

            #if time > 400:
            #    break
            time += 1

        self.total_time = time

        # print the comm algo
        if adaDual:
            print "Scheduling Algorithm: adaDual enabled, thres=%d." % comm_thres
        else:
            print "Scheduling Algorithm: adaDual disabled, thres=%d." % comm_thres
                
    def print_stat(self):

        for node in self.clust.node_list:
            print "node-%d: %d / %d." % (node.node_id, node.active_time, self.total_time)
            for gpu in node.gpu_list:
                print "\t gpu-%d: %d / %d." % (gpu.gpu_id, gpu.active_time, self.total_time)

        aver_job_time = np.mean([j.finish_time for j in self.job_set])
        print "Average Job Completion Time is %f ms." % aver_job_time

    def write_allocate(self):

        def gpu_allocate(nworkers):
            #nodes = {
            #         "gpu10":[i % 4 for i in range(nworkers/2)], 
            #         "gpu11":[i % 4 for i in range(nworkers/2)], 
            #}
            nodes = {
                     "localhost":[-1 for i in range(nworkers)], 
            }
            return nodes
            
        for idx, job in enumerate(self.job_set):

            job_json = job.job_conf
            schedule = job_json.copy()

            # allocate nodes and GPUs
            node_gpu = gpu_allocate(job_json['nworkers'])
            hostfile = os.path.join(self.job_root, "cluster_j%d" % job_json['job_id'])
            schedule["hostfile"] = hostfile
            schedule["gpus"] = []
            with open(hostfile, "w") as f:
                for node in node_gpu:
                    f.write("%s slots=%d\n" % (node, len(node_gpu[node])))
                    schedule["gpus"].extend(node_gpu[node])
            
            # schedule the tasks
            schedule["schedule"] = {}
            for r in range(job_json['nworkers']):
                tmp_plan = {}
                f = []
                b = []
                c = []
                for i in range(job_json['iters']):
                    f.append(0)
                    b.append(0)
                    c.append(0)
                tmp_plan["forward"] = f
                tmp_plan["backward"] = b
                tmp_plan["comm"] = c
                schedule["schedule"]["rank_%d"%r] = tmp_plan
       
            with open(os.path.join(self.job_root, "schedule_%d.json"%idx), "w") as f:
                yaml.safe_dump(schedule, f)

    def write_schedule(self):
        pass

num_jobs = 4
jobG = job_generator("test_%djobs" % num_jobs, num_jobs)
jobS = job_scheduler("test_%djobs" % num_jobs)
jobS.write_allocate()
#jobG = job_generator("test_%djobs" % num_jobs, num_jobs)
#jobG.random_generate()
#jobS = job_scheduler("test_%djobs" % num_jobs)

#jobG = job_generator("microsoft-160", 160)
#jobG.microsoft_generate()

#jobS = job_scheduler("microsoft-80")
#jobS.allocate(big_first=False, pick_thres=32)
#jobS = job_scheduler("microsoft-80")
#jobS.allocate(big_first=True)
##jobS.print_jobs()
#jobS.schedule()
#jobS.print_stat()
# compare different placement
#jobS = job_scheduler("microsoft-160")
#jobS.allocate(algo="random")
#jobS = job_scheduler("microsoft-160")
#jobS.allocate(algo="ssf", pick_thres=4)
#jobS = job_scheduler("microsoft-160")
#jobS.allocate(algo="slf", pick_thres=4)
#jobS = job_scheduler("microsoft-160")
#jobS.allocate(algo="bsf")

#jobS = job_scheduler("microsoft-160")
#jobS.allocate(algo="blf")
##jobS.print_jobs()
#
## compare three comm algos
#jobS.schedule(thres=0, adaDual=False)
#jobS.schedule(thres=1, adaDual=False)
#jobS.schedule(thres=1, adaDual=True)
#jobS.print_stat()

