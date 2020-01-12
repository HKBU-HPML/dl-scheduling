import os, glob, sys
from random import sample, randint
import json, yaml
from cluster import *
from job import dag_job
import numpy as np
import random, math
import operator

#SUPPORT_NETS = ["resnet20", "lstm", "lstman4", "resnet50"]
SUPPORT_NETS = ["vgg16", "resnet50", "inception-v3", "lstm"]
TEMPLATES = {
             #"resnet20": {"model_size":100, "lr":0.1, "dataset":"cifar10", "data_dir":"data", "fw_time":100, "bw_time":100, "batch_size":32},
             #"lstman4": {"model_size":100, "lr":0.1, "dataset":"an4", "data_dir":"data", "fw_time":100, "bw_time":100, "batch_size":20},
             "lstm": {"model_size":251.8, "lr":0.1, "dataset":"ptb", "data_dir":"data", "fw_time":32, "bw_time":47, "batch_size":64, "train_mem":2751},
             "vgg16": {"model_size":526.4, "lr":0.1, "dataset":"imagenet", "data_dir":"data", "fw_time":36, "bw_time":54, "batch_size":16, "train_mem":4527},
             "resnet50": {"model_size":99.2, "lr":0.1, "dataset":"imagenet", "data_dir":"data", "fw_time":25, "bw_time":37, "batch_size":16, "train_mem":3213},
             "inception-v3": {"model_size":103.0, "lr":0.1, "dataset":"imagenet", "data_dir":"data", "fw_time":35, "bw_time":52, "batch_size":16, "train_mem":3291},
             #"alexnet": {"model_size":235, "lr":0.1, "dataset":"imagenet", "data_dir":"data", "fw_time":45, "bw_time":85, "batch_size":256},
             #"resnet50": {"model_size":97.7, "lr":0.1, "dataset":"imagenet", "data_dir":"data", "fw_time":19, "bw_time":33, "batch_size":4},
             #"googlenet": {"model_size":26.7, "lr":0.1, "dataset":"imagenet", "data_dir":"data", "fw_time":30, "bw_time":46, "batch_size":16},
             #"alexnet": {"model_size":235, "lr":0.1, "dataset":"imagenet", "data_dir":"data", "fw_time":22, "bw_time":42, "batch_size":64},
}

CLUSTER = {"num_node": 16, "num_gpu":4, "gpu_mem":32768, "cpu_mem":16384, "network_speed": 1.28} # unit is MB.
scale = 1
ARRIVAL_MAX = 1200 * scale # 1.2 seconds

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
            job_json["iters"] = randint(1, 5)
            #job_json["start_time"] = randint(0, ARRIVAL_MAX / 4)
            job_json["start_time"] = randint(0, self.num_jobs / 4)

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

        min_iter = 1 * scale
        max_iter = 6 * scale

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
            job_json["iters"] = randint(min_iter, max_iter) 
            job_json["start_time"] = int(math.floor(np.random.uniform(0, ARRIVAL_MAX)))

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
            job_json["iters"] = randint(min_iter, max_iter) 
            job_json["start_time"] = int(math.floor(np.random.uniform(0, ARRIVAL_MAX)))

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
            job_json["iters"] = randint(min_iter, max_iter) 
            job_json["start_time"] = int(math.floor(np.random.uniform(0, ARRIVAL_MAX)))

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
            job_json["start_time"] = int(math.floor(np.random.uniform(0, ARRIVAL_MAX)))

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
            job_json["start_time"] = int(math.floor(np.random.uniform(0, ARRIVAL_MAX)))

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
            job_json["start_time"] = int(math.floor(np.random.uniform(0, ARRIVAL_MAX)))

            with open(os.path.join(self.job_root, "job_%d.json"%i), "w") as f:
                yaml.safe_dump(job_json, f)

            
class job_scheduler:

    def __init__(self, set_name):
        self.job_root = "job_configs/%s" % set_name
        self.job_files = glob.glob(r'job_configs/%s/job*.json' % set_name)
        self.num_jobs = len(self.job_files)

        self.job_set = [[] for i in range(ARRIVAL_MAX)] # simulate one day of 1440 minutes
        self.load_job_set()

        self.job_queue = []
        self.starve_queue = []
        self.job_running = []

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
                self.job_set[job_json["start_time"]].append(dag_job(job_json))

    def allocate(self, time, algo="blf", pick_thres=32, starve_thres=999999): # "ssf", "slf", "bsf", "blf", "random" 
 
        def pick_gpus(job, pick_thres):

            selected_gpus = [] 
            if job.nworkers <= pick_thres:
                available_gpus = [gpu for gpu in self.clust.gpu_list if gpu.rest_mem >= job.train_mem]
                # select the gpu with the minimum makespan
                available_gpus = sorted(available_gpus, key=lambda x: (x.makespan, x.node_id))
                #print ["n%d-g%d-%d" % (g.node_id, g.gpu_id, g.makespan) for g in self.clust.gpu_list]
                if len(available_gpus) >= job.nworkers:
                    selected_gpus = available_gpus[:job.nworkers]

            else:
                # select nodes with least makespan
                for host_node in self.clust.node_list:
                    host_node.update_makespan()
                self.clust.node_list = sorted(self.clust.node_list, key=lambda x: (x.makespan))

                available_gpus = []
                for node in self.clust.node_list:
                    available_gpus.extend([gpu for gpu in node.gpu_list if gpu.rest_mem >= job.train_mem])

                if len(available_gpus) >= job.nworkers:
                    selected_gpus = available_gpus[:job.nworkers]
            
            # not enough available gpus
            if len(selected_gpus) == 0:
                return
            # add the job to nodes and update the comm overhead
            nodes = [g.host_node for g in selected_gpus]
            selected_node_ids = []
            for n in nodes:
                if not n.node_id in selected_node_ids:
                    n.add_job(job)
                    selected_node_ids.append(n.node_id)
             
            if len(selected_node_ids) > 1:
                job.comm_duration = job.model_size * 7.8125 * job.iters
                job.comm_duration_once = job.model_size * 7.8125

            # add the job to gpus and update the makespan
            #max_makespan = max([g.makespan for g in selected_gpus])   
            for i, g in enumerate(selected_gpus):
                g.add_job(job, i)
            job.allocate_gpu_memory()

            # remove job from queue and add it to running
            if job in self.starve_queue:
                self.starve_queue.remove(job)
            if job in self.job_queue:
                self.job_queue.remove(job)
            self.job_running.append(job)

            print "Time-%d, allocate job-%d(st: %d) to gpus:" % (time, job.job_id, job.start_time), ["n-%d g-%d(%d/%d)" % (g.node_id, g.gpu_id, g.rest_mem, (g.rest_mem + g.allocated_mem)) for g in selected_gpus]


        # pick up those jobs that have waited too long
        self.starve_queue.extend([job for job in self.job_queue if (time - job.start_time) >= starve_thres])
        self.job_queue = [job for job in self.job_queue if job not in self.starve_queue]

        if algo == "sf":
            self.job_queue = sorted(self.job_queue, key=lambda x: x.nworkers)
        elif algo == "stf":
            self.job_queue = sorted(self.job_queue, key=lambda x: x.compute_duration)
        elif algo == "ssf":
            self.job_queue = sorted(self.job_queue, key=lambda x: x.gpu_compute_cumul_duration)
        elif algo == "blf":
            self.job_queue = sorted(self.job_queue, key=lambda x: (-x.nworkers, -x.compute_duration))
        elif algo == "random":
            random.shuffle(self.job_queue)
           
        #print ["job-%d-w%d-%d" % (j.job_id, j.nworkers, j.compute_duration) for j in self.job_set]
        #self.print_jobs()

        for job in self.starve_queue:

            # place the job
            pick_gpus(job, pick_thres)
            #print ["job-%d-%d to n-%d g-%d" % (job.job_id, job.compute_duration, g.node_id, g.gpu_id) for g in selected_gpus]

        for job in self.job_queue:
            # place the job
            pick_gpus(job, pick_thres)
            #print ["job-%d-%d to n-%d g-%d" % (job.job_id, job.compute_duration, g.node_id, g.gpu_id) for g in selected_gpus]

    def check_finished(self):
        finished_ids = []
        for job in self.job_running:
            if job.is_finished:
                finished_ids.append(job.job_id)
                job.release_gpu_memory()  # release GPU memory
                self.job_running.remove(job)

        return finished_ids

    def schedule(self, place_str='blf', pt=2, schedule_str='srsf', comm_thres=1, adaDual=True):

        for t in range(0, len(self.job_set)):
            for job in self.job_set[t]:
                 job.initial_task()

        cur_time = 0
        
        finished_ids = []

        time = 0
        print "Placement Algorithm: %s-%d, starve-free: %d." % (place_str, pt, 999999)
        while len(finished_ids) != self.num_jobs:

            print_log = ""

            # update compute/comm task list, check whether some have ended
            for node in self.clust.node_list:
                node.update_status(time)

            # process those finished jobs, record them
            tmp_finished_ids = self.check_finished()
            if len(tmp_finished_ids) != 0:
                finished_ids.extend(tmp_finished_ids)
                finished_ids.sort()
                print_log += "Finished ID: %s.\n" % finished_ids

            # enqueue new arriving jobs
            if time < ARRIVAL_MAX:
                self.job_queue.extend(self.job_set[time])

            # allocate resources(GPUs and their memory) for jobs in the queue, placement them if some GPUs are available
            self.allocate(time, algo=place_str, pick_thres=pt)

            comm_jobs = []
            for job in self.job_running:
                if not job.is_finished and job.ready_for_comm():
                    if len(job.nodes) == 1:
                        job.sync_comm(time)
                    else:
                        comm_jobs.append(job)

            if schedule_str == 'sf':
                comm_jobs = sorted(comm_jobs, key=lambda x: x.nworkers) # sf
            elif schedule_str == 'srtf':
                comm_jobs = sorted(comm_jobs, key=lambda x: x.remain_compute_workload) # srtf
            elif schedule_str == 'srsf':
                comm_jobs = sorted(comm_jobs, key=lambda x: x.remain_compute_cumul_workload) # srsf

            for job in comm_jobs:
                if job.is_all_nodes_free(time, thres=comm_thres, adaDual=adaDual):
                    comm_task = job.get_comm()
                    for node in job.nodes:
                        node.add_run(comm_task, time)
                    job.lock_comm()
                    print_log += "job-%d running comm iter-%d %s.\n" % (job.job_id, job.available_tasks[0]["iter"], job.get_nodes())

            # update comm speed
            for node in self.clust.node_list:
                node.update_comm()

            # print the speed of those comm jobs
            for job in self.job_running:
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
                 
                    if schedule_str == 'sf':   
                        chosen_compute = sorted(candidate_compute, key=lambda x: x.nworkers)[0] # sf
                    elif schedule_str == 'srtf':
                        chosen_compute = sorted(candidate_compute, key=lambda x: x.remain_compute_workload)[0] # srtf
                    elif schedule_str == 'srsf':
                        chosen_compute = sorted(candidate_compute, key=lambda x: x.remain_compute_cumul_workload )[0] # srsf

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
            print "Place Algorithm: %s(%d), Scheduling Algorithm: %s, adaDual enabled, thres=%d." % (place_str, pt, schedule_str, comm_thres)
        else:
            print "Place Algorithm: %s(%d), Scheduling Algorithm: %s, adaDual disabled, thres=%d." % (place_str, pt, schedule_str, comm_thres)
                
    def print_stat(self):

        for node in self.clust.node_list:
            print "node-%d: %d / %d." % (node.node_id, node.active_time, self.total_time)
            for gpu in node.gpu_list:
                print "\t gpu-%d: %d / %d." % (gpu.gpu_id, gpu.active_time, self.total_time)

        job_times = []
        for i in range(ARRIVAL_MAX):
            job_times.extend([(j.job_id, j.finish_time) for j in self.job_set[i]])
        aver_job_time = np.mean(job_times)
        print "Job Completion Time:"
        print job_times
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

## random job set for test
#num_jobs = 32
##jobG = job_generator("test_%djobs" % num_jobs, num_jobs)
##jobG.random_generate()
#jobS = job_scheduler("test_%djobs" % num_jobs)

## microsoft-80 job set
#jobG = job_generator("microsoft-80", 80)
#jobG.microsoft_generate()
#jobS = job_scheduler("microsoft-80")

# microsoft-160 job set
#jobG = job_generator("microsoft-160", 160)
#jobG.microsoft_generate()
jobS = job_scheduler("microsoft-160")

# compare three comm algos
adopted_algo = 'blf-2-srsf-0-true'
if len(sys.argv) == 2:
    adopted_algo = sys.argv[1]
place_str, pt, schedule_str, comm_thres, adaDual = adopted_algo.split('-')
pt = int(pt)
comm_thres = int(comm_thres)
if adaDual == 'true':
    adaDual = True
elif adaDual == 'false':
    adaDual = False
jobS.schedule(place_str, pt, schedule_str, comm_thres, adaDual)
jobS.print_stat()
