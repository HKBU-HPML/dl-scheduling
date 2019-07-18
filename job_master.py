import os, glob
from random import sample, randint
import json, yaml
from cluster import *
from job import dag_job

SUPPORT_NETS = ["resnet20", "lstm", "lstman4", "resnet50"]
TEMPLATES = {
             "resnet20": {"model_size":100, "lr":0.1, "dataset":"cifar10", "data_dir":"data", "fw_time":100, "bw_time":100, "batch_size":32},
             "lstm": {"model_size":100, "lr":0.1, "dataset":"ptb", "data_dir":"data", "fw_time":100, "bw_time":100, "batch_size":20},
             "lstman4": {"model_size":100, "lr":0.1, "dataset":"an4", "data_dir":"data", "fw_time":100, "bw_time":100, "batch_size":20},
             "resnet50": {"model_size":100, "lr":0.1, "dataset":"imagenet", "data_dir":"data", "fw_time":100, "bw_time":100, "batch_size":8},
}

CLUSTER = {"num_node": 4, "num_gpu":4, "gpu_mem":8192, "cpu_mem":16384, "network_speed": 128} # unit is MB. 

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
            job_json["iters"] = randint(100, 500)

            with open(os.path.join(self.job_root, "job_%d.json"%i), "w") as f:
                yaml.safe_dump(job_json, f)
            

class job_scheduler:

    def __init__(self, set_name):
        self.job_root = "job_configs/%s" % set_name
        self.job_files = glob.glob(r'job_configs/%s/job*.json' % set_name)
        self.num_jobs = len(self.job_files)

        self.job_set = []
        self.load_job_set()

        self.clust = cluster(CLUSTER)

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

    def allocate(self):
        
        self.job_set.sort(key=lambda x:x.nworkers, reverse=True)
        self.print_jobs()

        def get_node_compute_workload(node):
            return node.get_gpus_workload()

        for job in self.job_set:

            finished = False
            allocated_worker = 0
            while not finished:
                self.clust.update()
                self.clust.node_list.sort(key=lambda x:x.compute_load)
                #self.clust.node_list.sort(key=get_node_compute_workload)

                #for node in self.clust.node_list:
                #    print node.node_id, node.get_gpus_workload()

                gpu_to_give = min(job.nworkers, self.clust.node_list[0].num_gpu)
                self.clust.node_list[0].add_job(job, gpu_to_give, allocated_worker)
                allocated_worker += gpu_to_give
                if allocated_worker == job.nworkers:
                    finished = True
                
        for node in self.clust.node_list:
            node.print_jobs()

    def check_finished(self):
        finished_ids = []
        for job in self.job_set:
            if job.is_finished:
                finished_ids.append(job.job_id)

        return finished_ids

    def schedule(self):

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

            for job in self.job_set:
                if not job.is_finished and job.ready_for_comm():
                    if len(job.nodes) == 1:
                        job.sync_comm()
                    else:
                        if job.is_all_nodes_free():
                            comm_task = job.get_comm()
                            for node in job.nodes:
                                node.add_run(comm_task, time)
                            job.lock_comm()
                            print_log += "job-%d running comm %s.\n" % (job.job_id, job.get_nodes())
                    
            # update comm speed
            for node in self.clust.node_list:
                node.update_comm()

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

                    chosen_compute = max(candidate_compute, key=lambda x:x.nworkers)

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

            schedule = job.copy()

            # allocate nodes and GPUs
            node_gpu = gpu_allocate(job['nworkers'])
            hostfile = os.path.join(self.job_root, "cluster_j%d" % job['job_id'])
            schedule["hostfile"] = hostfile
            schedule["gpus"] = []
            with open(hostfile, "w") as f:
                for node in node_gpu:
                    f.write("%s slots=%d\n" % (node, len(node_gpu[node])))
                    schedule["gpus"].extend(node_gpu[node])
            
            # schedule the tasks
            schedule["schedule"] = {}
            for r in range(job['nworkers']):
                tmp_plan = {}
                f = []
                b = []
                c = []
                for i in range(job['iters']):
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

#jobG = job_generator("test_16jobs", 16)
#jobG.random_generate()
jobS = job_scheduler("test_16jobs")
jobS.print_jobs()
jobS.allocate()
jobS.schedule()
