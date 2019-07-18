import os, glob
from random import sample, randint
import json, yaml

SUPPORT_NETS = ["resnet20", "lstm", "lstman4", "resnet50"]
TEMPLATES = {
             "resnet20": {"model_size":100, "lr":0.1, "dataset":"cifar10", "data_dir":"data", "fw_time":100, "bw_time":100, "batch_size":32},
             "lstm": {"model_size":100, "lr":0.1, "dataset":"ptb", "data_dir":"data", "fw_time":100, "bw_time":100, "batch_size":20},
             "lstman4": {"model_size":100, "lr":0.1, "dataset":"an4", "data_dir":"data", "fw_time":100, "bw_time":100, "batch_size":20},
             "resnet50": {"model_size":100, "lr":0.1, "dataset":"imagenet", "data_dir":"data", "fw_time":100, "bw_time":100, "batch_size":8},
}

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
            #job_json["dnn"] = sample(SUPPORT_NETS, 1)[0]
            job_json["dnn"] = "resnet20"
            job_json.update(TEMPLATES[job_json["dnn"]])

            #job_json["nworkers"] = 2 ** randint(1, 3)
            #job_json["cuda_enabled"] = 1
            job_json["nworkers"] = 2
            job_json["cuda_enabled"] = 0
            job_json["nsteps_update"] = 1
            job_json["iters"] = 10

            with open(os.path.join(self.job_root, "job_%d.json"%i), "w") as f:
                yaml.safe_dump(job_json, f)
            

class job_scheduler:

    def __init__(self, set_name):
        self.job_root = "job_configs/%s" % set_name
        self.job_files = glob.glob(r'job_configs/%s/job*.json' % set_name)
        self.num_jobs = len(self.job_files)

        self.job_set = []

    def print_jobs(self):
        for jf in self.job_files:
            with open(jf, 'r') as f:
                job_json = yaml.safe_load(f)
                self.job_set.append(job_json)
                print job_json

    def direct_schedule(self):

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


jobG = job_generator("test_1jobs_cpu", 1)
jobG.random_generate()
jobS = job_scheduler("test_1jobs_cpu")
jobS.print_jobs()
jobS.direct_schedule()
