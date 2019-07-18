from job import dag_task, dag_job

class gpu:

    def __init__(self, mem, gpu_id):
        
        self.gpu_mem = mem
        self.gpu_id = gpu_id
        self.job_list = []
        self.wk_id_list = []
        self.task_list = []

        self.allocated_mem = 0
        self.free_mem = self.gpu_mem
        self.workload = 0

        self.is_busy = False
        self.event_start_time = 0
        self.event_end_time = 999

    # allocate stage
    def add_job(self, job, worker_id):
        self.job_list.append(job)
        self.wk_id_list.append(worker_id)
        self.workload += (job.fw_time + job.bw_time) * job.iters

        job.add_gpu(self)

    #def get_ready_jobs(self):
    #    ready_jobs = []
    #    for idx, job in enumerate(self.job_list):
    #        worker_id = self.wk_id_list[idx]
    #        if job.is_ready(worker_id):
    #            ready_jobs.append(job)

    #    return ready_jobs

    def update_status(self, time):
        if time == self.event_end_time:
            self.is_busy = False
            self.cur_job.pop_task(self.cur_worker_id)
        
    def add_run(self, job, time):
        worker_id = self.wk_id_list[self.job_list.index(job)]
        #print worker_id

        self.is_busy = True
        self.event_start_time = time
        self.event_end_time = time + 20
        self.cur_job = job
        self.cur_worker_id = worker_id
        #cur_task = job.pop_task(worker_id)

        return job.job_id, job.get_task(worker_id)

    def get_avail_compute(self):

        jobs = []
        for job in self.job_list:
            worker_id = self.wk_id_list[self.job_list.index(job)]

            if job.is_finished:
                continue
            #if (job.is_communicated[worker_id] == -1) or (sum(job.is_communicated) == 0):
            if job.is_communicated[worker_id] == -1:
                jobs.append(job)

        return jobs
        
    
class node:

    def __init__(self, cpu_mem, num_gpu, net_spd, node_id):

        self.cpu_mem = cpu_mem
        self.num_gpu = num_gpu
        self.net_spd = net_spd
        self.node_id = node_id

        self.gpu_list = [gpu(mem=8192, gpu_id=i) for i in range(self.num_gpu)]
        self.comm_task_list = []
        self.event_start_time = []
        self.event_end_time = []
        self.job_list = []

        self.net_load = 0
        self.compute_load = 0

        self.net_conf = {"full_speed": 128.0,
                         "alpha": 0.0,
                         "beta": 1.0,
                         "eta": 0.3,
                         "num_of_task": 0}

    # allocate stage
    def add_job(self, job, gpu_to_give, allocated_worker):
        self.job_list.append(job)
        job.add_node(self)

        for i in range(gpu_to_give):
            
            self.gpu_list.sort(key=lambda x: x.workload)
            self.gpu_list[0].add_job(job, allocated_worker + i)
   
    def add_run(self, comm_task, time):
        comm_task.set_time(time)
        self.comm_task_list.append(comm_task)

    def is_comm(self):
        return True if len(self.comm_task_list) > 0 else False

    def update_status(self, time):
        for gpu in self.gpu_list:
            gpu.update_status(time)

        for task in self.comm_task_list:
            task.processing(time)

        self.comm_task_list = [task for task in self.comm_task_list if not task.is_done(time)]
        
    def update_comm(self):
        if len(self.comm_task_list) != self.net_conf["num_of_task"]:
            self.net_conf["num_of_task"] = len(self.comm_task_list)
            #for task in self.comm_task_list:
            #    log = "node %d: %s." % (self.node_id, task.update_comm_time(time, self.net_conf))
            #    print log

    def get_avail_comm(self):
        return [job for job in self.job_list if job.ready_for_comm()]

    def print_jobs(self):
        job_ids = [job.job_id for job in self.job_list]
        print self.node_id, job_ids

        for gpu in self.gpu_list:
            gpu_job_ids = [job.job_id for job in gpu.job_list]
            print "\t", gpu.gpu_id, gpu_job_ids, gpu.wk_id_list

    def update(self):
        
        # update network workload
        self.net_load = 0
        for job in self.job_list:
            self.net_load += job.model_size

        # update compute workload
        self.compute_load = 0
        for gpu in self.gpu_list:
            self.compute_load += gpu.workload


class cluster:

    def __init__(self, config): # config is a dict

        self.num_node = config["num_node"]
        self.node_list = [node(config["cpu_mem"], config["num_gpu"], config["network_speed"], i) for i in range(self.num_node)]

    def update(self):

        for node in self.node_list:
            node.update()
        
