from job import dag_task, dag_job

class gpu:

    def __init__(self, mem, gpu_id, host_node):
        
        self.gpu_mem = mem
        self.gpu_id = gpu_id
        self.host_node = host_node
        self.node_id = self.host_node.node_id
        self.job_list = []
        self.wk_id_list = []
        self.task_list = []

        self.allocated_mem = 0
        self.free_mem = self.gpu_mem
        self.workload = 0

        self.is_busy = False
        self.event_start_time = 0
        self.event_end_time = 9999999

        # variable for scheduling
        self.active_time = 0
        self.makespan = 0

        # gpu memory resource
        self.allocated_mem = 0
        self.rest_mem = self.gpu_mem

    def free_mem(self, model_size):
        self.allocated_mem -= model_size
        self.rest_mem += model_size

    def allocate_mem(self, model_size):
        self.allocated_mem += model_size
        self.rest_mem -= model_size

    # allocate stage
    def add_job(self, job, worker_id, max_makespan):
        self.job_list.append(job)
        self.wk_id_list.append(worker_id)

        self.makespan = max_makespan + job.compute_duration

        job.add_gpu(self)

    #def get_ready_jobs(self):
    #    ready_jobs = []
    #    for idx, job in enumerate(self.job_list):
    #        worker_id = self.wk_id_list[idx]
    #        if job.is_ready(worker_id):
    #            ready_jobs.append(job)

    #    return ready_jobs

    def update_status(self, time):
        # update statistical data
        cur_util = 0
        if self.is_busy == True:
            cur_util = 1
            self.active_time += 1

        if time == self.event_end_time:
            self.is_busy = False
            self.cur_job.pop_task(self.cur_worker_id)

        return cur_util
        
    def add_run(self, job, time):
        worker_id = self.wk_id_list[self.job_list.index(job)]
        #print worker_id

        task_duration = 0
        if job.available_tasks[worker_id]["type"] == "fw":
            task_duration = job.fw_time
        elif job.available_tasks[worker_id]["type"] == "bw":
            task_duration = job.bw_time

        self.is_busy = True
        self.event_start_time = time
        self.event_end_time = time + task_duration
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

        self.gpu_list = [gpu(mem=8192, gpu_id=i, host_node=self) for i in range(self.num_gpu)]
        self.comm_task_list = []
        self.event_start_time = []
        self.event_end_time = []
        self.job_list = []

        self.net_load = 0
        self.compute_load = 0

        self.net_conf = {"full_speed": 128.0,
                         "alpha": 0.0,
                         "beta": 1000.0 / 128.0,
                         "eta": 0.7,
                         "num_of_task": 0}

        # record the variable for scheduling
        self.active_time = 0
        self.makespan = 0

    # allocate stage
    def add_job(self, job):
        self.job_list.append(job)
        job.add_node(self)

        #for i in range(gpu_to_give):
        #    
        #    self.gpu_list.sort(key=lambda x: x.workload)
        #    self.gpu_list[0].add_job(job, allocated_worker + i)
   
    def update_makespan(self):
        self.makespan = max([g.makespan for g in self.gpu_list])

    def add_run(self, comm_task, time):
        comm_task.set_time(time)
        self.comm_task_list.append(comm_task)

    def is_comm(self):
        return True if len(self.comm_task_list) > 0 else False

    def update_status(self, time):
        cur_utils = 0
        for gpu in self.gpu_list:
            cur_utils += gpu.update_status(time)
        if cur_utils != 0:
            self.active_time += 1

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

        self.gpu_list = []
        for n in self.node_list:
            self.gpu_list.extend(n.gpu_list)

    def update(self):

        for node in self.node_list:
            node.update()
        
