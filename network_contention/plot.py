import matplotlib.pyplot as plt

def parse_log(log_f):

    with open(log_f, 'r') as f:
       content = f.readlines()

    return float(content[-1].split()[-1])

def main():

    job_nums = [1, 2, 4, 8, 16]
    msg_sizes = [1024, 4096, 16384, 65536, 262144]

    for msg_size in msg_sizes:
        
        latency = []
        for job_num in job_nums:
            print "logs/job_n%d_s%d/job_1.log" % (job_num, msg_size)
            latency.append(parse_log("logs/job_n%d_s%d/job_1.log" % (job_num, msg_size)))

        #plt.plot(job_num, latency)
        throughput = [msg_size / l for l in latency]
        plt.plot(throughput)
    plt.show()

if __name__ == '__main__':
    main()
