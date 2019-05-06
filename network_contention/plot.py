import matplotlib.pyplot as plt
import glob
import itertools
import numpy as np

markers=['.','x','o','v','^','<','>','1','2','3','4','8','s','p','*']
#markers=[None]
colors = ['b', 'g', 'r', 'm', 'y', 'k', 'orange', 'purple', 'olive']
markeriter = itertools.cycle(markers)
coloriter = itertools.cycle(colors)

def parse_log(log_root):

    logs = glob.glob(r'%s/*.log' % log_root)
    latencies = []
    for log in logs:
        with open(log, 'r') as f:
            content = f.readlines()
        latencies.append(float(content[-1].split()[-1]))

    #return np.mean(latencies)
    #return min(latencies)
    return latencies

def main():

    job_nums = [1, 2, 4, 8, 16, 32]
    msg_sizes = [64, 256, 1024, 4096, 16384, 65536, 262144]
    ther_max = [12.5 for n in range(len(job_nums))]
    #thrt_max = [12.5 for n in range(len(job_nums))]

    fig, ax = plt.subplots(figsize = (12, 8))
    for msg_size in msg_sizes:
        
        throughput = []
        for job_num in job_nums:
            #latency.append(parse_log("logs/job_n%d_s%d" % (job_num, msg_size)))
            throughput.append(sum([msg_size / l for l in parse_log("logs/job_n%d_s%d" % (job_num, msg_size))]))

        #plt.plot(job_num, latency)
        #throughput = [msg_size / l for l in latency]
        marker = markeriter.next()
        color = coloriter.next()
        ax.plot(throughput, label="size=%d B" % (msg_size), marker=marker, markerfacecolor='none', color=color)
    marker = markeriter.next()
    color = coloriter.next()
    ax.plot(ther_max, label="theoretical Full-Occupied", marker=marker, markerfacecolor='none', color=color)
        
    ax.set_xlabel("Job Number", size=18)
    ax.set_ylabel("Throughput/MB", size=18)
    ax.yaxis.set_tick_params(labelsize=18)
    ax.set_ylim(top=35)
    ax.set_xticklabels([0, 1, 2, 4, 8, 16, 32], size=18)
    ax.grid(linestyle=':')
    ax.legend(fontsize=18, loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()
