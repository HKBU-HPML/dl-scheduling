import numpy as np
import matplotlib.pyplot as plt
import glob

def draw_scheduling(job_set):

    fig, ax = plt.subplots(figsize = (16, 4))

    job_list = glob.glob(r'logs/%s/*.log' % job_set)
    job_list.sort()

    print job_list
    job_name = [fn.split("/")[-1].replace(".log", "") for fn in job_list]
    print job_name

    y_bases = (np.arange(len(job_name)) + 1) * 0.5
    for idx, job in enumerate(job_list):
        draw_one_worker(job, y_bases[idx], fig, ax)

    ax.set_yticks(y_bases + 0.2)
    ax.set_yticklabels(job_name)

    plt.show()

def draw_one_worker(fn, y_base, fig, ax):

    offset = 0.4
    with open(fn, "r") as f:
        content = f.readlines()

    # filter out different task types
    fw_lines = [l.replace("...", "") for l in content if "Forward task" in l]
    fw_lines = [l.split("=")[1:] for l in fw_lines]
    fw_ts = [[float(t.split(",")[0].strip()) for t in l] for l in fw_lines]

    bw_lines = [l.replace("...", "") for l in content if "Backward task" in l]
    bw_lines = [l.split("=")[1:] for l in bw_lines]
    bw_ts = [[float(t.split(",")[0].strip()) for t in l] for l in bw_lines]

    comm_lines = [l.replace("...", "") for l in content if "Comm task" in l]
    comm_lines = [l.split("=")[1:] for l in comm_lines]
    comm_ts = [[float(t.split(",")[0].strip()) for t in l] for l in comm_lines]

    print fw_ts
    print bw_ts
    print comm_ts

    for ft in fw_ts:
        x = np.arange(ft[0], ft[1] + 1)
        y_lower = np.ones(ft[2] + 1) * y_base
        y_upper = np.ones(ft[2] + 1) * (y_base + offset)
        ax.fill_between(x, y_lower, y_upper, facecolor = 'red')

    for bt in bw_ts:
        x = np.arange(bt[0], bt[1] + 1)
        y_lower = np.ones(bt[2] + 1) * y_base
        y_upper = np.ones(bt[2] + 1) * (y_base + offset)
        ax.fill_between(x, y_lower, y_upper, facecolor = 'green')

    for ct in comm_ts:
        x = np.arange(ct[0], ct[1] + 1)
        y_lower = np.ones(ct[2] + 1) * y_base
        y_upper = np.ones(ct[2] + 1) * (y_base + offset)
        ax.fill_between(x, y_lower, y_upper, facecolor = 'blue')

#draw_one_worker("logs/job_set_1/0-gpu15-0.log", 0.5)
draw_scheduling("job_set_1")
    
