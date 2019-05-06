#import matplotlib.pyplot as plt
import glob
import pandas as pd

def parse_log(log_f):

    filelist = glob.glob(r'%s/*.log' % log_f)
    start_frame = pd.DataFrame({'Date':[], \
                               'Time':[], \
                               'Microsecond':[], \
                               'Latency':[]})
    end_frame = pd.DataFrame({'Date':[], \
                               'Time':[], \
                               'Microsecond':[], \
                               'Latency':[]})
 
    dfs = []
    for log in filelist:
        with open(log, 'r') as f:
            content = f.readlines()[4:]
            content = [line.split() for line in content]
            df = pd.DataFrame({'Date':[elem[0] for elem in content], \
                               'Time':[elem[1] for elem in content], \
                               'Microsecond':[elem[2] for elem in content], \
                               'Latency':[elem[3] for elem in content]})

            df['Date'] = df['Date'].astype('str')
            df['Time'] = df['Time'].astype('str')
            df['Microsecond'] = df['Microsecond'].astype('int')
            df['Latency'] = df['Latency'].astype('float')
            df = df.sort_values(by=['Date', 'Time', 'Microsecond'])
      
            start_frame.loc[len(start_frame)] = df.loc[0]
            end_frame.loc[len(end_frame)] = df.loc[len(df) - 1]

            dfs.append(df)

    start_frame = start_frame.sort_values(by=['Date', 'Time', 'Microsecond']).loc[len(start_frame) - 1]
    end_frame = end_frame.sort_values(by=['Date', 'Time', 'Microsecond']).loc[0]

    #print start_frame
    #print end_frame

    latencies = []
    for df in dfs:
        print len(df)
        data = df[(df['Date'] >= start_frame.Date) & (df['Time'] >= start_frame.Time) & (df['Microsecond'] >= start_frame.Microsecond) & \
                  (df['Date'] <= end_frame.Date) & (df['Time'] <= end_frame.Time) & (df['Microsecond'] <= end_frame.Microsecond)]

        print len(data)
        latencies.extend(list(data.Latency))
    #print latencies
    return np.mean(latencies)

def main():

    #job_nums = [1, 2, 4, 8, 16]
    #msg_sizes = [1024, 4096, 16384, 65536, 262144]
    job_nums = [16]
    msg_sizes = [65536]

    for msg_size in msg_sizes:
        
        latency = []
        for job_num in job_nums:
            latency.append(parse_log("logs/job_n%d_s%d" % (job_num, msg_size)))

        #plt.plot(job_num, latency)
    #plt.show()

if __name__ == '__main__':
    main()
