# Script filename: join.py
def join():
    """
    Join the training set and properties data.
    """
    properties = "properties_2016.csv"
    training = "train_2016_v2.csv"
    result = "trainingData.csv"

    pids = dict()

    with open(training) as tf:
        for line in tf:
            pid, logerror, tdate = line.split(",")
            if pid in pids:
                pids[pid] = [(logerror,tdate[:7])] + pids[pid]
            else:
                pids[pid] = [(logerror,tdate[:7])]

    combine = lambda a,b,c: a + ',' +  b + ',' + c + '\n'

    with open(properties) as pf:
        for line in pf:
            pid = line.split(",")[0]
            nline = line.rstrip()
            if pid in pids:
                with open(result, "a") as rf:
                    for l,d in pids[pid]:
                        rf.write(combine(nline,d,l))

join()


