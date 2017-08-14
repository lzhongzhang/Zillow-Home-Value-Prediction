# Script filename: src/csv2libsvm.py
def csv2libsvm():
    """ Transform the csv file to libsvm format. """

    dfn = 'drop_properties_2016.csv'
    training = "train_2016_v2.csv"

    pids = dict()
    vals = dict()
    val_c = 1

    with open(training) as tf:
        for line in tf:
            pid, logerror, tdate = line.split(",")
            if pid in pids:
                pids[pid] = [(logerror,tdate[5:7])] + pids[pid]
            else:
                pids[pid] = [(logerror,tdate[5:7])]

    with open(dfn, 'r') as dfnr:
        with open('libsvm_'+dfn, 'a') as nlibsvm:
            for l in dfnr:
                list_l = l.strip().split(',')
                nl = str()
                for i,val in enumerate(list_l):
                    if val == '':
                        continue
                    else:
                        try:
                            n_val = float(val)
                        except ValueError:
                            if val not in vals:
                                vals[val] = val_c
                                val_c = val_c + 1
                                n_val = vals[val]
                        nl = nl + str(i+1) + ':' + str(n_val) + ' '
                if list_l[0] in pids:
                    jdata = open('libsvm_train.csv', 'a')
                    for l,d in pids[list_l[0]]:
                        jdata.write(l + ' ' + nl + '28:' + d + '\n')
                    jdata.close()
                nl = '0 ' +  nl + '\n'
                nlibsvm.write(nl)

csv2libsvm()

