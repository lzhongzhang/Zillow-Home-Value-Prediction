filename = 'train_2016_v2.csv'

with open(filename, 'r') as ori:
    with open('n_'+filename, 'a') as nori:
        for line in ori:
            tid, err, data = line.strip().split(',')
            nori.write(tid + ',' + err + ',' + data[5:7] + '\n')
