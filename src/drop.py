# Script filename: drop.py
def drop():
    """ Drop the column with low correlation by the index. """

    fn = 'properties_2016.csv'

    with open(fn, 'r') as ori:
        with open('drop_'+fn, 'a') as nfile:
            for line in ori:
                l = line.strip().split(',')
                d58 = [57, 56, 55, 49, 46, 45, 43, 42, 41, 37, 35, 31, 30, 29,
                        28, 27, 22, 19, 18, 16, 15, 14, 13, 12, 11, 10, 9, 8,
                        6, 3, 2]
                d59 = [58, 57, 56, 50, 47, 46, 44, 43, 42, 38, 35, 34, 31, 30,
                        29, 28, 27, 22, 19, 18, 16, 15, 14, 13, 12, 11, 10, 9,
                        8, 6, 3, 2]
                if len(l) == 59:
                    for n in d59:
                        del l[n]
                else:
                    for n in d58:
                        del l[n]
                newstr = ','.join(l) + '\n'
                nfile.write(newstr)
drop()
