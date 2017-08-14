# Script filename: src/add_transac.py
def add_transac(date):
    """ Add transaction date to libsvm file. """

    libsvm  = 'libsvm_drop_properties_2016.csv'
    plibsvm = 'predict_libsvm_drop_properties_2016.csv'

    with open(libsvm, 'r') as dfnr:
        with open(plibsvm, 'a') as nlibsvm:
            for l in dfnr:
                nlibsvm.write(l.strip() + ' 28:' + str(date) + '\n')

add_transac(10)
