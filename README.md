# Zillow’s Home Value Prediction (Zestimate)

This is a project with big data techniques. I used R to do a data analysis, used PySpark to do data cleansing and build model. Finally, we used the model to perform predictions on millions of instances.

For a wrapped-up report, please refer [here](./FinalProjectReport.pdf).

## DataSet Source

- All the properties with their home features for 2016:
  [properties_2016.csv](https://www.kaggle.com/c/zillow-prize-1/download/properties_2016.csv.zip)

- The training set with transactions from 1/1/2016 to 12/31/2016:
  [train_2016_v2.csv](https://www.kaggle.com/c/zillow-prize-1/download/train_2016_v2.csv.zip)


## Model Selection Random Forest Regression:

To select model, we first performed an inner join between `train_2016_v2.csv`
and `properties_2016.csv` locally, to select the home features of the
properties that were sold from 1/1/2016 to 12/31/2016.

The python script is as follow:

```python
# Script filename: src/join.py
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
```

The script will generate a new csv file `trainingData.csv`. Upload the file to
Databricks cluster. Next operations are as follow:

- [Random Forest
  Model](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/7299685736592057/2614468934209616/3042362412750717/latest.html),
  the corresponding python script is stored as
  `src/finalProject_randomforest.py`.

- [Gradient-Boosted Trees
  Model](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/5516575657271442/1185864460293412/8718661597938584/latest.html),
  the corresponding python script is stored as `src/finalProject_GBT.py`.

Based on results, random forest model was chosen.

## Final Model Build

After choosing the model, we now consider the whole dataset.

### Use Dataframe to Build Model:

DataFrame API read in [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
format. We need to transform the dataset in to format of

> `Tuple( Label, List[(feature id, feature value)] )`

Since Databricks's limit computation ability, transformations are accomplished
locally.

1. First reduce dimension. Since in pre-processing data analysis, we have
   identified column with low correlation. Directly drop those column.

   ```python
    # Script filename: src/drop.py
    def drop():
        """
        Drop the column with low correlation by the index.
        """
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
    ```

    The script would read `properties_2016.csv` file line by line, and drop
    specific column, and store the data into file `drop_properties_2016.csv`.

2. Transform the csv data to `libsvm` format.

    ```python
    # Script filename: src/csv2libsvm.py
    def csv2libsvm():
        """Transform the csv file to libsvm format."""
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
    ```

    The script would read `drop_properties_2016.csv` generateid in first step
    line by line. For each line, split and parse, two files would be generated.

    + `libsvm_drop_properties_2016.csv`

      For each line, set label as 0 (i.e. to be predict), generate
      `libsvm` format data based on the data list after split.

    + `libsvm_train.csv`

      For each line after split, if the first item (i.e. the `parcel_id`) exist
      in the `train_2016_v2.csv` (i.e. the data can be used for training), set
      label as the entry in the `train_2016_v2.csv` and generate a record in
      `libsvm_train.csv`.

    3. For prediction, we need to add transaction date to be predicted.

    ```python
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
    ```

    The script would read `libsvm_drop_properties_2016.csv` generated in first
    step line by line. Based on specific value (which should be an integer no
    greater than 12), add a twenty-eighth attribute at the end of each line.
    The new generated file would be `predict_libsvm_drop_properties_2016.csv`.

After these three steps, upload the files to Databricks cluster. Train based on
`libsvm_train.csv`, and predict with `predict_libsvm_drop_properties_2016.csv`.
Operations are as follow:

- [Build Model and Predict with DataFrame
  API](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/7299685736592057/3514397187244983/3042362412750717/latest.html),
  the corresponding python script is stored as `src/DF_RandomForestModel.py`.

### Use RDD to Build Model:

RDD API could directly read csv file. However, to simplify data, we first
change the transaction date in `train_2016_v2.csv` as two digit integer (i.e.
discard the `year` in the date). It is easily achieved by python with following
script.

```python
# Script filename: modifydata.py
filename = 'train_2016_v2.csv'

with open(filename, 'r') as ori:
    with open('n_'+filename, 'a') as nori:
        for line in ori:
            tid, err, data = line.strip().split(',')
        nori.write(tid + ',' + err + ',' + data[5:7] + '\n')
```

The script simply read `train_2016_v2.csv` line by line and delete the year of
the transaction date, and a new file named `n_train_2016_v2.csv` would be
generated..

After such modification, upload the file `n_train_2016_v2.csv` and
`properties_2016.csv` to Databricks cluster, and next operations are as follow:

- [Build Model and Predict with RDD
  API](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/7299685736592057/1213976575370233/3042362412750717/latest.html),
  the corresponding python script is stored as `src/PredictRandomForest.py`.
