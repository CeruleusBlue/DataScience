<span>
    <h5>The cell below:</h5>
    <ul>
        <li>Reads the names and the discrete or continuous nature of the features </li>
        <br>
        <li>Reads the data and loads the data into dataframe</li>
        <br>
        <li>Maps the type of connection except normal to its corresponding attack class</li>
        <br>
        <li>
            Drops the features which has only one value for all rows in the dataset
            <br>
            <i>
                Note: This has been done because these features wouldn't be able to classify the data
                in any way.
            </i>
        </li>
    </ul>
</spaN>

```py 
import pandas as pd, numpy as np
TARGETCOLUMN = 'connection_type'

names = pd.read_table('Task1/kddcup.names', names=['col1','col2'], skiprows=[0], sep=': ', engine='python')

df = pd.read_csv(
    'Task1/kddcup.data_10_percent_corrected', 
    names=pd.concat([names['col1'], pd.Series(TARGETCOLUMN)], ignore_index=True)
    )
# 0 : Symbolic
# 1 : COntinuous
columnValueType = [[],[]]
def getColumnType(x):
    if x['col2']=='symbolic.':
        columnValueType[0].append(x['col1'])
    else:
        columnValueType[1].append(x['col1'])   

names.apply(getColumnType, axis=1)

del names

attackclasses = {
    'DOS':['back', 'land', 'neptune','pod','smurf','teardrop'],
    'R2L':['ftp_write', 'guess_passwd', 'imap', 'multihop','phf','spy','warezclient','warezmaster'],
    'U2R':['buffer_overflow', 'loadmodule', 'perl', 'rootkit'],
    'Probe':['ipsweep','nmap','portsweep','satan']
}

def type_to_class(x:str)->str:
    x = x[0:len(x)-1]
    for i in attackclasses:
        if x in attackclasses[i]:
            return i
    return x
df[TARGETCOLUMN] = df[TARGETCOLUMN].apply(type_to_class)

del attackclasses
columns = []
for i in df.columns:
    if len(df[i].unique())==1:
        columns.append(i)
        if(i in columnValueType[0]):
            columnValueType[0].remove(i)
        else:
            columnValueType[1].remove(i)
df = df.drop(columns=columns)

del columns, i
```

<span>
    <h5>The cell below:</h5>
    <span>
        Separates the data into train and test. This has been done by sampling rows of
        each target column with a train test split of 7:3.
    </span>
</span>

```py
train_x = pd.DataFrame()

for i in  df[TARGETCOLUMN].unique():
    subset = df[df[TARGETCOLUMN]==i]
    train_x = pd.concat([train_x,subset.sample(int(0.7*len(subset))).drop(columns=[TARGETCOLUMN])])
del subset, i
train_y = df[TARGETCOLUMN].loc[train_x.index]
test_x= df.loc[:,df.columns!=TARGETCOLUMN].loc[~df.index.isin(train_x.index)]
test_y = df[TARGETCOLUMN].loc[test_x.index]
```

<span>
    <h5>The cell below:</h5>
    <span style='font-weight:700'>Creates the Naive Bayes model class</span>
    <ul>
        <li>The object of the class must be created with the unique values of the target data as arguments</li>
        <br>
        <li>The gaussian function takes the value and distribution as parameters and returns the log of the gaussian probability</li>
        <br>
        <li>
            The laplace function takes the frequency of the feature for in the subset containing only one target class value,
            the length of the aforementioned subset, the number of categorical features and the smoothing alpha as parameters.
            It then returns a smooth probability value.
        </li>
        <br>
        <li>
            The fit function:
            <ol>
                <li>Accepts the train features and target as parameters</li>
                <li>Iterates for each class present in the target data</li>
                <li>
                    For each iteration:
                    <ol>
                        <li>Creates a subset of train features with only rows which contain a specific target class</li>
                        <li>Stores probability of the occurence of the target class and the length of the target class</li>    
                        <li>Iterates each column of the subset</li>
                        <li>If the nature of the column values is continuous, stores the normal distribution of the column values</li>
                        <li>If the nature of the column values is discrete, stores the probability of the occurence of each distinct column value</li>
                    </ol>
                </li>
                <li>Returns a Naive Bayes object which has been fitted with train data</li>
            </ol>
        </li>
        <br>
        <li>
            The predict function:
            <ol>
                <li>Stores the priors for each target class</li>
                <li>Calls the rowPredictor function for each row of the test dataset</li>
                <li>
                    The rowPredictor function, calls the columnPredictor function passing along a dictionary 
                    of the probabilities of each target class for each column.
                    <br> 
                    The dictionary of probabilities already contains the priors
                    <br> 
                    It then returns dictionary of probabilities.
                </li>
                <li>
                    The column predictor function, iterates through each target class.
                    <br>
                    If the column is discrete, the log of the laplace smoothened probability is added to the dictionary.
                    <br>
                    If the column is continuous, the gaussian probability is added to the dictionary.  
                </li>
                <li>The dictionary of probabilities for each row is returned as a Pandas series for the predict function</li>
            </ol>
        </li>
    </ul>
    <span style='font-weight:700'>Creates a Naive Bayes object, fits the training data, and predicts the target class values for the test data</span >
</span>

```py        
class NaiveBayes:
    targetClasses = None
    def __init__(self, targetClasses) -> None:
        self.targetClasses = dict(zip(targetClasses, [dict() for x in targetClasses]))
    
    def gaussian(self, x, mean, std):
        if(std==0):
            return 1
        return np.log(1/(std*np.sqrt(2*np.pi)))*pow(np.e,-(0.5*pow((x-mean)/std, 2)))
    
    def laplace(self, n, N, k=len(columnValueType[0]),alpha = 0.0001):
        return (n+alpha)/(N+(alpha*k))

    def fit(self, train_x:pd.DataFrame, train_y:pd.Series):
        for i in self.targetClasses:
            subset = train_x.loc[train_y[train_y==i].index]
            self.targetClasses[i] = {
                "p(y)":len(subset)/len(train_x),
                "p(X)": dict(),
                "gaussian": dict(),
                "length": len(subset)}
            for j in train_x.columns:
                if subset[j].name in columnValueType[1]:
                    mean = subset[j].mean()
                    std = subset[j].std()
                    self.targetClasses[i]['gaussian'].update({j:{
                        'mean': mean, 
                        'std': std}})
                        
                else:
                    self.targetClasses[i]['p(X)'][j] = dict()
                    for k in train_x[j].unique():
                        if(k in subset[j].unique()):
                            self.targetClasses[i]['p(X)'][j].update({
                                k:self.laplace(len(subset[subset[j]==k]),len(subset))})
                        else:
                            self.targetClasses[i]['p(X)'][j].update({
                                k:self.laplace(0,len(subset))})
        return self

    def predict(self, test_x:pd.DataFrame):
        predictClassTemplate = dict(zip(self.targetClasses, [None]*len(self.targetClasses)))
        predictClasses = dict()
        for i in predictClassTemplate:
            predictClassTemplate[i] = np.log(self.targetClasses[i]['p(y)'])

        def columnPredictor(columnValue, columnName, predictRowClass):
            for i in predictRowClass:
                if(columnName in columnValueType[0]):
                    if columnName in self.targetClasses[i]['p(X)']:
                        probability = self.targetClasses[i]['p(X)'][columnName][columnValue]
                    else:
                        probability = self.laplace(0,self.targetClasses[i]['length'])
                    np.log(probability)
                else:    
                    mean = self.targetClasses[i]['gaussian'][columnName]['mean']
                    std =  self.targetClasses[i]['gaussian'][columnName]['std']
                    probability = self.gaussian(columnValue, mean, std)
                predictRowClass[i]+=probability

        def rowPredictor(row):
            rowProbability = None
            predictRowClass = predictClassTemplate.copy()
            for i in row.index:
                columnPredictor(row[i],i,predictRowClass)
            return predictRowClass
        
        return test_x.apply(rowPredictor,axis=1)

nb = NaiveBayes(df[TARGETCOLUMN].unique()).fit(train_x, train_y)
predictedProbabilities = nb.predict(test_x)
```

```
1         {'normal': -7.640525424546775, 'U2R': -22.5071...
2         {'normal': -8.132483750685834, 'U2R': -22.8953...
9         {'normal': -9.065408863972142, 'U2R': -19.1331...
16        {'normal': -11.897675880110306, 'U2R': -18.821...
17        {'normal': -12.791466245678883, 'U2R': -19.446...
                                ...                        
494011    {'normal': -12.783593698624781, 'U2R': -19.411...
494014    {'normal': -12.977825048731841, 'U2R': -19.681...
494015    {'normal': -13.733744926312813, 'U2R': -20.700...
494018    {'normal': -19.15888897344578, 'U2R': -19.4423...
494019    {'normal': -13.55357900906668, 'U2R': -19.9178...
Length: 148209, dtype: object
```
<span>
    <h5>The cell below: </h5>
    <ul>
        <li>Creates a function dictMax which returns the key of the maximum value present in a dictionary</li>
        <br>
        <li>Calculates the accuracy of the predicted values</li>
    </ul>
</span>

```py
def dictMax(x:dict):
    keyList = list(x.keys())
    valueList = list(x.values())
    maxVal, maxInd = 0, 0
    for i in range(len(x)):
        maxInd, maxVal = (maxInd, maxVal) \
            if (maxVal>valueList[i]) and (i!=0) \
                else (i, valueList[i])
    return keyList[maxInd]

sum(pd.concat({'predict':predictedProbabilities\
    .apply(dictMax), 'actual':test_y}, axis=1)\
        .apply(lambda x: 1 if x['predict']==x['actual'] else 0, axis=1))/len(test_x)
```

```
0.7923675350349844
```