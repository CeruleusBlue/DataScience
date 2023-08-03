<p>
    The cell below:
    <ul>
        <li>Imports the libraries required for data preprocesssing</li>
        <li>Loads the data into a dataframe</li>
    </ul>
</p>

```py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Task1/train.csv')
el = pd.read_csv('Task1/unique_m.csv')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Task1/train.csv')
el = pd.read_csv('Task1/unique_m.csv')
```
<p>
    The cell below plots the features of the dataframe as a scatter diagram in a 4x4 grid and saves them png files.
    <br><em>Note: The plot doesn't include the elements of the material, as it is a sparse matrix and no meaningful conclusions can drawn from their scatter diagrams.</em>
</p>

```py
y = df['critical_temp']
img_c = 1
for i in range(0,len(df.columns),16):
    plotmat,ax = plt.subplots(4,4)
    plotmat.set_size_inches(w=24,h=13.5)
    c, d = 0,-1
    jrange = range(i, len(df.columns)) if (i+16)>=len(df.columns) else range(i, i+16)
    for j in jrange:
        x = df[df.columns[j]]
        if(j%4==0): d=d+1
        ax_curr =  ax[c, d]
        ax_curr.set_title(df.columns[j])
        ax_curr.plot(x,y,'o')
        c = c+1
        if(c%4==0):c=0
        del ax_curr, x
    plotmat.savefig(fname=("Task1/plots/scatter_"+str(img_c)+".png"),format="png",bbox_inches='tight')
    img_c=img_c+1
    plt.close(plotmat)
    del c,d, plotmat, ax, jrange, j
del y, img_c, i
```
<p>
    The cell below:
    <ul>
        <li>Imports the sklearn library</li>
        <li>Scales the features to make it easier for regression models to predict the critical temperature as unscaled data might affect the accuracy of the models.</li>
    </ul>
</p>

```py
import sklearn as sk
from sklearn.model_selection import *

raw=df
scaler= sk.preprocessing.StandardScaler()
scaler.fit_transform(raw.drop(columns='critical_temp'))
df = pd.DataFrame(scaler.transform(raw.drop(columns='critical_temp')), columns=scaler.feature_names_in_)
df = df.join(raw['critical_temp'])

del scaler

```

<p>
    The cell below:
    <ul>
        <li>Imports three sklearn models, LinearRegression, MLPRegressor and SVR.</li>
        <li>Linear Regression is a simple form of machine learning and can provide insight on the data with very little complexity.</li>
        <li>Perceptrons and Vectors are two popular machine learning models and usually have a higher accuracy rate than Linear Regression</li>
        <li>Cross-validates each model and returns the performance data</li>
        <li>The scoring method for each model is the negative rmse.</li>
        <li>The data used to train the models in this dataset includes the elemental composition of the superconductors</li>
    </ul>
</p>

```py
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

df = pd.merge(left=df, right=el.drop(columns=['critical_temp','material']), left_index=True, right_index=True)

models = {
    'linear':LinearRegression(),
    'mlp':MLPRegressor(max_iter=1000),
    'svr':SVR(),
}

modelsCVData=dict()
for x in models:
    data = cross_validate(
        models[x], 
        df.drop(columns='critical_temp'),
        df['critical_temp'],
        scoring='neg_root_mean_squared_error'
    )
    modelsCVData.update({x:data.copy()})
    del data
    
modelsCVData

```

```
{'linear': {'fit_time': array([0.12003613, 0.1280036 , 0.10899305, 0.10699773, 0.11600399]),
  'score_time': array([0.00796247, 0.00799775, 0.00400162, 0.00499988, 0.00699782]),
  'test_score': array([-22.21040084, -38.66385678, -18.45108094, -26.12685429,
         -19.39930081])},
 'mlp': {'fit_time': array([70.24267817, 78.10168839, 70.16589141, 60.68983412, 59.38645649]),
  'score_time': array([0.01200175, 0.01199985, 0.01000428, 0.01799846, 0.01400256]),
  'test_score': array([-18.59710192, -46.5772667 , -32.52138018, -54.25534495,
         -32.82162623])},
 'svr': {'fit_time': array([45.61604357, 45.50501323, 44.28600526, 43.0784719 , 42.28161836]),
  'score_time': array([15.24100018, 15.35997033, 16.10524511, 15.17158699, 15.23725533]),
  'test_score': array([-22.60445818, -22.20926211, -12.90080624,  -6.09489383,
         -13.53382878])}}
```

<div style="color:rgb(47, 173, 53)">
    <p>
        The ouput above shows that the linear model has the fastest fit time with an approximate mean fit time of 0.1 second, followed by the svr model with 45 seconds and mlp model with 65 seconds.
    </p>
    <p>
        The scoring time is ranked differently to the fit time with the linear model being the fastest with 0.007 second, mlp with 0.01 second and svr with 15 seconds.
    </p>
</div>
<br>
<p>
The cell below calculates the mean score of the cross-validated data in the above cell. The mean score helps us to identify the best models.
</P>

```py
modelsCVDataMeanScore = dict()
for x in modelsCVData:
    s,c = sum(modelsCVData[x]['test_score']),len(modelsCVData[x]['test_score'])
    modelsCVDataMeanScore.update({x:(s/c)})
    del s,c
del x
modelsCVDataMeanScore
```
```
{'linear': -24.97029873372854,
 'mlp': -36.954543994736646,
 'svr': -15.468649828870536}
```
<p style="color:rgb(47, 173, 53)">
    The output shows that svr is the best model by a significant margin.
</p>
<br>
<p>
    The cell below splits the data into training and test data with a ratio of 2:1 and then fits the best model(svr) with the train data and returns the score for the test data
</p>

```py

train_x,test_x,train_y,test_y=train_test_split(df.drop(columns='critical_temp'),df['critical_temp'], test_size=(1/3))
models['svr'].fit(train_x, train_y)
models['svr'].score(test_x, test_y)

```
```
0.7750088693568424
```

<p style="color:rgb(47, 173, 53)">
    The output shows that the accuracy of the model is approximately 77.5% .
</p>
<br>
<p style="color:rgb(252, 85, 63)"> 
    The cell below performs the same steps as the previous cell which performs cross-validation with the difference being that the dataset for cross-validation doesn't include the elemental composition of the superconductors.
</p>

```py
modelsCVDataAlt=dict()
df = df.drop(columns=el.drop(columns=['critical_temp','material']).columns)
for x in models:
    data = cross_validate(
        models[x], 
        df.drop(columns='critical_temp'),
        df['critical_temp'],
        scoring='neg_root_mean_squared_error'
    )
    modelsCVDataAlt.update({x:data})
    del data
modelsCVDataAlt
```
```
{'linear': {'fit_time': array([0.04498529, 0.05800056, 0.05600214, 0.04200125, 0.04600143]),
  'score_time': array([0.00400162, 0.0030005 , 0.00299978, 0.00299859, 0.00299883]),
  'test_score': array([-23.40631078, -22.47723626, -16.982837  , -13.40407585,
         -16.45510657])},
 'mlp': {'fit_time': array([35.70056796, 43.3179996 , 49.71924686, 27.45067072, 39.91627717]),
  'score_time': array([0.00699878, 0.00499892, 0.00799966, 0.00799966, 0.0049994 ]),
  'test_score': array([-18.76551813, -18.91906919, -12.15287648,  -7.90920505,
         -19.05000037])},
 'svr': {'fit_time': array([27.39599633, 25.77498174, 24.80275345, 24.76703691, 24.39903069]),
  'score_time': array([9.93803358, 8.91700125, 9.02896452, 9.28699899, 8.90696764]),
  'test_score': array([-22.78818233, -22.8291182 , -13.52305512,  -5.95266532,
         -13.84947947])}}
```

<div style="color:rgb(47, 173, 53)">
    <p>
        The ouput above shows that the linear model has the fastest fit time with an approximate mean fit time of 0.05 second, followed by the svr model with 25 seconds and mlp model with 35 seconds.
    </p>
    <p>
        The scoring time is ranked differently to the fit time with the linear model being the fastest with 0.003 second, mlp with 0.07 second and svr with 25 seconds.
    </p>
</div>
<br>
<p>
The cell below calculates the mean score of the cross-validated data in the above cell. The mean score helps us to identify the best models.
</p>

```py
modelsCVDataMeanScoreAlt = dict()
for x in modelsCVDataAlt:
    s,c = sum(modelsCVDataAlt[x]['test_score']),len(modelsCVDataAlt[x]['test_score'])
    modelsCVDataMeanScoreAlt.update({x:(s/c)})
    del s,c
del x
modelsCVDataMeanScoreAlt
```

```
{'linear': -18.545113291953776,
 'mlp': -15.835865325351062,
 'svr': -15.788500087340537}
```

<p style="color:rgb(47, 173, 53)">
    The output shows that mlp and svr both have similar scores and therefore both will be used to check which provides the best accuracy.
</p>
<br>
<p>
    The cell below splits the data into training and test data with a ratio of 2:1 and then fits the best model(svr) with the train data and returns the score for the test data
</p>

```py
train_x,test_x,train_y,test_y=train_test_split(df.drop(columns='critical_temp'),df['critical_temp'], test_size=(1/3))
models['svr'].fit(train_x, train_y)
models['mlp'].fit(train_x, train_y)

print({'svr': models['svr'].score(test_x, test_y), 'mlp':models['mlp'].score(test_x, test_y)})
```
```
{'svr': 0.761359716629135, 'mlp': 0.8576887626967645}
```

<p style="color:rgb(47, 173, 53)">
    The output shows that mlp has a higher score than svr which itself doesn't differ much from the svr score calculated previously.
</p>

<table style="color:rgb(67, 147, 250)">
    <tr>
        <th>Model Name</th>
        <th>TestScore(with Elements)</th>
        <th>CVScore(with Elements)(-rmse)</th>
        <th>TestScore(without Elements)</th>
        <th>CVScore(without Elements)(-rmse)</th>
    </tr>
    <tr>
        <td>SVR</td>
        <td>0.7750088693568424</td>
        <td>-15.468649828870536</td>
        <td>0.761359716629135</td>
        <td>-15.788500087340537</td>
    </tr>
    <tr>
        <td>MLP</td>
        <td>-</td>
        <td>-36.954543994736646</td>
        <td>0.8576887626967645</td>
        <td>-15.835865325351062</td>
    </tr>
    <tr>
        <td>Linear</td>
        <td>-</td>
        <td>-24.97029873372854</td>
        <td>-</td>
        <td>-18.545113291953776</td>
    </tr>
</table>

<p style="color:rgb(67, 147, 250)"> Best Model(MLP Without Elements) accuracy = 85.7%</p>