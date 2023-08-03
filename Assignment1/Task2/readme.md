<p>
The cell below:
    <ol>
        <li>Imports all libraries used in the notebook</li>
        <li>Loads the dataset</li>
    </ol>
</p>

```py 
import pandas as pd
import numpy as np
import math
import copy

names=['parents', 'has_nurse', 'form', 'children','housing','finance', 'social','health', 'status']

df = pd.read_csv('Task2/nursery.data',names=names)


del names
```

<p>
The cell below:
    <ol>
        <li>Splits the dataset into train and test with a ratio 6 : 4</li>
    </ol>
</p>

```py

train = df.sample(int(len(df.index)*0.6))
test = df.loc[~df.index.isin(train.index)]
train_x,train_y = train.drop(columns='status'),train['status']
test_x, test_y = test.drop(columns='status'), test['status']
del train, test
```
<P>
    The cell below creates three binary decision trees. The trees differ with each other based on how the gain for each tree is calculated; Information Gain, Gain Ratio and GINI index.
    <br>
    <em>Note: Since the dataset only contains dataset discrete values, the model cannot work on datasets with continuous values</em>
    <br>
    It does this in the following way:
    <br>
    <ol>
        <li>fit
            <ul>
                <li>The function fit has two parameters the train data and train target.</li>
                <li>It builds a tree, three times. The build function is called to build the tree.</li>
            </ul>
        </li>
        <br>
        <li>build
            <ul>   
                <li>The build function calls itself recursively for every node which is not a leaf node.</li>
                <li>The build function calls the getBestFeature function for the current node to get the feature which scores the highest.</li>
                <li>It creates child nodes based on the best split feature</li>
            </ul>
        </li>
        <br>
        <li>getBestFeature
            <ul>   
                <li>The getBestFeature function performs nested iteration.</li>
                <li>The outer layer iterates through every column present in the dataset.</li>
                <li>The inner layer iterates through every unique value of the current iterated column present in the current node.</li>
                <li>Within the inner layer the entropy and the score for each unique value split is calculated</li>
                <li>Within the outer layer the best score for the unique value of the column is selected as the score of the column</li>
                <li>The unique value of the column with the best score is selected as the split criteria</li>
            </ul>
        </li>
    </ol>
</p>

```py
tree = None
treeList = None
TREESCHEMA = {
    'nodeList':list(),
    'trainData':pd.DataFrame,
    'trainTarget':pd.Series,
    'targetClasses':dict()
}

def getFrequency(currentColumn=[])->dict:
    targetClasses = dict()
    trainTarget = tree['trainTarget']
    if(len(currentColumn)==0):
        for x in tree['targetClasses']:
            targetClasses[x] = len(trainTarget[trainTarget==x])
    else:
        for x in tree['targetClasses']:
            targetClasses[x] = len(pd.merge(currentColumn,
                                        trainTarget[trainTarget==x],
                                        left_index=True, right_index=True))
    return targetClasses

def getEntropy(targetClasses=dict(), count=-1)->float:
    entropy = 0
    if(count==-1):
        count=0
        for x in targetClasses:
            count = count+targetClasses[x]
    for x in targetClasses:
        n = targetClasses[x]/count
        entropy = entropy+(n*math.log2(n)) if(n!=0) else entropy
    return -entropy

def getGini(targetClasses=dict(), count=-1)->float:
    gini = 1
    if(count==-1):
        count=0
        for x in targetClasses:
            count = count+targetClasses[x]
    for x in targetClasses:
        n = (targetClasses[x]/count)**2
        gini = gini - n
    return gini

def getInformationGain(split1,split2, parent=None)->float:
    parentNode = tree['nodeList'][parent]
    entropy0 = parentNode['data']['entropy']
    entropy1 = split1['entropy']
    entropy2 = split2['entropy']
    splitSize = split1['size']+split2['size']
    weight1 = split1['size']/splitSize
    weight2 = split2['size']/splitSize
    return (entropy0 - ((weight1*entropy1)+(weight2*entropy2)))

def getGainRatio(split1,split2, parent=None)->float:
    informationGain = getInformationGain(split1,split2,parent)
    splitEntropyClass = {'split1':split1['size'],'split2':split2['size']} 
    splitSize = split1['size']+split2['size']
    splitEntropy = getEntropy(splitEntropyClass, splitSize)
    return informationGain/splitEntropy

def getGiniIndex(split1,split2)->float:
    splitSize = split1['size']+split2['size']
    split1Gini = getGini(split1['classFrequency'], split1['size'])
    split2Gini = getGini(split2['classFrequency'], split2['size'])
    weight1 = split1['size']/splitSize
    weight2 = split2['size']/splitSize
    return (weight1*split1Gini)+(weight2*split2Gini)
    
def getMax(d:dict):
    maximum = max(d.values())
    key = ''
    for i in d:
        if(maximum==d[i]):
            key = i
            break
    return key

def checkLeaf(targetClasses:dict)->bool:
    c=0
    for x in targetClasses:
        c = c if targetClasses[x]==0 else c+1
    return True if c==1 else False

def createNode(nodeData:dict, parent=None, isLeft=True)->dict:
    node = dict()
    node['data'] = nodeData.copy()
    node['data']['classFrequency'] = nodeData['classFrequency'].copy()
    node['parentIndex'],node['index'] = parent,len(tree['nodeList'])
    node['isLeaf'] = checkLeaf(node['data']['classFrequency'])
    parentNode = tree['nodeList'][node['parentIndex']]
    if(isLeft):
        parentNode['childLIndex'] = node['index']
    else:
        parentNode['childRIndex'] = node['index']
    tree['nodeList'].append(node)
    return node

def getBestFeature(columnNames, rows, parent=None, gainType=0):
    columnGainDict = dict()
    columnValueDict = dict()
    columnDataDict =dict()
    for name in columnNames:
        currentColumn = rows[name]
        uniqueValues = currentColumn.unique()
        valueGainDict = dict()
        valueDataDict = dict()
        for value in uniqueValues:
            split1Col = currentColumn[currentColumn==value]
            split2Col = currentColumn[currentColumn!=value]
            cf1,cf2 = getFrequency(split1Col), getFrequency(split2Col) 
            size1, size2 = len(split1Col),  len(split2Col) 
            entropy1, entropy2 = getEntropy(cf1), getEntropy(cf2)
            index1, index2 = split1Col.index, split2Col.index
            split1,split2 = dict(), dict()
            split1['classFrequency'],split1['entropy'],split1['size']=cf1, entropy1, size1
            split1['index'] = index1
            split2['classFrequency'],split2['entropy'],split2['size']=cf2, entropy2, size2
            split2['index'] = index2
            gainFunc = [getInformationGain,getGainRatio,getGiniIndex]
            gain = gainFunc[gainType](split1,split2,parent) if gainType!=2 else 1 - gainFunc[gainType](split1,split2)
            valueGainDict.update({value:gain})
            valueDataDict.update({value:[split1,split2]})
        maxValKey = getMax(valueGainDict)
        columnValueDict.update({name:maxValKey})
        columnGainDict.update({name:valueGainDict[maxValKey]})
        columnDataDict.update({name:valueDataDict[maxValKey]})
    maxColKey = getMax(columnGainDict)
    maxColVal = columnValueDict[maxColKey]
    maxColData = columnDataDict[maxColKey]
    return maxColKey, maxColVal, maxColData

def createTargetClasses(trainData, trainTarget):
    tree['trainData'],tree['trainTarget']  = trainData, trainTarget
    for i in trainTarget.unique():
        tree['targetClasses'][i] =  0

def createRoot():
    trainTarget = tree['trainTarget']
    targetClasses = getFrequency()
    entropy = getEntropy(targetClasses)
    root = {
        'parentIndex':-1,
        'childLIndex':-1,
        'childRIndex':-1,
        'isLeaf':checkLeaf(targetClasses),
        'index':len(tree['nodeList']),
    }
    root['data'] = {
        'entropy' : entropy, 
        'classFrequency' : targetClasses, 
        'size' : len(trainTarget),
        'index' : trainTarget.index
    }
    tree['nodeList'].append(root)

def build(nodeIndex=0, gainType=0):
    node = tree['nodeList'][nodeIndex]
    rows = tree['trainData'][tree['trainData'].index.isin(tree['nodeList'][nodeIndex]['data']['index'])]
    if((not node['isLeaf'])):
        columnNames = tree['trainData'].columns
        bestColumn, bestColumnValue, bestColumnValueData = getBestFeature(columnNames, rows, nodeIndex, gainType=0)
        tree['nodeList'][nodeIndex]['childSplitCondition'] = (bestColumn,bestColumnValue)
        nodeL = createNode(bestColumnValueData[0], parent=nodeIndex)
        nodeR = createNode(bestColumnValueData[1], parent=nodeIndex, isLeft=False)
        build(nodeL['index'])
        build(nodeR['index'])

def fit(trainData, trainTarget):
    global tree 
    global treeList
    gainLabel = ['informationGain', 'gainRatio', 'giniIndex']
    treeList=dict()
    for i in range(0,3):
        tree = copy.deepcopy(TREESCHEMA)
        createTargetClasses(trainData, trainTarget)
        createRoot()
        build(gainType=i)
        treeList[gainLabel[i]] = copy.deepcopy(tree)

fit(train_x, train_y)
```
<P>
    The cell below classifies the test values for the three trees built in the cell above.
    <br>
    It does this in the following way:
    <br>
    <ol>
        <li>classify
            <ul>
                <li>The function classify has two parameters the test data and test target.</li>
                <li>It iterates three times, for each type of score calculation and classifiees the data for each iteration.</li>
                <li>During each iteration, it calls the traverse function to navigate the trees built in the previous node.</li>
            </ul>
        </li>
        <br>
        <li>traverse
            <ul>   
                <li>The traverse function calls itself recursively until it reaches a leaf node</li>
                <li>If the function reaches a leaf node, it classifes the rows present within that node</li>
                <li>For all other nodes, the function calls itself twice, once for every child node.</li>
            </ul>
        </li>
        <br>
    </ol>
</p>

```py
PREDICTIONTREESCHEMA ={
    'testData':pd.DataFrame,
    'testTarget':pd.Series,
    'predictClassIndex':dict()
}

predictionTree = None
predictionTreeList = None


def traverse(tree=None, nodeIndex=0, rowIndex=None):
    node = tree['nodeList'][nodeIndex]
    if((not node['isLeaf'])):
        rows = predictionTree['testData'][predictionTree['testData'].index.isin(rowIndex)]
        currentColumn = predictionTree['testData'][node['childSplitCondition'][0]]
        rowIndexL = rows[currentColumn==node['childSplitCondition'][1]].index
        rowIndexR = rows[currentColumn!=node['childSplitCondition'][1]].index
        traverse(nodeIndex = node['childLIndex'], tree=tree, rowIndex=rowIndexL)
        traverse(nodeIndex = node['childRIndex'], tree=tree, rowIndex=rowIndexR)
    else:
        classFrequency = node['data']['classFrequency']    
        for x in classFrequency:
            if(classFrequency[x]!=0):
                predictionTree['predictClassIndex'][x]=np.append(predictionTree['predictClassIndex'][x],rowIndex) 

def classify(testData, testTarget):
    global predictionTree 
    global predictionTreeList
    predictionTreeList=dict()
    for x in treeList:
        predictionTree = copy.deepcopy(PREDICTIONTREESCHEMA)
        predictionTree['testData'], predictionTree['testTarget'] = testData, testTarget
        tree = copy.deepcopy(treeList[x])
        for y in tree['targetClasses']:
            predictionTree['predictClassIndex'][y] = np.array([])
        traverse(tree=tree, rowIndex=predictionTree['testData'].index)   
        predictionTreeList.update({x:copy.deepcopy(predictionTree)})
        

classify(test_x,test_y)
```
<P>
    The cell below ranks the predicted values which were calculated in the previous cell.
    <br>
    It does this by getting the accuracy rate of each model.
</P>

```py
def getBestModel():
    score = dict()
    for i in predictionTreeList:
        predict = copy.deepcopy(predictionTreeList[i])
        predictedClassIndex = predict['predictClassIndex']
        predictedTarget = pd.Series(name='predicted')
        for j in predictedClassIndex:
            pSeries = pd.Series(np.full(len(predictedClassIndex[j]),j), 
                                    index=predictedClassIndex[j], 
                                    name='pSeries')
            predictedTarget = pd.concat([predictedTarget,pSeries])
        compareDF = pd.merge(predictedTarget.rename('predictedTarget'), test_y, left_index=True, right_index=True)
        scoreVal = sum(compareDF.apply(lambda x: 1 if x['status']==x['predictedTarget'] else 0 , axis=1)/len(compareDF))
        score.update({i:scoreVal})
    return score

getBestModel()
```