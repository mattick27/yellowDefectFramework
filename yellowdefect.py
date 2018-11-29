import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import matplotlib 
import pandas as pd 
tolList=[1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4]
CList=[1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4]
modelList = {'0': LinearSVC , '1' : LogisticRegression}    
def buildModel(x_Train,y_Train,model='All'):
    if model == 'All':
        arrList = []
        for idx in range(2) : 
            print("Model's order is {}".format(modelList[str(idx)].__name__))
            arr = []
            for tol in tolList :
                for C in CList:
                    clf = modelList[str(idx)](tol=tol,C=C,dual=False)
                    clf = clf.fit(x_Train,y_Train)
                    arr.append(clf)
                    clf = 0
            arrList.append(np.array(arr))
        return np.array(arrList) 
    elif model == 'Logistic':
        pass
    elif model == 'SVC':
        pass

def validateModel(modelList,x_validation,y_validation):
    table = [[],[]]
    for idx,model in enumerate(modelList) :
        for i in model :
            pred = i.predict(x_validation)
            confusion = confusion_matrix(pred,y_validation)
            acc = accuracy_score(pred,y_validation)
            preRe = precision_recall_fscore_support(pred,y_validation,average=None)
            table[idx].append(
                {
                'confusion' : np.array(confusion),
                'accuracy' : acc,
                'precisionRecall' : np.array(preRe[0:2])
                }
            )
    return np.array(table)

def reportTheReport(data,mode='overview'):
    print(data.shape)
    if mode == 'overview':
        dictList = {'Model' : []}
        for idxModel,unpackData in enumerate(data) : 
            for IteData in unpackData:
                preData = np.concatenate((IteData['precisionRecall'],IteData['accuracy']),axis=None)
                numIndex = 1 
                for idx,i in enumerate(preData):
                    if idx == len(preData) -1 :
                        try :
                            dictList['accuracy'].append(IteData['accuracy'])
                        except:
                            dictList['accuracy'] = [IteData['accuracy']]
                    elif idx%2 == 0:
                        try :
                            dictList['precision{}'.format(numIndex)].append(i)
                        except:
                            dictList['precision{}'.format(numIndex)] = [i]
                    else : 
                        try :
                            dictList['recall{}'.format(numIndex)].append(i)
                            numIndex+= 1 
                        except:
                            dictList['recall{}'.format(numIndex)] = [i]
                            numIndex+= 1 
                dictList['Model'].append(modelList[str(idxModel)].__name__)
        df = pd.DataFrame(dictList)
        writer = pd.ExcelWriter("test.xlsx")
        df.to_excel(writer)
    if mode == 'confusion':
        for unpackData in data : 
            for IteData in unpackData:         
                print('--------------------')       
                print(IteData['confusion'])
                print('--------------------')
                pass


iris = datasets.load_iris()
x = iris.data[:,:]  # we only take the first two features.
y = iris.target

arr = buildModel(x,y,'All')
data = validateModel(arr,x,y)
reportTheReport(data = data)

