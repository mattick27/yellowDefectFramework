import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib 

tolList=[1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4]
CList=[1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4]

def buildModel(x_Train,y_Train,model='All'):
    modelList = {'0': LinearSVC , '1' : LogisticRegression}    
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
            table[idx].append(
                {
                'confusion' : np.array(confusion),
                'accuracy' : acc
                }
            )
    return np.array(table)

def reportTheReport(data,mode='confusion'):
    print(data.shape)
    if mode == 'confusion':
        for unpackData in data : 
            for IteData in unpackData:
                print(IteData['confusion'])

iris = datasets.load_iris()
x = iris.data[:, :2]  # we only take the first two features.
y = iris.target


arr = buildModel(x,y,'All')
# print(arr.shape)
data = validateModel(arr,x,y)
# print(data)
# data = [{'confusion' : np.array([[10,20],[20,10],[2,5]]),'accuracy' : 78.21},
#         {'confusion' : np.array([[10,50],[2,21],[1,3]]),'accuracy' : 63.11}]
reportTheReport(data = data)

