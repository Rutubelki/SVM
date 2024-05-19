import sklearn
from sklearn import datasets
from sklearn import metrics
from sklearn import svm

dataset=datasets.load_digits()

x=dataset.data
y=dataset.target
print(x)
print(y)
clc=svm.SVC(kernel="linear",C=1);

x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
clc.fit(x_train,y_train)

acc=clc.predict(x_test)
ssore=metrics.accuracy_score(y_test,acc)
print(ssore)

name=[0,1,2,3,4,5,6,7,8,9]

for i in range(len(name)):
    print("predicted:",name[acc[i]],"Actual:",name[y_test[i]])