import sklearn
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import datasets

cancer=datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)

#clg=KNeighborsClassifier(n_neighbors=7)
clg=svm.SVC(kernel='linear',C=1)
x=cancer.data
y=cancer.target

x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
clg.fit(x_train,y_train)
y_predict=clg.predict(x_test)
acc=metrics.accuracy_score(y_test,y_predict)
print(acc)

name = ['malignant','Benign']
for x in range(len(y_predict)):
    print("predicted:",name[y_predict[x]],"actual :",name[y_test[x]])
