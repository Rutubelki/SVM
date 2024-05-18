import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing,linear_model

data=pd.read_csv("data.csv",sep=";")
print(data.head())

le=preprocessing.LabelEncoder()
#attendance=le.fit_transform(list(data['"Daytime/evening attendance	"']))
pgarde=le.fit_transform(list(data["Educational special needs"]))
f_grade=le.fit_transform(list(data["Debtor"]))
#s_grade=le.fit_transform(list(data["Tuition fees up to date"]))
course=le.fit_transform(list(data["Gender"]))
#fees=le.fit_transform(list(data["Scholarship holder"]))
target=le.fit_transform(list(data["Target"]))

predict="target"
x=list(zip("pgrage","f_grade","s_garde","course","fees","target"))
y=list(target)

x_test,x_train,y_test,y_train=sklearn.model_selection.train_test_split(x,y,test_size=0.1)

model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
acc=model.score(x_test)
print(acc)
