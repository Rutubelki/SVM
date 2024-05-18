import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")


data = data[["G1","G2","G3","absences","studytime","failures","health"]]

predict = "G3"
x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

#linear regression
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
'''
best=0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear= linear_model.LinearRegression()
    linear.fit(x_train,y_train)
    acc=linear.score(x_test,y_test)
    print(acc)

    if acc>best:
        best=acc
        #pickle the object to a file
        with open("studentmodel.pickle","wb") as f:
            pickle.dump(linear,f)'''
#unpickle the object from the file
pickle_in=open("studentmodel.pickle", "rb")
linear=pickle.load(pickle_in)

print("Co:",linear.coef_)
print("Intercept:",linear.intercept_)

#comparing the x_test with y_test after training the model
predictions= linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])

#Saving models and visualize data
#saving the model is  done through pickle module
p='G1'
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final grade")
pyplot.show()

