# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: N.NAVYA SREE
RegisterNumber: 212223040138 
*/
```

import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

## Output:


![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/6336b502-e02e-49eb-a134-4f619ed25048)

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/18dd61a4-ecce-4b9a-9be4-fa6e74cd09c1)

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/530c7485-cabd-4e5a-b3ed-ea4f67187790)

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/6ebcd87f-8867-4595-bbd3-dabc53ed6c29)

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/acd4e1dd-228b-4d7d-94d0-f2dbad0ef92c)

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/a057710b-ec40-454b-bedd-0597ee2cb744)

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/a5996e96-bc44-4cfe-8486-ee100d27b9ab)

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/d95b185e-58f3-44b6-a39b-1fa2ebff2ca1)

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/615399fb-41ce-4844-b9ca-303bbee3a9da)

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/6a6eb1e8-de54-43c9-9add-ff565737ab47)

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/be817ccb-c4d0-4c69-b289-289be27d2973)

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/620373e3-7333-4513-8551-84a094fa975c)

![image](https://github.com/23004513/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138973069/64118a8e-0e1f-4cb0-8869-70bd499376d8)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
