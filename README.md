# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function. 
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Manjupriya P
RegisterNumber: 212220220024 
*/
```
import pandas as pd

data = pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])

data.head()

x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()

y = data["left"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion = "entropy")

dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)

from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)

accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

## Output:
1.Data Head

![image](https://github.com/Manjupriya1207/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113583090/82d1cd03-a39b-4b1a-96d0-266a83a476b8)

2.Data Info

![image](https://github.com/Manjupriya1207/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113583090/4ce0177c-b938-4bbc-81f4-d1f75fbf9eb1)

3.Data isnull

![image](https://github.com/Manjupriya1207/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113583090/8f18cda6-64c1-4977-8d6a-0bf822ac7f9c)

4.Data Left

![image](https://github.com/Manjupriya1207/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113583090/89bcfc2b-2723-4bf8-bfd5-3acef01c78f8)

5.X Head

![image](https://github.com/Manjupriya1207/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113583090/d9625800-c5e5-4ca8-9db3-187c297b796b)

6.Data fit

![image](https://github.com/Manjupriya1207/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113583090/14e78dfb-b2d0-4945-9511-f9691b20a7f3)

7.Accuracy

![image](https://github.com/Manjupriya1207/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113583090/9ed32075-28bf-4f69-9b7b-ce87d4f786e5)

8.Predicted Values

![image](https://github.com/Manjupriya1207/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113583090/db39fddf-92f8-432e-aa51-ac2e2aa340ed)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
