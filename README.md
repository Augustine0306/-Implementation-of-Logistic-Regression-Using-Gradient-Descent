# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: 212222240015
RegisterNumber:  AUGUSTINE J
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt('/content/ex2data1 (1).txt',delimiter=',')
X = data[:,[0,1]]
y = data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:,0],X[y == 1][:,1],label="Admitted",color="red")
plt.scatter(X[y == 0][:,0],X[y == 0][:,1],label="Not Admitted",color='black')
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1 + np.exp(-z))

  plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot),color="red")
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J= -(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad = np.dot(X.T,h-y)/X.shape[0]
  return J,grad

  X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
J,grad = costFunction(theta,X_train,y)
print(J)
print(grad)

X_train= np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([-24, 0.2, 0.2])
J,grad = costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  J = -(np.dot(y,np.log(h))+ np.dot(1-y, np.log(1-h)))/ X.shape[0]
  return J

  def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad = np.dot(X.T,h-y)/X.shape[0]
  return grad

  X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(X_train,y),
                        method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max = X[:,0].min() - 1, X[:,0].max()+1
  y_min,y_max = X[:,1].min() - 1, X[:,0].max()+1
  xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot = np.c_[xx.ravel(),yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted",color="red")
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted",color="black")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1score")
  plt.ylabel("Exam 2score")
  plt.legend()
  plt.show()

  plotDecisionBoundary(res.x ,X ,y)

  prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

  np.mean(predict(res.x,X)==y)
```
## Output:
## Array Value of X:
![image](https://github.com/Augustine0306/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404460/71b9e1f8-ec52-4544-9f15-ee17a862f531)
## Array Value of Y:
![image](https://github.com/Augustine0306/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404460/433ca950-d05d-4f36-a3b3-bd446f7e1373)
## Exam Score Graph:
![image](https://github.com/Augustine0306/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404460/1ef1be44-b431-4f8e-8a05-da638a575f2d)
## Sigmoid function graph:
![image](https://github.com/Augustine0306/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404460/f1782fd5-084a-4795-be61-3f2f77ea3619)
## X_train_grad value:
![image](https://github.com/Augustine0306/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404460/069c561a-b28f-43eb-b24b-c080627ff02b)
## Y_train_grad value:
![image](https://github.com/Augustine0306/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404460/c893af06-3071-4833-b4b4-052db09b90e0)
## Print res.x:
![image](https://github.com/Augustine0306/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404460/dbf7eaeb-3c22-4e6f-8ffd-d5a2a1894342)
## Decision boundary - graph for exam score:
![image](https://github.com/Augustine0306/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404460/13de0781-1cc5-4ce1-90c3-8f5519846955)
## Proability value:
![image](https://github.com/Augustine0306/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404460/cae60a2c-e08c-4c73-ad8a-fe7ad99b0b77)
## Prediction value of mean:
![image](https://github.com/Augustine0306/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119404460/90ddb773-3401-4150-aee9-74a75de9b5ca)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

