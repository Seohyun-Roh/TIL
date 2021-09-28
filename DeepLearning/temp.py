'''import numpy as np

x=np.array([[1,2],[3,4]])
y=np.array([[5,6],[7,8]])
print(np.concatenate((x,y),axis=1))
print(np.vstack((x,y)))

###

a=np.arange(12)
print(a.reshape(3,4))
print(a.reshape(6,-1)) #-1 => 12개로 6행. 열은 알아서

###

array=np.arange(30).reshape(-1,10)
print(array)
arr1,arr2=np.split(array,[3],axis=1) #[3]=> 몇개로 자를건지, axis 1이면 세로, 0이면 가로로
print(arr1)
print(arr2)

###

a=np.array([1,2,3,4,5,6])
print(a.shape)
a1=a[np.newaxis,:] #newaxis-> ,기준 앞에 있으면 가로를 기준으로 축 추가.
print(a1.shape)
a2=a[:,np.newaxis] #newaxis-> ,기준 뒤에 있으면 세로 축 추가.
print(a2.shape)

###자주쓰임!

ages=np.array([18,19,25,30,28])
print(ages[1:3]) #idx 1~2까지
print(ages[:2])
y=ages>20
print(ages[ages>20])

###

a=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a[0:2,1:3])

###

data=np.arange(16).reshape(4,-1)
print(data[0])
print(data[1,:])
print(data[:,2])
print(data[0:2,0:2])
print(data[0:2,2:4])

# ::2 -> 2개씩 건너뛰기. 1::2 -> 1다음 2개 건너뛰기
#numpy 곱셈 ( * )-> 행렬의 곱셈과는 다르다.;;
#골뱅이 (@) 쓰면 행렬 곱셈

scores=np.array([[99,93,60],[98,82,93],[93,65,81],[78,82,81]])
print(scores.mean(axis=0)) #0 -> 가로 평균
print(scores.mean(axis=1)) #1 -> 세로 평균

print(np.random.seed(100))
print(np.random.rand(5))
print(np.random.rand(5,3)) #5행 3열

###

print(np.random.randn(5)) #음수, 양수 섞어서 나옴

###

m, sigma = 10,2
print(m+sigma*np.random.randn(5))

###

mu, sigma = 0, 0.1
print(np.random.normal(mu, sigma, 5)) #평균, 표준편차

###

a=np.array([11,11,12,13,14,15,16,17,12,13,11,14,18,19,20])
unique_values = np.unique(a) #중복되는 값 제거
print(unique_values)

###

#2차원을 1차원으로 바꿀 때
x=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(x.flatten())

###

import pandas as pd

x=pd.read_csv('countries.csv', header=0).values
print(x)

df=pd.DataFrame(x)
print(df)

df.to_csv('test.csv')

###

import matplotlib.pyplot as plt
%matplotlib inline
X=["Mon", "Tue", "Wed", "Thur","Fri", "Sat","Sun"]
Y1=[15.6, 14.2, 16.3, 18.2, 17.1, 20.2, 22.4]
Y2=[20.1, 23.1, 23.8, 25.9, 23.4, 25.1, 26.3]

plt.plot(X, Y1, label="Seoul")
plt.plot(X, Y2, label="Busan")
plt.xlabel("day")
plt.ylabel("temperature")
plt.legend(loc="lower right") #범례를 어디다 놓을지
plt.title("Temperatures of Cities") #표이름
plt.show() #그래프가 그려짐

###

plt.plot(X, [15.6,14.2,16.3, 18.2, 17.1, 20.2, 22.4], "sb") #s->square mark 네모 표시자 m-> magenta, g->green
plt.show()

###

numbers=np.random.normal(size=10000)

plt.hist(numbers)
plt.xlabel("value")
plt.ylabel("freq")
plt.show()

###

def sigmoid(x):
    s=1/(1+np.exp(-x))
    ds=s*(1-s)
    return s, ds

X= np.linspace(-10, 10, 100)
Y1, Y2 = sigmoid(X)

plt.plot(X, Y1, label="Sigmoid")
plt.plot(X, Y2, label="Sigmoid'")
plt.xlabel("x")
plt.ylabel("Sigmoid(X), Sigmoid'(X)")
plt.legend(loc="upper left")
plt.show()'''

'''
#1번 -> [2, 4, 6]
print("1번 -> [2, 4, 6]\n")

#2번 -> [[2,4,6], [5,7,9]]
print("2번 -> [[2,4,6], [5,7,9]]\n")

#3번
print("3번")
import numpy as np
arr=np.zeros(10)
arr[4]=1
print(arr)

#4번
print("\n4번")
arr=np.arange(10,20)
print(arr)

#5번
print("\n5번")
arr=np.arange(10)
arr=arr[::-1]
print(arr)

#6번
print("\n6번")
arr=np.arange(9)
arr=arr.reshape(3,3)
print(arr)

#7번
print("\n7번")
arr=np.random.rand(3,3)
print(arr)

#8번
print("\n8번")
arr=np.random.rand(10,10)
min=arr.min()
max=arr.max()
print("최솟값 =", min, "최대값 =", max)

#9번
print("\n9번")
arr=np.ones(9)
arr=arr.reshape(3,3)
arr[1:-1,1:-1]=0
print(arr)

#10번
print("\n10번")
arr=np.zeros(25)
arr=arr.reshape(5,5)
arr[0::2, 1::2]=1
arr[1::2, ::2]=1
print(arr)

#11번
print("\n11번")
arr=np.random.rand(3,3)
mean = np.mean(arr)
std = np.std(arr)
res = (arr-mean)/std
print(res)

#12번
print("\n12번")
arr=np.arange(10)
arr[5:9]=arr[5:9]*-1
print(arr)

#13번
print("\n13번")
arr = np.arange(0,9).reshape(3,3)
print("원본 배열:\n",arr)
sum=arr.sum()
print("모든 요소의 합:", sum)
row = arr.sum(axis=0)
print("각 열의 합:",row);
col = arr.sum(axis=1)
print("각 행의 합:",col)

#14번
print("\n14번")
x = [4,5]
y = [7, 10]
print("원본 벡터 :\n",x,"\n",y)
dot = np.dot(x,y)
print("벡터의 내적:",dot)

#15번
print("\n15번")
import matplotlib.pyplot as plt
%matplotlib inline
Y = [2, 0, 3, 6, 4, 6, 8, 12, 10, 9, 18, 20, 22]
plt.plot(Y)
plt.show()
'''

###

#경사하강법 구현
'''
X = np.array([0.0, 1.0, 2.0])
y = np.array([3.0, 3.5, 5.5])

W = 0
b = 0

lrate = 0.01
epochs = 1000

n = float(len(X))

for i in range(epochs):
    y_pred = W*X + b
    dW = (2/n) * sum(X* (y_pred-y))
    db = (2/n) * sum(y_pred-y)
    W = W - lrate * dW
    b = b- lrate * db

print(W, b)

y_pred = W*X + b

plt.scatter(X, y)

plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)], color='red')
plt.show()

###

#선형 회귀 예제

from sklearn import linear_model

reg = linear_model.LinearRegression()

X = [[0],[1],[2]]
y = [3, 3.5, 5.5]

reg.fit(X, y) #학습

print(reg.coef_) #직선의 기울기
print(reg.intercept_) #직선의 y절편
print(reg.score(X, y))
print(reg.predict([[5]]))

plt.scatter(X, y, color='black')

y_pred = reg.predict(X)

plt.plot(X, y_pred, color='blue', linewidth=3)
plt.show()

###

#선형 회귀 실습

reg = linear_model.LinearRegression()

X = [[174],[152],[138],[128],[186]]
y = [71, 55, 46, 38, 88]

reg.fit(X, y) #학습

print(reg.predict([[165]]))

plt.scatter(X, y, color='black')

y_pred = reg.predict(X)

plt.plot(X, y_pred, color='pink', linewidth=3)
plt.show() '''

###

#선형 회귀 당뇨병 예제

import matplotlib.pylab as plt
from sklearn import linear_model
from sklearn import datasets

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

diabetes_X_new = diabetes_X[:, np.newaxis, 2]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes_X_new, diabetes_y, test_size=0.1, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

plt.plot(X_test, y_pred, '.')

plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.show()
