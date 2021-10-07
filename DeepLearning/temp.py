import numpy as np

'''x=np.array([[1,2],[3,4]])
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

df.to_csv('test.csv')'''

###

import matplotlib.pyplot as plt
#%matplotlib inline

'''X=["Mon", "Tue", "Wed", "Thur","Fri", "Sat","Sun"]
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
plt.show() 

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
plt.show() '''

###

'''
#회귀 분석 과제 1번
from sklearn import linear_model

reg1 = linear_model.LinearRegression()

X = [[2015], [2016], [2017], [2018], [2019]]
y = [12, 19, 28, 37, 46]

reg1.fit(X, y)

print("m:",reg1.coef_)
print("b:",reg1.intercept_)
print("2020 매출:",reg1.predict([[2020]]))

plt.scatter(X, y, color='black')

y_pred = reg1.predict(X)

plt.plot(X, y_pred, color='pink', linewidth=3)
plt.show() 


#회귀 분석 과제 2번
reg2 = linear_model.LinearRegression()
X = [[1930], [1940], [1950], [1960], [1970], [1980], [1990], [2010], [2016]]
y = [59, 62, 70, 69, 71, 74, 75, 76, 78]

reg2.fit(X,y)

print("m:",reg2.coef_)
print("b:",reg2.intercept_)
print("1962년 기대 수명:",reg2.predict([[1962]]))

plt.scatter(X, y, color='black')

y_pred = reg2.predict(X)

plt.plot(X, y_pred, color='blue', linewidth=3)
plt.show() 


#회귀 분석 과제 3번
reg3 = linear_model.LinearRegression()
X = [[30], [35], [20], [15], [3]]
y = [90, 95, 70, 40, 10]

reg3.fit(X, y)

print("직선의 기울기:",reg3.coef_)
print("직선의 y절편",reg3.intercept_)

plt.scatter(X, y, color='black')

y_pred = reg3.predict(X)

plt.plot(X, y_pred, color='purple', linewidth=3)
plt.show()
'''

###

#p.185 Mini Project: 퍼셉트론으로 분류
'''
import numpy as np

epsilon = 0.0000001

def step_func(t):
    if t > epsilon: return 1
    else: return 0

X = np.array([
    [160, 55, 1],
    [163, 43, 1],
    [165, 48, 1],
    [170, 80, 1],
    [175, 76, 1],
    [180, 70, 1]
])

y = np.array([0, 0, 0, 1, 1, 1])
W = np.zeros(len(X[0]))

def perceptron_fit(X, Y, epochs=20):
    global W
    eta = 0.2

    for t in range(epochs):
        print("epoch=", t, "======================")
        for i in range(len(X)):
            predict = step_func(np.dot(X[i], W))
            error = Y[i] - predict
            W += eta * error * X[i]
            print("현재 처리 입력=",X[i],"정답=",Y[i],"출력=",predict,"변경된 가중치=", W)
        print("================================")

def perceptron_predict(X, Y):
    global W
    for x in X:
         print(x[0], x[1], "->", step_func(np.dot(x, W)))

perceptron_fit(X, y, 20)
perceptron_predict(X, y)
'''

###

#5장 9번 문제
#퍼셉트론으로 아이리스 데이터 분류하기
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from matplotlib import pyplot as plt

epsilon = 0.0000001

def step_func(t):
    if t > epsilon: return 1
    else: return 0

iris = load_iris()
X = iris.data[:, (0, 1)] #꽃의 너비와 높이만을 입력으로 함
y = (iris.target == 0).astype(np.int64) #출력: "Iris Setosa인가 아닌가"

percep = Perceptron(random_state=32)
percep.fit(X, y)
print(percep.score(X, y))
print(percep.predict(X))

#plt.scatter(X[:,0], X[:,1], c=y, s=100)
#plt.xlabel("width")
#plt.ylabel("height")

#plt.show()

#-------#

#표준편차 이용(위의 방법보다 더 정확성이 높음)
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
# Load the iris dataset
iris = datasets.load_iris()

# Create our X and y data
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#ppn = Perceptron(tol=1e-3, eta0=0.2, random_state=0)
ppn = Perceptron(max_iter=40, eta0=0.1, tol=1e-3, random_state=1) #max_iter->epochs
# Train the perceptron
ppn.fit(X_train, y_train)


y_pred = ppn.predict(X_test)

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

print("====================")


sc = StandardScaler() #스케일링 : 전체 자료의 분포를 평균 0 표준편차 1이 되도록 변환
sc.fit(X_train) #X_train의 평균과 표준편차를 구한다.
X_train_std = sc.transform(X_train) #학습용 데이터를 입력으로 하여 transform 실행시 학습용 데이터를 표준화한다.
X_test_std = sc.transform(X_test) #마찬가지로 테스트 데이터 표준화

ml = Perceptron(n_iter_no_change=40, eta0=0.1, random_state=0)#eta0 : learning rate, n_iter : epochs over data
ml.fit(X_train_std, y_train)
y_pred = ml.predict(X_test_std)
print("총 테스트 개수:%d, 오류개수:%d" %(len(y_test), (y_test != y_pred).sum()))
print("정확도: %.2f" %accuracy_score(y_test, y_pred)) #accuracy_score() : y_test와 y_pred를 비교하여 정확도를 계산해줌
'''

###

#breast_cancer Perceptron으로 분류하기
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the breast cancer dataset
breast_cancer = datasets.load_breast_cancer()

# Create our X and y data
X = breast_cancer.data
y = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#ppn = Perceptron(tol=1e-3, eta0=0.2, random_state=0)
ppn = Perceptron(max_iter=40, eta0=0.1, tol=1e-3, random_state=1) #max_iter->epochs
# Train the perceptron
ppn.fit(X_train, y_train)


y_pred = ppn.predict(X_test)

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

print("====================")


sc = StandardScaler() #스케일링 : 전체 자료의 분포를 평균 0 표준편차 1이 되도록 변환
sc.fit(X_train) #X_train의 평균과 표준편차를 구한다.
X_train_std = sc.transform(X_train) #학습용 데이터를 입력으로 하여 transform 실행시 학습용 데이터를 표준화한다.
X_test_std = sc.transform(X_test) #마찬가지로 테스트 데이터 표준화

ml = Perceptron(n_iter_no_change=40, eta0=0.1, random_state=0)#eta0 : learning rate, n_iter : epochs over data
ml.fit(X_train_std, y_train)
y_pred = ml.predict(X_test_std)
print("총 테스트 개수:%d, 오류개수:%d" %(len(y_test), (y_test != y_pred).sum()))
print("정확도: %.2f" %accuracy_score(y_test, y_pred)) #accuracy_score() : y_test와 y_pred를 비교하여 정확도를 계산해줌
