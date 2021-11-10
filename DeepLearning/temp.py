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
'''
import matplotlib.pyplot as plt
#%matplotlib inline

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
print("정확도: %.2f" %accuracy_score(y_test, y_pred)) #accuracy_score() : y_test와 y_pred를 비교하여 정확도를 계산해줌'''

###
'''
# p.201 MLP 순방향 패스
import numpy as np

def actf(x):
    return 1/(1+np.exp(-x))

def actf_deriv(x):
    return x*(1-x)

inputs, hiddens, outputs = 2, 2, 1
learning_rate=0.2

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
T = np.array([[0], [1], [1], [0]])

W1 = np.array([[0.10, 0.20], [0.30, 0.40]])
W2 = np.array([[0.50], [0.60]])
B1 = np.array([0.1, 0.2])
B2 = np.array([0.3])

def predict(x):
    layer0 = x
    Z1 = np.dot(layer0, W1)+B1
    layer1 = actf(Z1)
    Z2 = np.dot(layer1, W2)+B1
    layer2 = actf(Z2)
    return layer0, layer1, layer2

def test():
    for x, y in zip(X, T):
        x = np.reshape(x, (1, -1))
        layer0, layer1, layer2 = predict(x)
        print(x, y, layer2)

test()

###

# p.211 경사 하강법 실습

x = 10
learning_rate = 0.2
precision = 0.00001
max_iterations = 100

loss_func = lambda x: (x-3)**2+10

gradient = lambda x: 2*x-6

for i in range(max_iterations):
    x = x - learning_rate * gradient(x)
    print("손실함수값(", x, ")=", loss_func(x))
    
print("최소값 = ", x)

###

# p.213 2차원 그래디언트 시각화

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.5)
y = np.arange(-5, 5, 0.5)
X, Y = np.meshgrid(x, y)
Z = X**2+Y**2

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z)
plt.show()

#

x = np.arange(-5, 5, 0.5)
y = np.arange(-5, 5, 0.5)
X, Y = np.meshgrid(x, y)
U = -2*X
V = -2*Y

plt.figure()
Q = plt.quiver(X, Y, U, V, units = 'width')
plt.show()

###

# p.224 넘파이만을 이용한 MLP 구현

import numpy as np
def actf(x):
    return 1/(1+np.exp(-x))

def actf_deriv(x):
    return x*(1-x)

inputs, hiddens, outputs = 2, 2, 1
learning_rate=0.2

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
T = np.array([[1], [0], [0], [1]])

W1 = 2*np.random.random((inputs, hiddens))-1
W2 = 2*np.random.random((hiddens, outputs))-1
B1 = np.zeros(hiddens)
B2 = np.zeros(outputs)

# 순방향 전파 계산
def predict(x):
    layer0 = x
    Z1 = np.dot(layer0, W1)+B1
    layer1 = actf(Z1)
    Z2 = np.dot(layer1, W2)+B2
    layer2 = actf(Z2)
    # 00 01 10 11
    # 
    return layer0, layer1, layer2

# 역방향 전파 계산
def fit():
         global W1, W2, B1, B2
         for i in range(90000):
             for x, y in zip(X, T):
                 x = np.reshape(x, (1, -1))
                 y = np.reshape(y, (1, -1))
                 
                 layer0, layer1, layer2 = predict(x)  #순방향 계산
                 layer2_error = layer2-y
                 layer2_delta = layer2_error*actf_deriv(layer2)
                 layer1_error = np.dot(layer2_delta, W2.T)
                 layer1_delta = layer1_error*actf_deriv(layer1)
                 
                 W2 += -learning_rate*np.dot(layer1.T, layer2_delta)
                 W1 += -learning_rate*np.dot(layer0.T, layer1_delta)
                 B2 += -learning_rate*np.sum(layer2_delta, axis=0)
                 B1 += -learning_rate*np.sum(layer1_delta, axis=0)
             
def test():
    for x, y in zip(X, T):
        x = np.reshape(x, (1, -1))
        layer0, layer1, layer2 = predict(x)
        print(x, y, layer2)
        
fit()
test()
'''

###
'''
# p.246 미니배치 실습

import numpy as np
import tensorflow as tf

# 데이터를 학습 데이터와 테스트 데이터로 나눈다. 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

data_size = x_train.shape[0]
batch_size = 12	# 배치 크기

selected = np.random.choice(data_size, batch_size)
print("p.246 미니배치 실습")
print(selected)
x_batch = x_train[selected]
y_batch = y_train[selected]

###

# p.250 미니배치 구현
import numpy as np

def actf(x):
    return 1/(1+np.exp(-x))

def actf_deriv(x):
    return x*(1-x)

inputs, hiddens, outputs = 2, 2, 1
learning_rate = 0.5

X = np.array([[0,0], [0,1], [1,0], [1,1]])
T = np.array([[0], [1], [1], [0]])

W1 = 2*np.random.random((inputs, hiddens))-1
W2 = 2*np.random.random((hiddens, outputs))-1
B1 = np.zeros(hiddens)
B2 = np.zeros(outputs)

# 순방향 전파 계산
def predict(x):
    layer0 = x
    Z1 = np.dot(layer0, W1)+B1
    layer1 = actf(Z1)
    Z2 = np.dot(layer1, W2)+B2
    layer2 = actf(Z2)
    return layer0, layer1, layer2

# 역방향 전파 계산
def fit():
    global W1, W2, B1, B2
    for i in range(60000):
        layer0, layer1, layer2 = predict(X)
        layer2_error = layer2-T
        
        layer2_delta = layer2_error*actf_deriv(layer2)
        layer1_error = np.dot(layer2_delta, W2.T)
        layer1_delta = layer1_error*actf_deriv(layer1)
        
        W2 += -learning_rate*np.dot(layer1.T, layer2_delta)/4.0
        W1 += -learning_rate*np.dot(layer0.T, layer1_delta)/4.0
        B2 += -learning_rate*np.sum(layer2_delta, axis = 0)/4.0
        B1 += -learning_rate*np.sum(layer1_delta, axis = 0)/4.0
        
def test():
    for x, y in zip(X, T):
        x = np.reshape(x, (1, -1))
        layer0, layer1, layer2 = predict(x)
        print(x, y, layer2)
        
print("\np.250 미니배치 구현")        
fit()
test()

###

# MNIST 숫자 인식 케라스를 이용해서 정확률 확인
import matplotlib.pyplot as plt
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

train_images = train_images.reshape((60000, 784))
train_images = train_images.astype('float32') / 255.0

test_images = test_images.reshape((10000, 784))
test_images = test_images.astype('float32') / 255.0

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

model.fit(train_images, train_labels, epochs=5, batch_size=50)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('테스트 정확도:', test_acc)

import cv2 as cv

image = cv.imread('test.png', cv.IMREAD_GRAYSCALE)
image = image.astype('float32')
image = image.reshape(1, 784)
image = 255-image
image /= 255.0

plt.imshow(image.reshape(28, 28),cmap='Greys')
plt.show()

pred = model.predict(image.reshape(1, 784), batch_size=1)
print("추정된 숫자=", pred.argmax())

###

# 그림판에 임의 숫자 그린 후 인식되는지 확인
# a) 은닉층 유닛 개수 조절 후 정확률 확인 (2배 or 1/2배로)
# x2 0.97 -> 0.98 x0.5 0.979 -> 0.978
# b) 배치 크기 조절 후 정확률 확인
# 0.979 -> 0.98
# c) relu -> 시그모이드로 바꾸고 정확률 확인
# 연습문제 9번(레이어 추가) 10, 11, 12, 13, 14
'''

###

# p.287 그리드 검색 예제
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')/255

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

def build_model():
    network = tf.keras.models.Sequential()
    network = tf.keras.models.Sequential()
    network.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(28*28,)))
    network.add(tf.keras.layers.Dense(10, activation='sigmoid'))
    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return network

# 하이퍼 매개변수 딕셔너리
param_grid = {
              'epochs':[1, 2, 3],	# 에포크 수: 1, 2, 3
              'batch_size':[32, 64]	# 배치 크기: 32, 64
             }

model = KerasClassifier(build_fn = build_model, verbose=1)

# 그리드 검색
gs = GridSearchCV(
    estimator=model,
    param_grid=param_grid, 
    cv=3, 
    n_jobs=-1 
)

# 그리드 검색 결과 출력
grid_result = gs.fit(train_images, train_labels)

print(grid_result.best_score_)
print(grid_result.best_params_)
'''
###

# p.326 과잉 적합
'''
import numpy as numpy
import tensorflow as tf
import matplotlib.pyplot as plt

# 데이터 다운로드
(train_data, train_labels), (test_data, test_labels) = \
    tf.keras.datasets.imdb.load_data( num_words=1000)

# 원-핫 인코딩으로 변환하는 함수
def one_hot_sequences(sequences, dimension=1000):
    results = numpy.zeros((len(sequences), dimension))
    for i, word_index in enumerate(sequences):
        results[i, word_index] = 1.
    return results

train_data = one_hot_sequences(train_data)
test_data = one_hot_sequences(test_data)

# 신경망 모델 구축
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(1000,)))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

# 신경망 훈련, 검증 데이터 전달
history = model.fit(train_data,
                    train_labels,
                    epochs=20,
                    batch_size=512,
                    validation_data=(test_data, test_labels),
                    verbose=2)

# 훈련 데이터의 손실값과 검증 데이터의 손실값을 그래프에 출력
history_dict = history.history
loss_values = history_dict['loss']		# 훈련 데이터 손실값
val_loss_values = history_dict['val_loss']	# 검증 데이터 손실값
acc = history_dict['accuracy']			# 정확도
epochs = range(1, len(acc) + 1)		# 에포크 수

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Plot')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train error', 'val error'], loc='upper left')
plt.show()
'''

###

# p.335 MNIST 필기체 인식
'''
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
'''

###

# p.336 예제: 패션아이템 분류
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

plt.imshow(train_images[0])

train_images = train_images / 255.0
test_images = test_images / 255.0

model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('정확도:', test_acc)
'''
###

# p.340 예제: 타이타닉 생존자 예측하기

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


train = pd.read_csv("train.csv", sep=',')
test = pd.read_csv("test.csv", sep=',')

train.drop(['SibSp', 'Parch', 'Ticket', 'Embarked', 'Name',\
        'Cabin', 'PassengerId', 'Fare', 'Age'], inplace=True, axis=1)

train.dropna(inplace=True)


for ix in train.index:
    if train.loc[ix, 'Sex']=="male":
       train.loc[ix, 'Sex']=1 
    else:
       train.loc[ix, 'Sex']=0 

target = np.ravel(train.Survived) 

train.drop(['Survived'], inplace=True, axis=1)
train = train.astype(float)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train, target, epochs=30, batch_size=1, verbose=1)
