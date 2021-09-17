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
print(ages[ages>20])'

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

numpy 곱셈 ( * )-> 행렬의 곱셈과는 다르다.;;
골뱅이 (@) 쓰면 행렬 곱셈

scores=np.array([[99,93,60],[98,82,93],[93,65,81],[78,82,81]])
print(scores.mean(axis=0)) #0 -> 가로 평균
print(scores.mean(axis=1)) #1 -> 세로 평균 '''

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

import matplotlib.pyplot as plt
%matplotlib inline

#X=["Mon", "Tue", "Wed", "Thur","Fri", "Sat","Sun"]

'''Y1=[15.6, 14.2, 16.3, 18.2, 17.1, 20.2, 22.4]
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
plt.show()