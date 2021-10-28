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
''' 
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
