'''
# 6장 연습문제 2번
import matplotlib.pyplot as plt
import numpy as np

x = 10  
learning_rate = 0.01
precision = 0.00001  
max_iterations = 2000

# 손실함수를 람다식으로 정의한다. 
loss_func = lambda x: x**2-6*x+4
# 그래디언트를 람다식으로 정의한다. 손실함수의 1차 미분값이다. 
gradient = lambda x: 2*x-6

X = np.arange(0.0, 10.0, 0.01)
plt.plot(X, loss_func(X),'b-')

x0 = 10
plt.scatter(x0, loss_func(x0), c='orange')

# 그래디언트 강하법
for i in range(max_iterations):
    x = x - learning_rate * gradient(x)
    plt.scatter(x, loss_func(x), c='orange')
    print("손실함수값(", x, ")=", loss_func(x))

plt.show()
print("최소값 = ", x)'''

###

# 6장 연습문제 6번
import numpy as np

# 시그모이드 함수
#def actf(x):
#	return 1/(1+np.exp(-x))


# ReLU 함수
def actf(x):
    return np.maximum(0, x)

# tanH 함수
#def actf(x):
#    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

# 입력유닛의 개수, 은닉유닛의 개수, 출력유닛의 개수
inputs, hiddens, outputs = 2, 2, 1
learning_rate=0.2

# 훈련 샘플과 정답 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
T = np.array([[0], [1], [1], [0]])
W1 = np.array([[0.10, 0.20],
                 [0.30, 0.40]])
W2 = np.array([[0.50], [0.60]])
B1 = np.array([0.1, 0.2])
B2 = np.array([0.3])   

# 순방향 전파 계산
def predict(x):
        layer0 = x			# 입력을 layer0에 대입한다. 
        Z1 = np.dot(layer0, W1)+B1	# 행렬의 곱을 계산한다. 
        layer1 = actf(Z1)		# 활성화 함수를 적용한다. 
        Z2 = np.dot(layer1, W2)+B2	# 행렬의 곱을 계산한다. 
        layer2 = actf(Z2)		# 활성화 함수를 적용한다. 
        return layer0, layer1, layer2
def test():
    for x, y in zip(X, T):
        x = np.reshape(x, (1, -1))	# x를 2차원 행렬로 만든다.입력은 2차원이어야 한다.
        layer0, layer1, layer2 = predict(x)
        print(x, y, layer2)
test() 
