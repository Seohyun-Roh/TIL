import numpy as np

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
print(scores.mean(axis=1)) #1 -> 세로 평균