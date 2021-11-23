# 예제: 스팸 메일 분류하기

import numpy as np
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

docs = ['additional income',
		'best price',
		'big bucks',
		'cash bonus',
		'earn extra cash',
		'spring savings certificate',
		'valero gas marketing',
		'all domestic employees',
		'nominations for oct',
		'confirmation from spinner']

labels = np.array([1,1,1,1,1,0,0,0,0,0])

vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)

max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(padded_docs, labels, epochs=50, verbose=0)

loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('정확도=', accuracy)

test_doc = ['big income']
encoded_docs = [one_hot(d, vocab_size) for d in test_doc]
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

print(model.predict(padded_docs))

###
'''
# 예제: 다음 단어 예측하기

import numpy as np
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text_data="""Soft as the voice of an angel\n
Breathing a lesson unhead\n
Hope with a gentle persuasion\n
Whispers her comforting word\n
Wait till the darkness is over\n
Wait till the tempest is done\n
Hope for sunshine tomorrow\n
After the shower
"""

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_data])
encoded = tokenizer.texts_to_sequences([text_data])[0]
print(encoded)

print(tokenizer.word_index)
vocab_size = len(tokenizer.word_index) + 1
print('어휘 크기: %d' % vocab_size)

sequences = list()
for i in range(1, len(encoded)):
	sequence = encoded[i-1:i+1]
	sequences.append(sequence)
print(sequences)
print('총 시퀀스 개수: %d' % len(sequences))

sequences = np.array(sequences)
X, y = sequences[:,0],sequences[:,1]
print("X=", X)
print("y=", y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN, LSTM

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
	metrics=['accuracy'])

model.fit(X, y, epochs=500, verbose=2)

# 테스트 단어를 정수 인코딩한다. 
test_text = 'Wait'
encoded = tokenizer.texts_to_sequences([test_text])[0]
encoded = np.array(encoded)

# 신경망의 예측값을 출력해본다. 
onehot_output = model.predict(encoded)
print('onehot_output=', onehot_output)

# 가장 높은 출력을 내는 유닛을 찾는다. 
output = np.argmax(onehot_output)
print('output=', output)

# 출력층의 유닛 번호를 단어로 바꾼다. 
print(test_text, "=>", end=" ")
for word, index in tokenizer.word_index.items():
	if index == output:
		print(word)


###

# 예제: 영화 리뷰 감성 판별하기

import numpy as np

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

imdb = keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

print(x_train[0])

# 단어 ->정수 인덱스 딕셔너리
word_to_index = imdb.get_word_index()

# 처음 몇 개의 인덱스는 특수 용도로 사용된다. 
word_to_index = {k:(v+3) for k,v in word_to_index.items()}
word_to_index["<PAD>"] = 0		# 문장을 채우는 기호
word_to_index["<START>"] = 1		# 시작을 표시
word_to_index["<UNK>"] = 2  		# 알려지지 않은 토큰 
word_to_index["<UNUSED>"] = 3

index_to_word = dict([(value, key) for (key, value) in word_to_index.items()])

print(' '.join([index_to_word[index] for index in x_train[0]]))

from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

x_train = pad_sequences(x_train, maxlen=100)
x_test = pad_sequences(x_test, maxlen=100)

vocab_size = 10000

model = Sequential()
model.add(Embedding(vocab_size, 64,
                    input_length=100))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
          batch_size=64, epochs=20, verbose=1,
          validation_data=(x_test, y_test))

results = model.evaluate(x_test, y_test, verbose=2)
print(results)

review = """What can I say about this movie that was already said? It is my 
favorite time travel sci-fi, adventure epic comedy in the 80's and I love
this movie to death! When I saw this movie I was thrown out by its theme. An
excellent sci-fi, adventure epic, I LOVE the 80s. It's simple the greatest time
travel movie ever happened in the history of world cinema. I love this movie to
death, I love, LOVE, love it!"""

import re
review = re.sub("[^0-9a-zA-Z ]", "", review).lower()

review_encoding = []
# 리뷰의 각 단어 대하여 반복한다. 
for w in review.split():
		index = word_to_index.get(w, 2)	# 딕셔너리에 없으면 2 반환
		if index <= 10000:		# 단어의 개수는 10000이하
			review_encoding.append(index)
		else:
			review_encoding.append(word_to_index["UNK"])

# 2차원 리스트로 전달하여야 한다. 
test_input = pad_sequences([review_encoding], maxlen = 100) 
value = model.predict(test_input) # 예측
if(value > 0.5):
	print("긍정적인 리뷰입니다.")
else:
	print("부정적인 리뷰입니다.")
'''
