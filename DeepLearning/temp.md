# 딥러닝 이해 9장, 10장 연습문제

---

## 9장

1. 컨볼루션 신경망은 일반적인 심층 신경망에 비하여 어떤 장점을 가지는가?

- 일반적인 심층 신경망은 매개 변수(가중치)가 너무 많으면 과잉적합에 빠질 수도 있고 학습이 늦어진다. 컨볼루션 신경망에서는 하위 레이어의 유닛들과 상위 레이어의 유닛들이 부분적으로만 연결되어 있어 복잡도가 낮아지고 과잉적합에 빠지지 않는다는 장점이 있다.

2. 컨볼루션 신경망은 동물 시각 피질 세포의 구조에서 영감을 받았다. 동물 시각 세포에서 받은 영감이 무엇인지 조사해보자.

- 컨볼루션 신경망의 유닛 사이의 연결 패턴은 고양이의 시각 세포에서 영감을 얻었다. 그들은 고양이에게 여러 방향의 직선 패턴을 보여주고 고양이의 시각 피질을 관찰하였다. 실험의 결과로, 시각 피질 뉴런들은 제한된 시야 영역에서만 자극에 반응함을 알아냈다. 이러한 관찰을 통해 높은 수준을 갖춘 뉴런이 바로 옆에 있는 낮은 수준을 가진 뉴런의 출력에 기반한다는 아이디어를 고안했다.

3. 컨벌루션 연산에 대하여 설명해보자.

- 컨벌루션은 주변 화소값들에 가중치를 곱해서 모두 더한 후에 이것을 새로운 화소값으로 하는 연산이다. 이는 필터링 연산이라고도 하며, 이미지로부터 어떤 특징값을 얻을 때 사용한다.

4. 풀링이란 어떤 연산인가? 풀링에서 보폭이란 무엇인가?

- 풀링이란 서브 샘플링이라고도 하는 것으로 입력 데이터의 크기를 줄이면서, 입력 데이터를 요약하는 연산이다. 풀링 연산을 수행하면 데이터의 크기가 줄어든다. 입력 데이터의 깊이는 건드리지 않는다. 보폭은 커널을 적용하는 거리이다. 보폭이 1이면 커널을 한 번에 1픽셀씩 이동하면서 커널을 적용하는 것이다.

5. “valid” 패딩과 “same” 패딩에 대하여 설명해보자.

- “valid” 패딩은 커널을 입력 이미지 안에서만 움직인다. 즉 커널이 이미지 외부로 나가지 못하게 하는 것이다. 가장 자리 픽셀은 아예 처리하지 않는다. 컨벌루션이 진행될수록 출력은 점점 작아진다.
- “same” 패딩은 입력 이미지의 주변을 특정값으로 채우는 것이다. 0으로 채우는 것을 제로-패딩이라고 한다. 패딩을 적용하면 컨벌루션 후에 입력과 출력의 크기는 같아진다.

6. RGB 색상 채널이 있는 1000x1000 크기의 이미지를 처리하는 신경망의 가중치는 몇 개나 될까? 대략 계산해보자.

- 1000x1000x3으로 3,000,000개의 가중치를 필요로 한다.

7. 컨벌루션 신경망은 “참조의 지역성”을 중시하는 모델이다. 참조의 지역성이란 이미지에서 서로 가까이 있는 화소들은 멀리 떨어져 있는 화소보다 더 중요하다는 의미이다. 왜 그럴까?

- 영상에서 특정 위치의 픽셀들은 그 주변에 있는 픽셀들과 상관성이 높고, 거리가 멀어질수록 영향이 감소한다는 것으로 볼 수 있다. 때문에 특정 범위만 한정해서 처리를 하면 훨씬 더 인식을 잘한다.

8. 컨벌루션 신경망은 일반적으로 3차원으로 배열된 레이어를 가진다. 이유는 무엇일까?

- 컨벌루션 레이어는 여러 개의 커널을 적용한다. 따라서 이런 특징 맵들을 박스 형태로 표시하고, 박스의 깊이는 커널의 개수를 의미한다.

10. 다음 계층에서 2x2 최대 풀링을 수행하고 결과를 적어보자.  
    | | | | |
    |--|--|--|--|
    |28|15|27|190|
    |1|99|70|38|
    |15|12|45|2|
    |10|8|7|6|

    결과

    |     |     |
    | --- | --- |
    | 15  | 38  |
    | 8   | 2   |

11. 7장에서 패션 아이템을 분류하는 MLP 신경망을 작성한 적이 있다. 동일한 작업을 컨벌루션 신경망을 이용하여 시도해보자. 성능이 얼마나 증가되는가?

    ```python
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras import datasets, layers, models

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))

    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=5)
    ```

    MLP
    ![image](https://user-images.githubusercontent.com/76952602/143684455-8dea0480-8233-45bb-982c-a5fb3af2eb51.png)

    CNN
    ![image](https://user-images.githubusercontent.com/76952602/143684464-3f69ce92-561b-417c-86a6-0197909ef420.png)

CNN을 사용하는 것이 시간은 오래걸리지만 정확하다.

## 10장

1. 전통적인 영상 인식 방법과 신경망을 이용한 영상 인식 방법의 차이점을 설명하라.

- 전통적인 영상 인식 방법은 전처리 후 특징을 추출해 물체를 분류하고 신경망을 이용한 영상 인식 방법은 특징 학습과 분류를 모두 전체 학습을 한다.

2. 케라스에서 제공하는 이미지 전처리 기능에 대하여 설명해보자.

- 학습 데이터 혹은 테스트 데이터로 사용자의 이미지 파일을 사용하기 위해 이미지 데이터를 불러올 때 사용할 수 있다. 이미지 파일을 불러와 압축을 풀어 RGB 형태로 픽셀값을 복원한 후, 픽셀값들은 실수 형식의 넘파이 텐서로 변환해준다. 0~255 사이의 픽셀값들을 0.0~1.0 사이의 실수로 스케일링 해주면 된다.

3. CIFAR-10 데이터 세트를 기본적인 심층 신경망으로 처리하는 프로그램을 작성해보자. 이것과 본문의 컨벌루션 버전을 비교해보자. 어떤 쪽이 더 성능이 높은가?

4. 데이터 증대라는 것은 무엇이며, 왜 필요한가?

- 데이터 증대란 한정된 데이터에서 여러 가지로 변형된 데이터를 만들어내는 기법이다.

5. 케라스의 imageDataGenerator() 메소드를 이용해서 주어진 이미지를 다양하게 변형하는 프로그램을 작성해보자.

   ```python
   from tensorflow.keras.preprocessing.image import ImageDataGenerator

   datagen = ImageDataGenerator(rescale = 1./255, rotation_range=90, brightness_range=[0.8, 1.0], width_shift_range=0.2, zoom_range=[0.8, 1.2], height_shift_range=0.2)
   ```

   ![image](https://user-images.githubusercontent.com/76952602/143684501-b5459728-3500-45f3-8fe2-864f1e49bf1b.png)

6. 전이 학습이란 무엇인가? 전이 학습의 3가지 전략에 대하여 설명하라.

- 전이 학습이란 하나의 문제에 대해 학습한 신경망의 모델과 가중치를, 새로운 문제에 적용하는 것이다. 전이 학습의 첫 번째 전략은 새로 만들어진 모델은 전부 새로 학습시키는 사전 훈련 모델의 구조만 사용한다. 두 번째 전략은 사전 훈련된 모델의 일부분은 변경되지 않도록 한 상태에서 나머지 부분을 새로 학습시키고, 세 번째 전략은 특징을 추출한 레이어들은 학습시키지 않는다.

7. 케라스 라이브러리에서 가중치를 저장하려면 어떻게 해야 하는가? 저장된 가중치를 불러와서 사용하려면 어떻게 해야 하는가?

- 가중치를 저장하려면 모델이 가지고 있는 save() 함수를 호출하면 된다.  model.save(‘mymodel’)
  저장된 가중치를 가져오려면 load_model() 함수를 호출하면 된다.  model = load_model(‘mymodel’)

8. 사전 학습된 ConvNet, ResNet, MobileNet과 같은 케라스 애플리케이션 중에서 하나를 선택하여, 다양한 인터넷 사진을 인식하는 프로그램을 작성해보자.

   ```python
   from tensorflow.keras.applications.resnet50 import ResNet50
   from tensorflow.keras.preprocessing import image
   from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
   import numpy as np

   model = ResNet50(weights='imagenet')

   img_path = 'dog.jpg'
   img = image.load_img(img_path, target_size=(224, 224)) # 영상 크기를 변경하고 적재한다.
   x = image.img_to_array(img) # 영상을 넘파이 배열로 변환한다.
   x = np.expand_dims(x, axis=0) # 차원을 하나 늘인다. 배치 크기가 필요하다.
   x = preprocess_input(x) # ResNet50이 요구하는 전처리를 한다.

   preds = model.predict(x)
   print('예측:', decode_predictions(preds, top=3)[0])
   ```

   ![image](https://user-images.githubusercontent.com/76952602/143684606-d38fc7b0-fc46-4e75-8f02-96b1f5853272.png)
