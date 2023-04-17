import tensorflow as tf
# 텐서플로에서 불러온 데이터들을 이미지로 보기 위해서
# 맷플롯 라이브러리를 임폴트하여준다. 
import matplotlib.pyplot as plt
import numpy as np


'''
# 딥러닝 순서
 1. 모델만들기
 2. compile하기
 3. fit하기
'''

# 구글이 호스팅해주는 데이터셋 중 하나
# tf.keras.datasets.fashion_mnist.load_data()
# ((어쩌구, 저쩌구),(어쩌구, 저쩌구)) 라고 저장된 데이터들을 각각 변수로 쉽게 빼주는 문법

((trainX, trainY),(testX, testY)) = tf.keras.datasets.fashion_mnist.load_data()

# 이미지 데이터를 가공하기 쉽게 숫자로 바꿔서 셋팅해준것. trainX는 이미지 6만개
# print(trainX) 
# trainX[0]은 첫번째 이미지 이미지 한개에는 가로 28개 세로 28개의 행과 열로 이루어져있다.
# print(trainX[0])
# print(trainX.shape) 
# (60000, 28, 28) 28행 28열의 행렬 60000개로 이루어져 있다.
# print(trainY) # [9 0 0 ... 3 0 5] 처럼 정답 데이터가 들어있다. - 사진에 종류별로 라벨링을 한것.

# 이미지 데이터 전처리 0~255 => 0~1로 압축해서 넣음.
trainX = trainX / 255.0
testX = testX / 255.0

trainX = trainX.reshape( (trainX.shape[0], trainX.shape[1], trainX.shape[2], 1) )
TestX = testX.reshape( (testX.shape[0], 28, 28, 1) )

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# 이미지를 파이썬으로 띄워보는 법
# plt.imshow( trainX[1])
# plt.gray()
# plt.colorbar()
# plt.show()

# 1. 모델만들기
# relu(Rectified Linear Unit)함수 : 정류된 선형 함수로 +/-가 반복되는 신호에서 -흐름을 차단한다는 의미이다.
# 음수값을 다 0으로 만들어주며 convolution layer에서 자주 사용된다.
# https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=handuelly&logNo=221824080339
# sigmoid : 결과를 0~1로 압축시켜준다. - binary(2진) 예측문제에 마지막 노드 갯수 1개로 주로 사용된다.(예: 대학원 붙는다/ 안붙는다.)
# softmax : 결과를 0~1로 압축시켜주며, 여러 카테고리 중 어디에 속하는지 묻는 예측문제에 사용된다. 마지막 노드갯수는 카테고리 갯수이다.
# [0.2 0.2 0.1 ... 0.4] => 다 더하면 1이된다.

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28,1)), 
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28,1)), 
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28,1)), 
    tf.keras.layers.MaxPooling2D((2,2)),
    # tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28,1)), 
    # tf.keras.layers.MaxPooling2D((2,2)),
    # tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28,1)), 
    # tf.keras.layers.MaxPooling2D((2,2)),
  
    # 컨볼루션포함 레이어 구성 순서 
    # Conv - Pooling * 여러번
    # Flatten - Dense -> 출력
    
    # 32개의 다른 feature 생성해주세요, 3,3 의 kernel사이즈(실험적으로 적용)
    # padding : Convolution하게되면 가로, 세로 1픽셀씩 작아지게되는데 작아진 만큼 테두리를 추가하여 (28, 28)사이즈를 유지해준다.
    # activation='relu' : 이미지를 숫자로 바꾸면 0~255사이로 -값이 존재하지 않아야 한다. relu함수를 써서 음수는 모두 0으로 처리해준다.
    # MaxPooling2D를 이용하여 이미지의 중요한 포인트들을 가운데로 모아주는 작업 (2,2) => pooling_size
    # tf.keras.layers.Dense(128, input_shape=(28, 28), activation='relu'),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])
model.summary()
# exit()

# model.summary() : 모델을 잘 짰는지 한눈에 확인할수있는 요약본. 
# 요약본을 사용하기 위해서는 tf.keras.layers.Dense(128, input_shape=(28,28), activation='relu')를 넣어줘야한다.
# input_shape=(데이터 하나의 shape)를 넣어줘야 summary보기 가능.
# 마지막 Dense의 Output Shape가 (Nonw, 28, 10)이므로 마지막 레이어 앞에 tf.keras.layers.Flatten(),을 넣어 준다.
# Flatten 레이어는 행렬을 1차원으로 압축해준다. [[1, 2, 3, 4], [5, 6, 7, 8]] => [1,2,3,4,5,6,7,8]
# 마지막 열의 param은 해당 레이어에서 Train할수있는 w,b의 갯수(학습가능한 w,b갯수)

# 2. compile 하기
# loss= 'mse'(mean squared error), 'binary_crossentropy'(결과가 0과 1사이의 분류/ 확률문제에서 쓴다.)
# loss='Sparse_categorical_crossentropy' : 답안(trainY)데이터가 정수(0, 1, 2..)등으로 인코딩 되어있을때 사용
# loss='categorical_crossentropy' : 답안(trainY)데이터가 원핫 인코딩 되어있을때 사용
# optimizer - learning rate 값을 자동으로 스마트하게 조절해주는 고마운 기능
# optimizer 목록 : adam, adagrad, adadelta, rmsprop, sgd 등이 있으며, adam을 범용적으로 많이 쓴다.
# metrics = ['accuracy'] 모델을 어떤요소로 평가할건지를 정한다. 보통 'accuracy'로 쓴다.

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3. fit 하기
# model.fit(trainX, trainY, epochs=5)

# 마지막 테스트 및 평가
# 학습용 trainX 데이터를 넣으면 안된다.(답안지를 외울 수 있다.) 처음보는 데이터를 넣어줘야한다.
# score = model.evaluate( testX, testY) # [loss, accuracy] - overfitting 현상 (학습데이터의 답안을 외워버린 현상) - 새로운 데이터에 accuracy가 낮아짐.
# print(score)

# 그러므로 모델을 fit할때 validation_data를 이용해 epoch 1회가 끝날때마다 채점하는 방식을 적용해준다.
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5)

# val_loss와 val_accuracy를 이용하여 overfitting 전에 학습을 중단하고, val_accuracy를 높일 방법을 찾는다.
# 1. Dense layer 추가
# 2. Conv + Pooling 추가
'''
tf.keras.layers.Faltten()을 이용하여 2D 혹은 3D이미지를 1차원으로 바꿔주면 응용력이 떨어지고, 조금만 변해도 이전에 학습한 가중치가 의미가 없어진다.
해결책 : convolutional layer
 1. 이미지에서 중요한 정보를 추려서 복사본 20장을 만든다.
 2. 그곳엔 이미지의 중요한 feature, 특성이 담겨있다.
 3. 이를 통한 학습.
위의 해결방식을 feature extraction 이라한다. - 전통적인 사물인식 머신러닝방법에는 가이드가 필요하다.
convolution layer(feature map 만들기) - kernel 디자인
 '''
