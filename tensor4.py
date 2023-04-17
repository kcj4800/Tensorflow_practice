import tensorflow as tf
import pandas as pd
import numpy as np

# data preprocessing - 데이터 전처리
data = pd.read_csv('gpascore.csv')
# print(data)
# print(data.isnull().sum()) # 빈칸 세주는 함수
data = data.dropna() # data.dropna() 빈칸을 없애주는 함수
# print(data.isnull().sum())
# # data = data.fillna(100) # 빈칸을 임의값으로 바꿔주는 함수 데이터의 평균값등을 넣는다.
# print(data['gpa']) # data 안의 'gpa' 열 모두 출력
# print(data['gpa'].min()) # data 안의 'gpa' 열 값중 최솟값 출력
# print(data['gre'].max()) # data 안의 'gre' 열 값중 최댓값 출력
# print(data['gre'].count()) # data안의 'gre' 열 값이 몇개인지 출력
# exit()

y데이터 = data['admit'].values # admit을 리스트로 담아준다.
# print(y데이터)

x데이터 = []
# 판다스로 연 데이터를 dataframe이라고 한다.
# data.iterrows() : 판다스 데이터(dataframe)에서만 사용되는 한 행씩 출력 해주는 함수 
for i, rows in data.iterrows():
    x데이터.append([rows['gre'], rows['gpa'], rows['rank']])
    # print(rows)
    # print(rows['gre'])
    # print(rows['rank'])

# x데이터 : [[380, 3.21, 3],[660, 3.67, 3] ... [],[],]
# y데이터 : [0, 1, 1, ... 1, 0] 혹은 [[0],[1],[1]...[1],[0]] 등으로 되어있다.

# print(x데이터) 
# x데이터, y데이터 안에 리스트를 담기에 성공했지만 이것만으로는 텐서플로에 넣을 수 없다.
# 텐서플로에 넣기 위해 이를 numpy array 혹은 tf tensor에 담아 줘야 한다.
# exit()
# tf. keras.models.Sequential([
#   tf.keras.layers.Dense(노드의 갯수, 파라미터), - 파라미터 : activation = 'sigmoide', 'tanh', 'relu', 'softmax'  등등 
#   tf.keras.layers.Dense(128, activation = 'tanh'), 
#   tf.keras.layers.Dense(1, activation = 'sigmoid'),
#   마지막 레이어는 항상 예측 결과를 출력해야한다. 0 ~ 1 사이의 확률로 결과를 출력하기 위해서 simoid 함수를 쓴다.
# ])
# 
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# 2. model compile하기 optimizer - learning rate 값을 자동으로 스마트하게 조절해주는 고마운 기능
# optimizer 목록 : adam, adagrad, adadelta, rmsprop, sgd 등
# loss함수 - mse(mean squared error), binary_crossentropy(결과가 0과 1사이의 분류/ 확률문제에서 쓴다.)
# metrics = ['accuracy'] 모델을 어떤요소로 평가할건지를 정한다. 보통 'accuracy'로 쓴다.

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

# x데이터 : 트레이닝 데이터(학습시킬 데이터), y데이터 : 실제 정답이 되는 데이터, epochs : 몇번을 학습시킬건지를 정한다.
model.fit( np.array(x데이터), np.array(y데이터), epochs=1000) 
# tensorflow를 하기 위해서는 데이터를 np.array()에 담아줘야한다.
# numpy : 행렬 벡터 담을때나 리스트 안에 리스트(다차원 리스트)를 만들때 사용
'''
이런식으로 학습을 하게 되는데, loss : 예측값과 실제값의 차이(오차), accuracy : 정확도 
Epoch 100/100
14/14 [==============================] - 0s 3ms/step - loss: 0.6322 - accuracy: 0.6447 
'''

# 예측
예측값 = model.predict([[750, 3.70, 3], [400, 2.2, 1]])
print(예측값)


