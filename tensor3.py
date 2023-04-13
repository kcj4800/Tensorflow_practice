import tensorflow as tf
'''
# 딥러닝 순서
 1. 모델을 만든다.
 2. 손실함수를 만든다.
 3. 학습을 시킨다.
'''
# train_x를 이용해서 train_y값을 찾는 딥러닝
train_x = [1, 2, 3, 4, 5, 6, 7]
train_y = [3, 5, 7, 9, 11, 13, 15]

a = tf.Variable(0.1) # 실전에서는 변수값을 랜덤값으로 랜더마이징해준다.
b = tf.Variable(0.1)

# 손실함수 만들어 넣기
# mean squared error : 정수를 예측하고 싶을 때 사용 mean은 평균이라는 뜻
# cross entropy : 카테고리 분류 및 확률 예측 할때 사용

# mean squared error 사용 
'''
구하고자 하는 데이터 값이 하나 일때 쓰는 mean squred error
def 손실함수()
    return (예측값 - 실제값)^2

구하고자 하는 데이터 값이 두개 이상일때 쓰는 mean squared error
def 손실함수()
    return ((예측1 - 실제1)^2 + (예측2 - 실제2)^2 + ... + (예측n - 실제n)^2)/7

실제 mean squared error(mse) 사용법 
def 손실함수()
    예측값 = 표본값 * a + b
    retrun tf.keras.losses.mse(실제값, 예측값) 이때 실제값과 예측값에는 리스트형식이 들어간다.
    
    '''
def 손실함수():
    예측_y = train_x * a + b
    return tf.keras.losses.mse(train_y, 예측_y)

# 예측_y = train_x * a +b # a와 b는 리스트이므로 원래라면 계산이 불가 하겠지만, 텐서플로상에서는 행렬로 계산이 되기때문에 가능하다.
# print(예측_y)
# exit()

opt = tf.keras.optimizers.Adam(learning_rate=0.01) # learning_rate 잘 나오는 값을 경험적으로 찾을 것.

for i in range(3000):
    opt.minimize(손실함수, var_list=[a, b]) # var_list=[a, b] : 경사하강 하면서 업데이트 되는 변수 목록을 리스트로 적어준다.
    print(a.numpy(), b.numpy())


'''

# opt.minimize(손실함수, var_list=[a, b])에서 손실함수 안에 변수(a, b)를 담고 싶을 땐, 함수만 담을 수 있으므로
# 1. 또 다른 함수를 선언하여 그 안에 손실함수를 담는다.
# 2. lambda 함수를 이용하여 손실함수를 담는다. 익명함수 만들기라는 문법.
# 를 사용하여야 함수를 담아야 한다.

def 손실함수(a, b):
    예측_y = train_x * a + b
    return tf.keras.losses.mse(train_y, 예측_y)

for i in range(3000):
    opt.minimize(lambda : 손실함수(a, b), var_list=[a, b])
    print(a.numpy(), b.numpy())

'''