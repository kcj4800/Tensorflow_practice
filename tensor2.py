import tensorflow as tf

# 문제 : 키를 이용해 신발사이즈를 추론해보자.

키 = [170, 180, 175, 160]
신발 = [260, 270, 265, 255]

# y = ax+b 형태로 구한다.
a의_키 = 170
a의_신발 = 260

# 신발 = 키 * a + b

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def 손실함수():
    예측값 = a의_키 * a + b
    return tf.square(a의_신발 - (예측값))
# 손실값(오차) : 실제값 - 예측값에서 절대값으로 구하기 위해 제곱을 해준다. (tf.square(손실값))

# 경사하강법 - tf.keras.optimizers.Adam(learning_rate=0.1) : 경사하강법에 의해 w값을 업데이트해준다. 
# 이중 Adam이 가장 보편적으로 쓰인다. - 기울기값을 자동으로 조절해주며 스마트하게 업데이트해줌.
# learning_rate=0.1 값은 기본값으로 0.0001 정도 설정되어 있으므로, 반드시 넣어주지는 않아도 된다.
opt = tf.keras.optimizers.Adam(learning_rate=0.1)
# optimizer를 .minimize(손실함수(loss_function), var_list=[a, b] )해주면 경사하강이 일어난다. 
# 두가지 파라미터를 넣어준다. var_list=[a, b]에는 경사하강법으로 업데이트할 weight Variable 목록을 넣어준다.

# opt.minimize(손실함수, var_list=[a, b]) # 이대로 실행해주면 경사하강이 한번 일어난다.

for i in range(300):
    opt.minimize(손실함수, var_list=[a, b])
    print(a.numpy(), b.numpy())