import tensorflow as tf
'''텐서플로우에서는 변수와 상수 기능을 제공한다.
상수는 변하지 않는 숫자를 의미하며 텐서플로우에서는 constant() 함수를 이용해서 정의할 수 있다.
'''

a = tf.constant(1, tf.float32)  
# 정수를 넣으면 자동으로 int32의 데이터 타입으로 만들어진다. 보통 float형이 많이 필요하므로, int형일경우 float형으로 바꿔주는게 좋다.
b = tf.constant(2.0)  # 2.0으로 넣어주면 float32형의 데이터 타입으로 만들어진다.
c = tf.add(a,b)     # a와 b를 더한 그 tensor의 값을 담아줌
# d = tf.subtract(a, b)
d = tf.constant([[3, 4, 5],
                 [6, 7, 8]])
e = tf.divide(a, b)
f = tf.multiply(a, b)
print(a, b, c, d, e, f)

# sess = tf.session() # 하나의 session 객체 생성

tensor = tf.constant([3, 4, 5])
tensor2 = tf.constant([6, 7, 8])
tensor4 = tensor+tensor2

tensor3 = tf.constant([[1, 2], 
                       [3, 4]])
tensor5 = tf.constant([[3, 4],
                       [5, 6]])
print(tensor4, tensor3)
tensor6 = tf.matmul(tensor3, tensor5) #tf.matmul() : 행렬곱(Matrix Multiplication) a와 b의 행렬곱(dot product)
print(tensor6)

tensor7 = tf.zeros(10) # 0으로 가득 찬 텐서를 하나 만들어준다. [0,0 ....0]
print(tensor7)
print()
tensor8 = tf.zeros([2, 2]) # 2행 2열의 0으로 가득찬 텐서를 생성
tensor9 = tf.zeros([4, 2, 3]) # 3행 2열의 0으로 가득찬 텐서를 4개 생성 - (갯수, 열, 행) 순서로 생성해준다.
print(tensor8,'\n\n', tensor9)
print(tensor.shape, tensor5.shape, tensor8.shape, tensor9.shape) # (열, 행) 또는 (갯수, 열, 행)으로 표현해준다.
tf.cast(tensor6, tf.float32) # tf.cast함수를 이용해서 데이터 타입을 바꿔줄 수 있다. 하지만 값이 변하는게 아니다.
print(tensor6) # 위에서 tf.cast로 실수형으로 바꿨지만 여전히 int형으로 나온다.
print(tf.cast(tensor6, tf.float32)) # 고로 필요할때 바꿔주는 식으로 쓴다.

# constant : 상수를 생성 - 고정값으로 변경이 불가
# Variable : 변수를 생성해준다. - 딥러닝상에서는 weight값이라고 생각하면 된다. 변경이 가능하며, 수시로 바뀔 수 있는 값이다.
w = tf.Variable(1.0)
print(w) # Variable로 선언된 변수는 <tf.Variable 'Vaiable:0' shape=() dtype=float32, numpy=1.0> 식으로 표현된다.
print(w.numpy()) # Variable로 선언된 변수를 읽을 때 사용한다.
w.assign(2.0) # Variable로 선언된 변수를 변경해줄 때 사용한다.
print(w.numpy()) 