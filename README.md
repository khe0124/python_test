# Python Test
파이썬을 이용한 실습.

## Python 설치
* Python 버전
- https://www.python.org/downloads/release/python-375/

## PyCharm
은 나중에 설치하는 것으로

<hr>

## 실습하기
우선 패키지설치
```python
pip install numpy
pip install wordcloud
pip install konlpy
pip install tensorflow
```
- numpy 설치
- wordcloud 설치
- 한글화 설치
- Tensorflow 설치

[머신러닝 기반 데이터 분석]

1. 개론

  1) 머신러닝이란?
   
    - 인공지능은 컴퓨터가 사람의 지능을 흉내는 것을 말한다
      즉 프로그램이 사람 입장에서 내가 마주하고 있는 이 프로그램이 사람인지 컴퓨터인지 구분하지 못하도록 컴퓨터가 지능을 갖춘것을 말한다
      예를 들어 오목프로그램 : 똑같은 상황에서는 똑같은 수를 둔다 -> 명시적 프로그래밍( Explicit Program)

    - 머신러닝이란? 미리 정해진 알고리즘에 따라 동작하는 것이 아니라 학습된 결과에 따라서 프로그램이 작동하는,
      즉 기계를 학습시키는 원리를 철처히 따름.

  2) 머신러닝을 이해하는데 필요한 배경지식 : 선형대수, 미분, 통계, 확률, 프로그래밍

  3) 머신러닝과 딥러닝 관련 프로그램
      - 파이썬 설치
      - 아나콘다
      - 주피터 노트북

 

2. 개발환경 구축

  1) 파이썬

      https://www.python.org
      윈도우 텐서플로우 패키지는 파이썬 3.5와 파이썬 3.6 에서만 동작하므로
      Windows x86-64 executable installer (python-3.5.4-amd64.exe) 다운후 설치.

      ※ 설치시 Add Python 3.5 to PATH 체크할 것

  2) 텐서플로우 라이브러리 설치 : 딥러닝 알고리즘의 구현 및 실행에 최적화
 
      cmd -> pip install tensorflow

  3)  IDLE 파이썬 개발 환경
--------------------------------------------------------------------- hello.py
import tensorflow as tf                              # 오픈소스 라이브러리 모듈
hello = tf.constant('Hello World!')
sess = tf.Session()
print(sess.run(hello))
---------------------------------------------------------------------

 

※ 참고 블로그 https://blog.naver.com/ndb796


3. 텐서플로 기본 개론

  - 텐서플로의 텐서는 다차원 숫자 배열형의 데이터이다
  - 0차원 텐서는 스칼라, 1차원 텐서는 리스트(벡터), 2차원 텐서는 행렬, 3차원 이상일때는 다차원 행렬

  1) 상수, 변수

    - tf.Variable() : 학습 시에 업데이트하는 파라미터를 저장하는 변수. 텐서 생성
    - tf.global_variables_initializer() : 변수 초기화

  2) 플레이스홀더, 피드 딕셔너리

    - tf.placeholder() : 구체적인 형태나 값이 정해지지 않은 임시 변수 (속이 빈 상자)
    - feed_dict : 플레이스홀더와 실제값을 연결하는 역할. 상자 안에 내용을 채워 넣는 역할

 


4. 선형회귀(Linear Regression)  모델

  1) 선형회귀 분석
   
    - 변수 사이의 선형적인 관계를 모델링 한 것
    - 일상생활의 많은 현상들은 선형적인 성격을 가진다
       이러한 선형적인 관계에 적용하는 대표적인 기계 학습 이론(머신러닝)이 선형회귀이다
       학습을 시킨다?
       선형 회귀 모델을 구축한다! : 주어진 데이터를 학습시켜서 가장 합리적인 '직선'을 찾아 내는 것
                                                  따라서 데이터는 3개 이상일 때 의미가 있다

  2) 비용함수 : 반복이 일어날 때마다 개선되고 있는지 확인하기 위해 얼마나 좋은 직선인지를 측정하는 함수

  3) 경사하강법 : 일련의 매개변수로 된 함수가 주어지면 초기 시작점에서 함수의 값이 최소화되는 방향으로
                         매개변수를 변경하는 것을 반복적으로 수행하는 방법.
                         비용함수를 최소화는 매개변수를 찾을수 있다

 

    예제) 매출 예측 프로그램
            노동 시간과 매출 데이터는 아래와 같다
            만일, 8시간을 일했을때 하루 매출이 얼마나 될 것인가 예측해보자

            ---------------------------
            하루 노동 시간    하루 매출
            ---------------------------
              1                  25,000
              2                  55,000
              3                  75,000
              4                 110,000
              5                 128,000
              6                 155,000
              7                 180,000
            ---------------------------

 

 

※ 관련 파이썬 소스


① 해석
---------------------------------------------------------------------- 매출예측.py
import tensorflow as tf

# 데이터 셋팅
xData = [1, 2, 3, 4, 5, 6, 7]
yData = [25000, 55000, 75000, 110000, 128000, 155000, 180000]

# W값을 텐서에게 부여
W = tf.Variable(tf.random_uniform([1], -100, 100))                          

# b값을 텐서에게 부여
b = tf.Variable(tf.random_uniform([1], -100, 100))

# W의 값을 임의로 주기 위해 생성
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#리니어 예측하기
H = W * X + b  

# 예측값 - 실제값의 제곱
cost = tf.reduce_mean(tf.square(H - Y))


# 최소화
# 기울기의 값을 줄여 나가는 것
a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

# Session 생성
sess = tf.Session()
sess.run(init)

# linear 찾기 (학습)
for i in range(5001) :
     sess.run(train, feed_dict={X: xData, Y: yData})
     if i%500 == 0 :
         print(i, sess.run(cost, feed_dict={X: xData, Y: yData}), sess.run(W), sess.run(b))
print(sess.run(H, feed_dict={X: [8]}))

 

 

② 원본
---------------------------------------------------------------------- 매출예측.py
import tensorflow as tf

xData = [1, 2, 3, 4, 5, 6, 7]
yData = [25000, 55000, 75000, 110000, 128000, 155000, 180000]
W = tf.Variable(tf.random_uniform([1], -100, 100))                          
b = tf.Variable(tf.random_uniform([1], -100, 100))
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
H = W * X + b                                                                              
cost = tf.reduce_mean(tf.square(H - Y))
a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(5001) :
     sess.run(train, feed_dict={X: xData, Y: yData})
     if i%500 == 0 :
         print(i, sess.run(cost, feed_dict={X: xData, Y: yData}), sess.run(W), sess.run(b))
print(sess.run(H, feed_dict={X: [8]}))
---------------------------------------------------------------------
