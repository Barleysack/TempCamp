# Basics of Recurrent Neural Networks

---



RNN은 시퀀스데이터가 입출력으로 들어오는 상황에서, 각 타임스텝에서 들어오는 ```이번 타임스텝 전까지 들어온 정보와 현재 들어온 ht를 계산하는 것입니다. 

서로 다른 타임스텝에서 들어오는 입력 데이터를 처리할때 동일한 파라미터를 가진, RNN을 매 타입스텝에서 동일하게 사용하게 된다는 것입니다. 

둘둘 말아놓은 것을 Rolled 버전, 쭉 펼친것을 Unrolled version

매 타임스텝에서 우리가 원하는 출력값에 맞는 출력을 내면서, 다음 타임스텝의 입력으로 계산해야 한다. 

#### RNN의 구성요소

-  h_t-1 : 이전까지의 타임스텝에서의 은닉층 벡터 
- x_t : 입력 벡터
- h_t : 새로운 은닉층의 벡터
- f_w : RNN펑션, 파라미터 W
- y_t : 타임스텝 t에서의 출력벡터

RNN에서 정의해야하는 히든 스테이트 벡터의 노드수, 레이어 수는 역시 그 하이퍼파라미터가 됩니다. 


X_t가 주어지고 h_t-1이 주어져있을 때 



각 디멘전을 다음과 같이 나타낼 수 있을 것입니다. 
여기서 W는 일종의 완전 연결층에서의 선형변환 층입니다. 이를 W라 칭하겠습니다. 


이때 tanh를 통과해 비선형변환 함으로서 이번 층의 \h_t 를 만들고 

W_ht를 Y_t로 변환해 마지막 출력인 y_t를 낸다고 할 수 있습니다. 

### Type-of_RNN

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/41d7119c-2ab9-4d30-af38-2fe21b4638c1/Untitled.png)

- One-to-One :일반적인 뉴럴 네트워크를 나타낸다 .
- One-to-many : 입력은 하나의 단일 입력, 이 이미지의 설명등을 예측/ 생성하기 위해 각 타임스텝별로 순차적인 생성을 하게 되는 것입니다. 입력은 하나, 출력은 여러 타임스텝. 이때 같은 타임스텝수의 입력이 들어가지만, 첫번째 이후에는 모두 0인 텐서로 들어가기 마련.
- many-to-one : 여러 시퀀셜한 입력을 받아 타임스텝에 따라 계산해 마지막 출력은 마지막 타임스텝 h_t를 가지고 원하는 값을 도출해내는 것.
- many-to-many : 입력도 여러개, 출력도 여러개. EX) machine translation 입력 마지막 타임스텝에서 출력 첫 타임스텝 생성해주는 것인듯.
- many-to-many2 : 입력이 주어질때마다 예측을 수행하는 형태입니다. 장면분석 등에 사용할 수 있습니다. 실시간성이 요구되는 경우에 이 many-to-many의 구조를 활용한다.

#### Character-Level Language Model

- example of training sequence "hello"
- vocab : {h,e,l,o}
- ex ) "hello"

총 사전의 갯수, 만큼의 길이를 가지는 원-핫 벡터로 나타낼 수 있다. 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2865c928-6f41-4f2b-93ef-8cda68a1accb/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b401c2d9-dc05-41a7-8205-92c7fbe7cd4e/Untitled.png)

아웃풋 : logit이라 정하고, output layer의 노드 수는 사전의 크기와 동일하게 나오기 마련. 

output layer가 로짓값으로서 칭하는 이유는 softmax에 들어가기 전의 이야기를 나타내기 때문.

아웃풋 레이어에서 해당하는 벡터가 가장 큰 값을 가져 SOFTMAX에서 나갈 수 있도록 학습하는 것을 목표로 한다. 

물론, 한 문단에 대해서 학습하는 것 또한 가능. 이떄, 줄바꾸기, 공백, 쉼표 등을 모두 특수문자로서 학습할 수 있다.

캐릭터 레벨 언어모델을 통해, latex 마저 학습이 가능하다.

## BPTT

전체 시퀀스를 다 학습하는 것은 컴퓨팅 코스트에도 무리고, 여러모로 문제가 있으므로, 일정한 양을 truncate하여 학습한다. 학습할 시퀀스의 길이를 한정함으로서 이득을 얻어낼 수 있다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9a7acb7c-89a9-4e63-9fea-351d8a5ed054/Untitled.png)

지금까지의 정보는 은닉 벡터에 담겨져야 한다.

h 하나의 차원 값을 고정하고, 시퀀스가 진행함에 따라 해당 값이 어떻게 변하는지를 보며 해당 네트워크를 분석할 수 있다. 

### Vanishing/Exploding

- 바닐라 RNN에서는, 타임스텝이 지남에 따라, 여러 타임 스텝 이전에 학습한 정보를 쓰려면, $W_h$가 기하급수적으로 작아지거나, 폭발할 가능성이 생깁니다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f99eeaa2-52f0-4b1c-b2cf-9157d53aa80e/Untitled.png)

이런식으로 타임스텝마다 거듭제곱으로 걸리기때문에, 문제가 생긴다.

몇 타임스텝 너머 그래디언트를 전달하게 된다면, 기하급수적 변화로 사용이 어려워진다. 

이를 위해 사용하는 것으로서,

### LSTM&GRU

상기한 그래디언트 문제와, 롱-텀 문제를 어느정도 완화해주는 lstm과 Gru. 

- LSTM??

매 타임스텝마다 변화하는 히든스테이트 벡터를 단기기억을 담당하는 기억소자로서 볼 수 있습니다. 단기기억을 시퀀스가 타임스텝별로 진행됨에 따라 이를 더 길게 저장할 수 있도록 하는 것.

$h_t = f_w(x_{t1}h_{t-1})$

${c_t,h_t}=$LSTM($x_t,C_{t-1},h_{t-1})$

이로서 보다 완전한 정보를 포함한 정보가 되는 것. 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/022c4033-d7dd-4914-9bd2-b89504613b10/Untitled.png)

위의 게이트들이 셀스테이트와 히든스테이트를 계산해낸다. 

- i :
- f: 앞서 계산된 두개의 입력을 받아서 시그모이드에 해당되는 게이트를 통과하게 되고, 각 벡터가 forget 내부의 element-wise로 곱해지며 날아가게 되는것.
- g:  generate information to be added and cut it by input gate 
$i_t = \sigma (W_i * [h_{t-1},x_t] + b_c)$

    $C_t^{=} = tanh(W_c *[h_{t-1},x_t]+b_c)$
    $C_t = f_t* C_{t-1} + i_t* c_t$

- o: output, passing cell state to next time step, and output or next layer if needed

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f10ff873-6b2f-44bf-8d02-218e30b4b533/Untitled.png)

 

### GRU? :

경량화된 LSTM, 더 빠른 계산 속도. 

이원화 되었던 Cell state 벡터와 hidden state 벡터를 일원화해서 

오직 h만을 존재시키는 것이 제일 큰 특징이다. 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f5c251f0-d4d9-4ab4-a4a6-8bec1875dfbc/Untitled.png)

h_t의 업데이트 : 인풋게이트 만을 사용하고, 포겟게이트 자리에는 1-인풋게이트 값을 사용했다. 

GRU와 LSTM은 그 업데이트 과정에서 곱연산이 아닌 합연산을 사용한다는 점에서 그래디언트 소실/폭주 문제가 해결되었다고 볼 수 있다. 

덧셈 연산은 백프롭 때 변형 없이 전달할 수 있다. 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0cbc456d-e912-4349-a487-f6c0121c67f3/Untitled.png)
