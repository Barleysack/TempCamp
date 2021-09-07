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
![image-20210907100904599](C:\Users\Finally\AppData\Roaming\Typora\typora-user-images\image-20210907100904599.png)

X_t가 주어지고 h_t-1이 주어져있을 때 

![image-20210907101107385](C:\Users\Finally\AppData\Roaming\Typora\typora-user-images\image-20210907101107385.png)

각 디멘전을 다음과 같이 나타낼 수 있을 것입니다. 
여기서 W는 일종의 완전 연결층에서의 선형변환 층입니다. 이를 W라 칭하겠습니다. 



