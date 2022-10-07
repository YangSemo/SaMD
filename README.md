
# SaMD(Software as Medical Device)

- 현재 의료 서비스는 건강 데이터를 서버에 수집/저장, ML 모델 활용하여 건강 모니터링 서비스 지원
    
    ⇒ 개인 프라이버시와 관련되어 하나의 서버에 데이터를 수집/저장하면 개인정보 유출 위험 증가 <br>
    ⇒ 특히, 민감한 의료데이터는 비식별화 및 개인 정보 보호 문제가 중요
    
- 연합학습(FL)을 활용하여 로컬 클라이언트에 모든 민감한 데이터를 보관하면서 로컬 컴퓨팅 업데이트를 집계하여 중앙 서버의 조정 하에 공유 글로벌 모델을 학습
- **연합학습을 활용하여 개인화된 다중생체신호 기반 중증도 분류 연합학습 기술 개발**

![1](/images/1.png)

- 활용 Dataset
    - Physionet MIMIC-IV-ED 임상데이터베이스를 활용하여 BIDMC(Beth Israel Deaconess Medical Center)의 중환자실 환자에 대한 다중생체신호 수집
    - 환자의 Vital Signs 기반으로 환자의 중증도 분류를 위하여 데이터 전처리 및 레이블링 수행
        
        ⇒ 중증도 분류 기준: NEWS(National Early Warning Score)
        
        ![2](/images/2.png)
        

- 활용 FL Framework
    - Flower Framework
        - 선정 이유: 다른 Framework에 비해 코드 활용이 간편하고 확장성이 높음
        - 예제 코드 활용: [https://github.com/adap/flower/tree/main/examples/advanced_tensorflow](https://github.com/adap/flower/tree/main/examples/advanced_tensorflow)<br>


# Code Guide

## 0. 준비사항

- wandb 계정 생성 후 API Key 생성

![3](/images/3.png)

- Flower_Client와 Flower_Server 리파지토리에 .env 파일 생성 후 아래와 같이 입력

![4](/images/4.png)

- 각 리파지토리안에 있는 requirments.txt 파일 설치

![5](/images/5.png)

## 1-1. Flower Server

<aside>
💡 여러 Local Model의 weights를 Aggregation 하여 Global Model 생성

</aside>

### app.py

- 초기 글로벌 모델 생성

```python
def init_gl_model():
    init_model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            16, activation='relu',
            input_shape=(5,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='sigmoid'),
    ])

    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    ]

    init_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    return init_model
```

- FL Server 하이퍼파라미터 설정

```python
def fl_server_start(model):
    global server
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        # fraction_fit > fraction_eval이여야 함
        fraction_fit=1.0, # 클라이언트 학습 참여 비율
        fraction_evaluate=1.0, # 클라이언트 평가 참여 비율
        min_fit_clients=2, # 최소 학습 참여 수
        min_evaluate_clients=2, # 최소 평가 참여 수
        min_available_clients=2, # 최소 클라이언트 연결 필요 수
        evaluate_fn=get_eval_fn(model), # 모델 평가 결과
        on_fit_config_fn=fit_config, # batchsize, epoch 수
        on_evaluate_config_fn=evaluate_config, # val_step
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=server.num_rounds),
        strategy=strategy)
```

- global model 평가를 위한 Validation dataset

```python
# global model 평가를 위한 dataset 
  df, p_list = dataset.data_load()

  # Use the last 5k training examples as a validation set
  x_val, y_val = df.iloc[:10000, 1:6], df.loc[:9999, 'label']

  # y(label) one-hot encoding
  y_val = to_categorical(np.array(y_val))
```

- wandb 초기 설정

```python
# wandb login and init
    wandb.login(key=os.environ.get('wb_key'))
    wandb.init(entity='yangsemo', project='server', name=f'server_v_{server.next_gl_model_v}',
               config={"num_rounds": server.num_rounds, "local_epochs": server.local_epochs, "batch_size": server.batch_size})
```

- Global Model은 Flower_Server/gl_model에 버전에 따라 생성됨
    
    ⇒ 지속적으로 연합학습 실행할 때마다 버전 +1
    

### health_dataset.py

- NEWS 기반의 Vital Sign 데이터 불러오기
- `p_list`: 데이터를 많이 보유하고 있는 client(patient)의 ID list

## 1-2. Server_Status

<aside>
💡 FL Server의 상태 확인하여 Client_Manager에 전달

</aside>

## 2-1. Flower_Client

<aside>
💡 각 Client의 Data를 활용하여 Local Model 생성 →
다음 라운드부터 새로운 Global Model의 weights로 학습

</aside>

### app.py

- 데이터 불러오기 및 전처리

```python
def load_partition():

    global client_num

    # Load the dataset partitions
    data, p_list = dataset.data_load()
    p_df = data[data.subject_id == p_list[client_num]]

    # label 까지 포함 dataframe
    train_df, test_df = train_test_split(p_df.iloc[:,1:], test_size=0.1)

    # 특정 환자의 label 추출 => 환자마다 보유한 label이 다름
    label_column = train_df.loc[:,'label']
    label_count = Counter(label_column)
    label_list = list(label_count) # 보유 label
    print('patient label list: ', label_list)

    # one-hot encoding 범위 지정 => 4개 label
    # Client마다 보유 Label이 다르므로 => 전체 label 수를 맞춰야 함
    train_labels = to_categorical(np.array(train_df.pop('label')),4)
    test_labels = to_categorical(np.array(test_df.pop('label')),4)

    train_features = np.array(train_df)
    test_features = np.array(test_df)

    # 정규화
    # standard scaler
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    train_features = np.clip(train_features, -5, 5)
    test_features = np.clip(test_features, -5, 5)

    # return (train_df, train_labels), (test_df,test_labels), len(label_list) # 환자의 레이블 개수
    return (train_df, train_labels), (test_df, test_labels), label_list
```

- Local Model 생성

```python
def build_model():
    # 모델 및 메트릭 정의
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    ]

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            16, activation='relu',
            input_shape=(5,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='sigmoid'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    return model
```

- Local Model은 Flower_Client/local_model/num_clientID_local_model에 버전에 따라 생성됨
    
    ⇒ 지속적으로 연합학습 실행할 때마다 버전 +1
    
- 전체 연학학습 및 라운드 학습 수행 시간 추가

## 2-2. Client_Manager

<aside>
💡 Flower_Client/Server_Status의 상태 확인

</aside>

- 비동기적으로 계속해서 각 컴포넌트의 상태를 확인
    
    ⇒ **개발과정에서 각 구성요소의 상태가 불안정할수 있기 때문에** <br>
    ⇒ **manager가 일정주기로 상태를 확인하고 또는 명령에 대한 반환값을 가지고 정보를 갱신**
    
           manager가 일정주기로 상태를 확인하고 또는 명령에 대한 반환값을 가지고 정보를 갱신
    

# Code 실행

1. 터미널로 Server_Status의 [app.py](http://app.py) 실행

```bash
python app.py
```

1. 새로운 터미널로 Flower_Client의 [app.py](http://app.py) 실행 ⇒ 여러 Client를 실행(—n client_num)
    
    ⇒ Flower_Server의 app.py에서 `min_available_clients(최소 Client 연결 수)` 의 수 를 고려해야 함.
    

```bash
# sudo를 입력해야 wandb 정상 작동
sudo python app.py --n 0
```

1. 새로운 터미널로 Client_Manager의 [app.py](http://app.py) 실행 ⇒ 여러 Client를 실행(—n client_num)

```bash
# sudo를 입력해야 wandb 정상 작동
sudo python app.py --n 0
```

1. 새로운 터미널로 Flower_Server의 [app.py](http://app.py) 실행 

```bash
python app.py
```

1. 위 실행 프로세스를 수행 후 다음 연합학습 부터는 기존 터미널에 존재하는 Flower_Server의 app.py만 실행하면 됨
    
    ⇒ Flower_Server가 실행되면 Client_Manager가 이를 인지 →  앞서 실행한 Flower_Client에 라운드 참여 Trigger 발생시켜 정상적인 연합학습 수행
    

## 주의사항

- 생성하고자 하는 Client 개수 만큼 Flower_Client와 Client_Manager를 터미널로 여러 개 생성하고 python code를 실행해야 함.
    
    **⇒ 추후 이 방법을 간편하게 수정**
    
    - ex: 2개의 Client 생성
        - Flower_Server/Server_Status: 터미널 1개 씩  ⇒ 총 2개
        - Flower_Client: 터미널 2개
        - Client_Manager: 터미널 2개
    
    ![6](/images/6.png)
    

![7](/images/7.png)

### Simple 실행

1. Server_Status의 python app.py 실행
2. Flower_Server의 sudo python app.py 실행
3. Flower_Client의 run_client.sh 실행(sudo sh run_client.sh)

```bash
# client 수 설정(0~9 => 10 clients)
for i in `seq 0 9`; do
    echo "Starting client $i"
		python client.py의 경로 --n=${i} &
    # 예시: python /Users/yangsemo/pycharm/news_flower/Flower_Client/client.py --n=${i} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
```

# SaMD PoC Result

- 10명의 Client(Patient)를 Tensorflow 기반 아주 간단한 Sequential Model을 생성하여 연합학습 수행
- Aggregation: FedAvg

1. Round: 50, epochs: 20
    - Global Model Result
        
        ![8](/images/8.png)
        
    - 각 Client의 Local Model
        
        ![9](/images/9.png)
        
    - 각 Client의 Label 개수 및 하이퍼파라미터 설정 ⇒ Non IID 확인 가능
        - 예시: Client ID: 7
    
    ![10](/images/10.png)
    

1. Round: 100, epochs: 30
    - Global Model Result
    
    ![11](/images/11.png)
    
    - 각 Client의 Local Model

![12](/images/12.png)
