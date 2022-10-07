# https://github.com/adap/flower/tree/main/examples/advanced_tensorflow 참조
import logging
import re
from typing import Dict,Optional, Tuple

import flwr as fl
import tensorflow as tf
import tensorflow_addons as tfa

from keras.utils.np_utils import to_categorical

import numpy as np

import health_dataset as dataset

import wandb
import datetime
import os
import time

import requests, json

from dotenv import load_dotenv
load_dotenv(verbose=True)


# Log 포맷 설정
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


# FL 하이퍼파라미터 설정
class FL_server_parameter:
    num_rounds = 50
    local_epochs = 20
    batch_size = 32
    val_steps = 5
    latest_gl_model_v = 0 # 이전 글로벌 모델 버전
    next_gl_model_v = 0 # 생성될 글로벌 모델 버전


server = FL_server_parameter()


# latest global model download
def model_download():
    try:
        gl_list = os.listdir('./gl_model')

        # mac에서만 시행 (.DS_Store 파일 삭제)
        if '.DS_Store' in gl_list:
            i = gl_list.index(('.DS_Store'))
            del gl_list[i]

        s = gl_list[0] # 비교 대상(gl_model 지정) => sort를 위함
        p = re.compile(r'\d+') # 숫자 패턴 추출
        gl_list_sorted = sorted(gl_list, key=lambda s: int(p.search(s).group())) # gl model 버전에 따라 정렬

        gl_model_name = gl_list_sorted[len(gl_list_sorted)-1] # 최근 gl model 추출
        gl_model = tf.keras.models.load_model(f'./gl_model/{gl_model_name}')
        gl_model_v = int(gl_model_name.split('_')[2])

        logging.info(f'gl_model: {gl_model}, gl_model_v: {gl_model_v}')

        return gl_model, gl_model_v

    # s3에 global model 없을 경우
    except Exception as e:
        logging.info(f'global model read error: {e}')

        model_X = None
        gl_model_v = 0
        logging.info(f'gl_model: {model_X}, gl_model_v: {gl_model_v}')
        return model_X, gl_model_v


def main(model) -> None:

    global server

    logging.info(f'latest_gl_model_v: {server.latest_gl_model_v}')

    if not model:

        logging.info('init global model making')
        init_model = init_gl_model()

        fl_server_start(init_model)

    else:
        logging.info('load latest global model')

        fl_server_start(model)


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


def get_eval_fn(model):
    global server

    """Return an evaluation function for server-side evaluation."""
    # Load data and model here to avoid the overhead of doing it in `evaluate` itself

    # The `evaluate` function will be called after every round
    def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        
        # loss, accuracy, precision, recall, auc, auprc = model.evaluate(x_val, y_val)
        loss, accuracy = model.evaluate(x_val, y_val)

        wandb.log({'loss': loss, 'accuracy': accuracy})

        # model save
        model.save('./gl_model/gl_model_%s_V.h5' % server.next_gl_model_v)

        return loss, {"accuracy": accuracy, }

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """

    global server

    config = {
        "batch_size": server.batch_size,
        # "local_epochs": 1 if rnd < 2 else 2,
        "local_epochs": server.local_epochs,
        "num_rounds": server.num_rounds,
    }

    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    # val_steps = 5 if rnd < 4 else 10
    global server

    # wandb log upload
    wandb.config.update({"val_steps": server.val_steps},allow_val_change=True)
    
    return {"val_steps": server.val_steps}


if __name__ == "__main__":

    # global model dir 생성
    if not os.path.isdir('./gl_model'):
        os.mkdir('./gl_model')

    # global model download
    model, server.latest_gl_model_v = model_download()

    server.next_gl_model_v = server.latest_gl_model_v + 1

    # server_status 주소
    inform_SE: str = 'http://0.0.0.0:8050/FLSe/'

    # wandb login and init
    wandb.login(key=os.environ.get('wb_key'))
    wandb.init(entity='yangsemo', project='f_s', name=f'server_v_{server.next_gl_model_v}',
               config={"num_rounds": server.num_rounds, "local_epochs": server.local_epochs, "batch_size": server.batch_size})

    today = datetime.datetime.today()
    today_time = today.strftime('%Y-%m-%d %H:%M:%S')

    inform_Payload = {
            # 형식
            'S3_bucket': 'fl-flower-model', # 버킷명
            'S3_key': 'model_V%s.h5'%server.latest_gl_model_v,  # 모델 가중치 파일 이름
            'play_datetime': today_time, # server 수행 시간
            'FLSeReady': True, # server 준비 상태 on
            'GL_Model_V': server.latest_gl_model_v # GL 모델 버전
        }

    # server_status => FL server ready
    requests.put(inform_SE+'FLSeUpdate', data=json.dumps(inform_Payload))
    
    
    # global model 평가를 위한 dataset 
    df, p_list = dataset.data_load()

    # Use the last 5k training examples as a validation set
    x_val, y_val = df.iloc[:10000, 1:6], df.loc[:9999, 'label']

    # y(label) one-hot encoding
    y_val = to_categorical(np.array(y_val))
    
    try:
        # Flower Server 실행
        fl_start_time = time.time()

        main(model)

        fl_end_time = time.time() - fl_start_time  # 연합학습 종료 시간
        fl_server_operation_time = str(datetime.timedelta(seconds=fl_end_time))  # 연합학습 종료 시간
        logging.info(f'fl_server_operation_time: {fl_server_operation_time}')

        # server_status에 model 버전 수정 update request
        res = requests.put(inform_SE + 'FLRoundFin', params={'FLSeReady': 'false'})

        # FL server error
    except Exception as e:
        logging.error(f'error: {e}')
        data_inform = {'FLSeReady': False}
        requests.put(inform_SE + 'FLSeUpdate', data=json.dumps(data_inform))
         
    finally:
        wandb.finish()
        print('server close')
