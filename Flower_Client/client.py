# https://github.com/adap/flower/tree/main/examples/advanced_tensorflow 참조

import argparse
import os
import re
import time

import tensorflow as tf
import tensorflow_addons as tfa

import flwr as fl

from collections import Counter

import health_dataset as dataset

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# keras에서 내장 함수 지원(to_categofical())
from keras.utils.np_utils import to_categorical

import wandb

import datetime

from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import requests
from fastapi import FastAPI, BackgroundTasks
import asyncio
import uvicorn
from pydantic.main import BaseModel
import logging
import json

# Make TensorFlow logs less verbose
# TF warning log 필터링
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Parse command line argument `partition`
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--n", type=int, choices=range(0, 10), required=True)
args = parser.parse_args()

client_num = args.n  # client 번호

# FL client 상태 확인
app = FastAPI()

class FLclient_status(BaseModel):
    FL_client: int = client_num
    FL_client_online: bool = True
    FL_client_start: bool = False
    FL_client_fail: bool = False
    FL_server_IP: str = None
    FL_round: int = 1  # 현재 수행 round
    FL_next_gl_model: int = 0  # 글로벌 모델 버전


status = FLclient_status()


# Define Flower client
class PatientClient(fl.client.NumPyClient):
    global client_num, next_gl_model

    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        num_rounds: int = config["num_rounds"]

        # round 시작 시간
        round_start_time = time.time()

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.2,
        )

        # round 종료 시간
        round_end_time = time.time() - round_start_time  # 연합학습 종료 시간
        round_client_operation_time = str(datetime.timedelta(seconds=round_end_time))
        logging.info(f'round: {status.FL_round}, round_client_operation_time: {round_client_operation_time}')

        # 다음 라운드 수 증가
        status.FL_round += 1

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)

        loss = history.history["loss"][0]
        accuracy = history.history["accuracy"][0]

        results = {
            "loss": loss,
            "accuracy": accuracy,
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }

        wandb.config.update({'num_rounds': num_rounds, 'epochs': epochs, 'batch_size': batch_size})
        wandb.log({'loss': loss, 'accuracy': accuracy}, step=status.FL_round)

        # self.model.save(f'./local_model/num_{client_num}_local_model/local_model_V{next_gl_model}.h5')

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        print('')
        print('eavluate start')
        print('')

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)

        print(f'test_loss:{loss}, test_accuracy:{accuracy}, ')
        wandb.log({'test_loss': loss, 'test_accuracy': accuracy}, step=status.FL_round)
        num_examples_test = len(self.x_test)

        return loss, num_examples_test, {"accuracy": accuracy, }


@app.on_event("startup")
def startup():
    pass
    # loop = asyncio.get_event_loop()
    # loop.set_debug(True)
    # loop.create_task(run_client())


@app.get('/online')
def get_info():
    return status


@app.get("/start/{Server_IP}")
def main() -> None:
    # # Parse command line argument `partition`

    global client_num, status

    print('FL client start')
    status.FL_client_start = True

    # data load
    # 환자별로 partition 분리 => 개별 클라이언트 적용
    (x_train, y_train), (x_test, y_test), label_count = load_partition()
    print('data loaded')

    wandb.config.update({'label_count': label_count})

    # local_model 유무 확인
    local_list = os.listdir(f'./local_model/num_{client_num}_local_model')
    if not local_list:
        print('init local model')
        model = build_model()

    else:
        # 최신 local model 다운
        print('Latest Local Model download')
        model = download_local_model(local_list)

    try:
        # Start Flower client
        client = PatientClient(model, x_train, y_train, x_test, y_test)
        fl.client.start_numpy_client(server_address="[::]:8080", client=client)

    except Exception as e:
        logging.info('[E][PC0002] learning', e)

        # Client Error
        status.FL_client_fail = True
        # await notify_fail()
        status.FL_client_fail = False
        raise e

    print('FL client end')
    status.FL_client_start = False


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


def download_local_model(listdir):
    # del gl_list[1]  # mac에서만 시행 (.DS_Store 파일 삭제)
    s = listdir[0]  # 비교 대상(gl_model 지정) => sort를 위함
    p = re.compile(r'\d+')  # 숫자 패턴 추출
    local_list_sorted = sorted(listdir, key=lambda s: int(p.search(s).group()))  # gl model 버전에 따라 정렬

    local_model_name = local_list_sorted[len(local_list_sorted) - 1]  # 최근 gl model 추출
    model = tf.keras.models.load_model(f'./num_{client_num}_local_model/{local_model_name}')
    # local_model_v = int(local_model_name.split('_')[1])
    print('local_model_name: ', local_model_name)

    return model


# client manager에서 train finish 정보 확인
async def notify_fin():
    global status
    status.FL_client_start = False
    loop = asyncio.get_event_loop()
    future2 = loop.run_in_executor(None, requests.get, 'http://localhost:9001/trainFin')
    r = await future2
    print('try notify_fin')
    if r.status_code == 200:
        print('trainFin')
    else:
        print(r.content)


# client manager에서 train fail 정보 확인
async def notify_fail():
    global status
    status.FL_client_start = False
    loop = asyncio.get_event_loop()
    future1 = loop.run_in_executor(None, requests.get, 'http://localhost:9001/trainFail')
    r = await future1
    print('try notify_fail')
    if r.status_code == 200:
        print('trainFin')
    else:
        print(r.content)


def load_partition():
    # Load the dataset partitions
    global client_num

    # latest_gl_model_v 값으로 데이터셋 나누기
    data, p_list = dataset.data_load()
    p_df = data[data.subject_id == p_list[client_num]]

    # label 까지 포함 dataframe
    train_df, test_df = train_test_split(p_df.iloc[:, 1:], test_size=0.1)

    # 특정 환자의 label 추출 => 환자마다 보유한 label이 다름
    label_column = train_df.loc[:, 'label']
    label_count = Counter(label_column)
    # label_list = list(label_count)  # 보유 label
    print('patient label list: ', label_count)
    # wandb.config.update({'label_list': label_list})

    # one-hot encoding 범위 지정 => 4개 label
    # Client마다 보유 Label이 다르므로 => 전체 label 수를 맞춰야 함
    train_labels = to_categorical(np.array(train_df.pop('label')), 4)
    test_labels = to_categorical(np.array(test_df.pop('label')), 4)

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
    return (train_df, train_labels), (test_df, test_labels), label_count


if __name__ == "__main__":

    if not os.path.isdir('./local_model'):
        os.mkdir('./local_model')

    if not os.path.isdir(f'./local_model/num_{client_num}_local_model'):
        os.mkdir(f'./local_model/num_{client_num}_local_model')

    # server_status 주소
    inform_SE: str = 'http://0.0.0.0:8050/FLSe/'

    # server_status 확인 => 전 global model 버전
    server_res = requests.get(inform_SE + 'info')
    latest_gl_model_v = int(server_res.json()['Server_Status']['GL_Model_V'])

    # 다음 global model 버전
    next_gl_model = latest_gl_model_v + 1

    # wandb login and init
    wandb.login(key=os.environ.get('wb_key'))
    wandb.init(entity='yangsemo', project='c3', name=f'client_{client_num}_v_{next_gl_model}')

    try:
        # client api 생성 => client manager와 통신하기 위함
        # uvicorn.run("app:app", host='0.0.0.0', port=8001, reload=True)

        # client FL 수행
        main()

        # client FL 종료
        # notify_fin()
        # status.FL_client_fail=False
    # except Exception as e:
    # client error
    # status.FL_client_fail=True
    # notify_fail()
    finally:
        # wandb 종료
        wandb.finish()

        # FL client out
        # requests.get('http://localhost:8003/flclient_out')
        print('%s client close' % client_num)
