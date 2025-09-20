
from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path
from flwr.server import ServerConfig  # 추가

from grpc import server

import flwr as fl
import tensorflow as tf
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
)
# from typing import Callable
import numpy as np
from datetime import datetime
import os
from tensorflow.keras import layers as L
import efficientnet.tfkeras as efn


def fit_config(server_round: int) -> Dict[str, int]:
    """Return local training configuration for clients each round.

    기존 기본값(1 epoch) 대신 local_epochs를 늘려 연합학습 효과를 강화합니다.
    """
    # 기존: 서버에서 config를 전달하지 않아 클라이언트 기본 1 epoch 사용
    # return {}
    return {"local_epochs": 3}

def load_model():
    IMAGE_SIZE = [384, 384]

    model = tf.keras.Sequential([
        efn.EfficientNetB2(
            input_shape=(*IMAGE_SIZE, 3),
            weights=None,
            include_top=False
        ),
        L.GlobalAveragePooling2D(name='global_average_pooling2d'),
        L.Dense(1024, activation='relu', name='dense'), 
        L.Dropout(0.3, name='dropout'), 
        L.Dense(512, activation='relu', name='dense_1'), 
        L.Dropout(0.2, name='dropout_1'), 
        L.Dense(256, activation='relu', name='dense_2'), 
        L.Dropout(0.2, name='dropout_2'), 
        L.Dense(128, activation='relu', name='dense_3'), 
        L.Dropout(0.1, name='dropout_3'), 
        L.Dense(1, activation='sigmoid', name='dense_4')
    ])

    model.compile(
        optimizer='Adam',
        loss = 'binary_crossentropy',
        metrics=['binary_crossentropy', 'accuracy'],
    )
    
    return model

model = load_model()

# 안전한 가중치 로드: 여러 후보 파일 시도
def _try_load_weights(m):
    candidates = [
        './melamodel/melamodel_weights072.h5',
        './melamodel/melamodel_weights072.weights.h5',
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                m.load_weights(p)
                print(f"Loaded weights: {p}")
                return True
            except Exception as e:
                print(f"Failed to load {p}: {e}")
    print("Proceeding without pre-trained weights (random init).")
    return False

_ = _try_load_weights(model)


class SaveModelStrategy(fl.server.strategy.FedAvg):
    # def initialize_parameters(
    #     self, client_manager: ClientManager
    # ) -> Optional[Parameters]:
    #     """Initialize global model parameters."""
    #     # initial_parameters = self.initial_parameters
    #     # self.initial_parameters = None  # Don't keep initial parameters in memory
    #     initial_parameters = model.get_weights()
    #     return initial_parameters

    # def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    #     """Return a function which returns training configurations."""

    #     def fit_config(server_round: int) -> Dict[str, str]:
    #         """Return a configuration with static batch size and (local) epochs."""
    #         config = {
    #             "learning_rate": str(0.00001),
    #             "batch_size": str(8),
    #         }
    #         return config

    #     return fit_config

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:
        if not results:
            return None

        # model = load_model()
        # Weight accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {accuracy_aggregated}")
        
        # only 2 decimal places on accuracy_aggregated
        accuracy_agg2 = round(accuracy_aggregated, 2) 
        aggregated_weights = super().aggregate_fit(server_round, results, failures)   
        aggregated_params, _ = aggregated_weights
        if aggregated_params is not None:
            aggregated_weights_h : List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_params)
            # modell = tf.keras.models.clone_model(model)
            model.set_weights(aggregated_weights_h)
            print(f'Federated Learning session completed! The accuracy of the aggregated model is {accuracy_agg2}')
            print(f"Saving round {server_round} model weights...")
            date = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            # 기존 저장 경로/확장자 (Keras 3에서는 .weights.h5로 저장 필요)
            # model.save_weights(f"./workspace/clientResults/round-{server_round}-weights-{date}.h5")
            os.makedirs("./workspace/clientResults", exist_ok=True)  # 저장 폴더 보장
            model.save_weights(f"./workspace/clientResults/round-{server_round}-weights-{date}.weights.h5")

        return aggregated_weights


# Create strategy and run server
strategy = SaveModelStrategy(
    # 단일 클라이언트도 즉시 학습 시작되도록 최소 요구치 설정
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=1,
    min_evaluate_clients=1,
    min_available_clients=1,
    initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights())
)

# on_fit_config_fn으로 클라이언트에 local_epochs 전달
strategy.on_fit_config_fn = fit_config  # 기존: None (클라이언트 기본 1 epoch)

# 연합 라운드 수 증가 (기존 1)
config = ServerConfig(num_rounds=10)

# Windows 환경 호환을 위해 IPv4로 바인딩
fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, config=config)