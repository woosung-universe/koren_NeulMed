import os
import argparse
import warnings
import logging
import pandas as pd
import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K
import flwr as fl

warnings.filterwarnings("ignore")

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------- Strategy (TPU/Default) ----------------
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    logging.info(f"Running on TPU {tpu.master()}")
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

logging.info(f"Number of replicas in sync: {strategy.num_replicas_in_sync}")

# ---------------- Feature Eng ----------------
def train_feature_eng(df: pd.DataFrame) -> pd.DataFrame:
    if "path" not in df.columns:
        # CSV가 image_name 기준일 때 기본 경로 구성
        if "image_name" not in df.columns:
            raise ValueError("CSV must contain either 'path' or 'image_name' column.")
        # 실제 존재하는 훈련 이미지 디렉토리 자동 탐색
        possible_train_dirs = [
            "./isicdata/train/",
            "./isicdata/train/train/",
            "./isicdata/images/train/",
            "./isicdata/datasets/train/",
        ]
        base_dir = None
        for d in possible_train_dirs:
            if tf.io.gfile.exists(d):
                try:
                    if tf.io.gfile.glob(os.path.join(d, "*.jpg")):
                        base_dir = d
                        break
                except Exception:
                    pass
        if base_dir is None:
            logging.warning("Could not auto-detect train image directory. Falling back to ./isicdata/train/")
            base_dir = "./isicdata/train/"
        df["path"] = df["image_name"].apply(lambda n: os.path.join(base_dir, f"{n}.jpg"))

    # 누락 컬럼 대비
    if "anatom_site_general_challenge" in df.columns:
        df["anatom_site_general_challenge"] = df["anatom_site_general_challenge"].fillna("torso")
    if "sex" in df.columns:
        df["sex"] = df["sex"].fillna("male")
    if "age_approx" in df.columns:
        df["age_approx"] = df["age_approx"].fillna(df["age_approx"].mean())

    # 필수 타깃 컬럼 존재 체크
    if "target" not in df.columns:
        raise ValueError("CSV must contain 'target' column (0/1).")

    return df

# ---------------- Loss ----------------
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1.0 - pt_1, gamma) * K.log(pt_1)) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1.0 - pt_0)
        )
    return focal_loss_fixed

# ---------------- Model ----------------
def load_model():
    IMAGE_SIZE = (384, 384)
    with strategy.scope():
        model = tf.keras.Sequential(
            [
                efn.EfficientNetB2(input_shape=(*IMAGE_SIZE, 3), weights=None, include_top=False),
                L.GlobalAveragePooling2D(),
                L.Dense(1024, activation="relu"),
                L.Dropout(0.3),
                L.Dense(512, activation="relu"),
                L.Dropout(0.2),
                L.Dense(256, activation="relu"),
                L.Dropout(0.2),
                L.Dense(128, activation="relu"),
                L.Dropout(0.1),
                L.Dense(1, activation="sigmoid"),
            ]
        )
        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        model.compile(
            optimizer=opt,
            loss=focal_loss(gamma=2.0, alpha=0.25),
            metrics=["binary_crossentropy", "accuracy"],
        )
        # 로컬 가중치가 있으면 우선 시도 로드
        for p in [
            "./melamodel/melamodel_weights072.h5",
            "./melamodel/melamodel_weights072.weights.h5",
        ]:
            if os.path.exists(p):
                try:
                    model.load_weights(p)
                    logging.info(f"Loaded local weights: {p}")
                    break
                except Exception as e:
                    logging.warning(f"Failed to load {p}: {e}")
    logging.info("Model compiled")
    return model

# ---------------- Data Pipeline ----------------
AUTOTUNE = tf.data.AUTOTUNE

def _decode_resize(path, label, img_size=(384, 384)):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)        # 중요: 3채널 강제
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, img_size, antialias=True)
    return img, label

def prepare_train_df(df, mela_count=None, benign_count=None, train_ratio=0.8, batch_size=8):
    df = train_feature_eng(df)

    # 실제 존재하는 파일만 사용
    exists = df["path"].apply(lambda p: tf.io.gfile.exists(p))
    missing = df.loc[~exists, "path"]
    if len(missing) > 0:
        logging.warning(f"Missing files: {len(missing)} (first 5) -> {missing.head().tolist()}")
    df = df.loc[exists].copy()
    if df.empty:
        raise ValueError("No valid image files found from 'path' column.")

    # 클래스별 샘플링 (기존 코드 보존)
    # df_mela = df[df["target"] == 1].sample(frac=1, random_state=42)[: min(mela_count, (df["target"] == 1).sum())]
    # df_ben  = df[df["target"] == 0].sample(frac=1, random_state=42)[: min(benign_count, (df["target"] == 0).sum())]
    # df2 = pd.concat([df_mela, df_ben], ignore_index=True).sample(frac=1, random_state=42)

    # 변경: 기본은 전체 데이터 사용. mela_count/benign_count가 지정되면 샘플링
    if mela_count is None and benign_count is None:
        df2 = df.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        _m = (df["target"] == 1).sum()
        _b = (df["target"] == 0).sum()
        m_lim = _m if mela_count is None else min(mela_count, _m)
        b_lim = _b if benign_count is None else min(benign_count, _b)
        df_mela = df[df["target"] == 1].sample(n=m_lim, random_state=42)
        df_ben  = df[df["target"] == 0].sample(n=b_lim, random_state=42)
        df2 = pd.concat([df_mela, df_ben], ignore_index=True).sample(frac=1, random_state=42)

    # train/valid split
    slice_num = int(len(df2) * float(train_ratio))
    train_df = df2.iloc[:slice_num]
    valid_df = df2.iloc[slice_num:] if slice_num < len(df2) else df2.iloc[-1:]  # 최소 1개 보장

    n_train = len(train_df)
    n_valid = len(valid_df)

    # tf.data
    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_df["path"].values, train_df["target"].values))
        .shuffle(buffer_size=max(1024, n_train))
        .map(lambda x, y: _decode_resize(x, y, (384, 384)), num_parallel_calls=AUTOTUNE)
        .batch(batch_size, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )
    valid_ds = (
        tf.data.Dataset.from_tensor_slices((valid_df["path"].values, valid_df["target"].values))
        .map(lambda x, y: _decode_resize(x, y, (384, 384)), num_parallel_calls=AUTOTUNE)
        .batch(batch_size, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    return train_ds, valid_ds, batch_size, slice_num, n_train, n_valid

# ---------------- Flower Client ----------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, test_dataset, batch_size, slice_num, n_train, n_valid):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.n_train = n_train
        self.n_valid = n_valid

    def get_parameters(self, config=None):
        # 서버 초기 파라미터 동기화를 위해 가중치 반환
        return self.model.get_weights()

    def fit(self, parameters, config=None):
        self.model.set_weights(parameters)
        # 기존 기본값 1epoch → 서버 on_fit_config에서 전달(없으면 3으로 기본)
        epochs = int(config.get("local_epochs", 3)) if config else 3
        hist = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.test_dataset,
            class_weight={0: 0.025, 1: 11},
            verbose=1,
        )
        acc = float(hist.history.get("accuracy", [0])[-1])
        logging.info(f"Client fit done. acc={acc:.4f}, train_samples={self.n_train}")
        return self.model.get_weights(), self.n_train, {"accuracy": acc}

    def evaluate(self, parameters, config=None):
        self.model.set_weights(parameters)
        loss, bce, acc = self.model.evaluate(self.test_dataset, verbose=0)
        logging.info(f"Client eval done. loss={loss:.4f}, acc={acc:.4f}, valid_samples={self.n_valid}")
        return float(loss), self.n_valid, {"accuracy": float(acc)}

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--path", type=str, default="./isicdata/datasets/doctor_case2.csv", help="Path to the dataset CSV")
    args = parser.parse_args()

    logging.info(f"Loading dataset from {args.path}")
    df = pd.read_csv(args.path)

    # 기존: 극단적 샘플링 값 사용
    # train_ds, valid_ds, batch_size, slice_num, n_train, n_valid = prepare_train_df(
    #     df, mela_count=6, benign_count=43, train_ratio=0.8, batch_size=8
    # )
    # 변경: 기본 전체 데이터 사용 (필요하면 --mela_count/--benign_count로 제한)
    train_ds, valid_ds, batch_size, slice_num, n_train, n_valid = prepare_train_df(
        df, mela_count=None, benign_count=None, train_ratio=0.8, batch_size=8
    )

    model = load_model()
    client = FlowerClient(model, train_ds, valid_ds, batch_size, slice_num, n_train, n_valid)

    logging.info("Starting Flower NumPy client...")
    # Windows 호환: IPv4로 접속
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Unhandled exception in client")
        raise