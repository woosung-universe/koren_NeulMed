# # eval_fed.py  (venv311에서 실행)
# import os, glob, json, pandas as pd, tensorflow as tf
# import efficientnet.tfkeras as efn
# from tensorflow.keras import layers as L

# def detect_train_dir():
#     for d in ["./isicdata/train/", "./isicdata/train/train/", "./isicdata/images/train/", "./isicdata/datasets/train/"]:
#         if tf.io.gfile.exists(d) and tf.io.gfile.glob(os.path.join(d, "*.jpg")):
#             return d
#     raise FileNotFoundError("훈련 이미지 디렉토리를 찾지 못했습니다.")

# def _decode_resize(path, label, img_size=(384,384)):
#     img = tf.io.read_file(path)
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.image.convert_image_dtype(img, tf.float32)
#     img = tf.image.resize(img, img_size, antialias=True)
#     return img, label

# def normalize_label(ds):
#     return ds.map(lambda x, y: (x, tf.expand_dims(tf.squeeze(tf.cast(y, tf.float32)), -1)))

# def make_valid(csv_path="./isicdata/datasets/doctor_case2.csv", batch_size=8):
#     train_dir = detect_train_dir()
#     df = pd.read_csv(csv_path)
#     if "path" not in df.columns:
#         df["path"] = df["image_name"].apply(lambda n: os.path.join(train_dir, f"{n}.jpg"))
#     exists = df["path"].apply(lambda p: tf.io.gfile.exists(p))
#     df = df.loc[exists].reset_index(drop=True)
#     ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["target"].values))
#     ds = ds.map(lambda x, y: _decode_resize(x, y), num_parallel_calls=tf.data.AUTOTUNE)
#     ds = normalize_label(ds).batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     return ds

# def build_fed_model():
#     m = tf.keras.Sequential([
#         efn.EfficientNetB2(input_shape=(384,384,3), weights=None, include_top=False),
#         L.GlobalAveragePooling2D(name='global_average_pooling2d'),
#         L.Dense(1024, activation='relu', name='dense'),
#         L.Dropout(0.3, name='dropout'),
#         L.Dense(512, activation='relu', name='dense_1'),
#         L.Dropout(0.2, name='dropout_1'),
#         L.Dense(256, activation='relu', name='dense_2'),
#         L.Dropout(0.2, name='dropout_2'),
#         L.Dense(128, activation='relu', name='dense_3'),
#         L.Dropout(0.1, name='dropout_3'),
#         L.Dense(1, activation='sigmoid', name='dense_4'),
#     ])
#     m.compile(optimizer='Adam', loss='binary_crossentropy',
#               metrics=['binary_crossentropy','accuracy'], run_eagerly=True)
#     return m

# if __name__ == "__main__":
#     valid_ds = make_valid("./isicdata/datasets/doctor_case2.csv", batch_size=8)
#     fed_w_fixed = r"C:\Users\USER\Desktop\koren\koren_NeulMed\Melanoma-Classifier-Federated-Learning\workspace\clientResults\round-10-weights-2025_09_20-21_58_41.weights.h5"
#     if os.path.exists(fed_w_fixed):
#         fed_w = fed_w_fixed
#     else:
#         files = glob.glob("./workspace/clientResults/round-*-weights-*.weights.h5")
#         fed_w = max(files, key=os.path.getmtime) if files else None

#     if not fed_w:
#         raise FileNotFoundError("연합 가중치 파일을 찾지 못했습니다.")

#     model = build_fed_model()
#     model.load_weights(fed_w)
#     loss, bce, acc = model.evaluate(valid_ds, steps=1, verbose=1)
#     print("[연합 후] loss=%.4f bce=%.4f acc=%.4f" % (loss, bce, acc))

#     os.makedirs("./eval_out", exist_ok=True)
#     with open("./eval_out/fed_eval.json", "w", encoding="utf-8") as f:
#         json.dump({"loss": float(loss), "bce": float(bce), "acc": float(acc), "weights": fed_w}, f, indent=2, ensure_ascii=False)

# eval.py
import os, glob, json, argparse, logging, re
import pandas as pd
import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow.keras import layers as L

logging.getLogger().setLevel(logging.INFO)
AUTOTUNE = tf.data.AUTOTUNE

def detect_train_dir():
    for d in ["./isicdata/train/", "./isicdata/train/train/", "./isicdata/images/train/", "./isicdata/datasets/train/"]:
        if tf.io.gfile.exists(d) and tf.io.gfile.glob(os.path.join(d, "*.jpg")):
            logging.info(f"Detected train image dir: {d}")
            return d
    raise FileNotFoundError("훈련 이미지 디렉토리를 찾지 못했습니다.")

def _decode_resize(path, label, img_size=(384,384)):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, img_size, antialias=True)
    return img, label

def normalize_label(ds):
    # (None,), (None,1,1) 등 → (None,1)
    return ds.map(lambda x, y: (x, tf.expand_dims(tf.squeeze(tf.cast(y, tf.float32)), -1)))

def make_valid(csv_path="./isicdata/datasets/doctor_case2.csv", batch_size=8):
    td = detect_train_dir()
    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        if "image_name" not in df.columns:
            raise ValueError("CSV에 'path' 또는 'image_name' 컬럼이 필요합니다.")
        df["path"] = df["image_name"].apply(lambda n: os.path.join(td, f"{n}.jpg"))
    df = df[df["path"].apply(lambda p: tf.io.gfile.exists(p))].reset_index(drop=True)
    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["target"].values))
    ds = ds.map(lambda x, y: _decode_resize(x, y), num_parallel_calls=AUTOTUNE)
    ds = normalize_label(ds).batch(batch_size).prefetch(AUTOTUNE)
    logging.info(f"n_valid={len(df)} (csv={csv_path})")
    return ds, df

# 연합 전(베이스) 가중치: 파일이 가리키는 헤드는 GAP→Dense(1)인 경우가 흔했음
def build_base_min_head():
    m = tf.keras.Sequential([
        efn.EfficientNetB2(input_shape=(224,224,3), weights=None, include_top=False),  # 224로 축소!
        L.GlobalAveragePooling2D(name="global_average_pooling2d"),
        L.Dense(1, activation="sigmoid", name="dense"),
    ])
    m.compile(optimizer="Adam", loss="binary_crossentropy",
              metrics=["binary_crossentropy","accuracy"], run_eagerly=True)
    return m

# 연합 후(서버/클라) 헤드
def build_fed_head():
    m = tf.keras.Sequential([
        efn.EfficientNetB2(input_shape=(224,224,3), weights=None, include_top=False),  # 224로 축소!
        L.GlobalAveragePooling2D(name='global_average_pooling2d'),
        L.Dense(1024, activation='relu', name='dense'),
        L.Dropout(0.3, name='dropout'),
        L.Dense(512, activation='relu', name='dense_1'),
        L.Dropout(0.2, name='dropout_1'),
        L.Dense(256, activation='relu', name='dense_2'),
        L.Dropout(0.2, name='dropout_2'),
        L.Dense(128, activation='relu', name='dense_3'),
        L.Dropout(0.1, name='dropout_3'),
        L.Dense(1, activation='sigmoid', name='dense_4'),
    ])
    m.compile(optimizer='Adam', loss='binary_crossentropy',
              metrics=['binary_crossentropy','accuracy'], run_eagerly=True)
    return m

def eval_model(model, ds):
    # steps 지정하지 않음 → 전체 검증셋 사용
    loss, bce, acc = model.evaluate(ds, verbose=1)
    return {"loss": float(loss), "bce": float(bce), "acc": float(acc)}

def try_eval_baseline(ds, baseline_weights):
    # 1) 파일 지정 시도 (GAP→Dense(1) 헤드 가정)
    if baseline_weights and os.path.exists(baseline_weights):
        try:
            m = build_base_min_head()
            m.load_weights(baseline_weights)
            logging.info(f"[BASE] 가중치 로드: {baseline_weights}")
            return eval_model(m, ds) | {"source": "weights", "path": baseline_weights}
        except Exception as e:
            logging.warning(f"[BASE] 가중치 로드 실패: {e}")

    # 2) 베이스 성능 json에서 정량만 읽기 (test_accuracy 우선, 없으면 train_accuracy)
    perf_files = sorted(glob.glob("./base_model_performance_*.json"), key=os.path.getmtime, reverse=True)
    if perf_files:
        try:
            with open(perf_files[0], "r", encoding="utf-8") as f:
                perf = json.load(f)
            acc = perf.get("base_model", {}).get("test_accuracy") or perf.get("base_model", {}).get("train_accuracy")
            if acc is not None:
                logging.info(f"[BASE] json 사용: {perf_files[0]} acc={acc:.4f}")
                return {"loss": None, "bce": None, "acc": float(acc), "source": "json", "path": perf_files[0]}
        except Exception as e:
            logging.warning(f"[BASE] json 파싱 실패: {e}")

    logging.warning("[BASE] 베이스라인을 산출하지 못했습니다.")
    return {"loss": None, "bce": None, "acc": None, "source": "none", "path": None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./isicdata/datasets/doctor_case2.csv")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--baseline_weights", default=r"C:\Users\USER\Desktop\koren\koren_NeulMed\Melanoma-Classifier-Federated-Learning\melamodel\melamodel_weights072.weights.h5")
    ap.add_argument("--fed_weights", default=r"C:\Users\USER\Desktop\koren\koren_NeulMed\Melanoma-Classifier-Federated-Learning\workspace\clientResults\round-10-weights-2025_09_20-21_58_41.weights.h5")
    args = ap.parse_args()

    # 데이터
    valid_ds, valid_df = make_valid(args.csv, batch_size=args.batch)

    # 베이스라인 평가(가중치 or json)
    base = try_eval_baseline(valid_ds, args.baseline_weights)

    # 연합 후 평가
    fed_w = args.fed_weights
    if not os.path.exists(fed_w):
        files = glob.glob("./workspace/clientResults/round-*-weights-*.weights.h5")
        fed_w = max(files, key=os.path.getmtime) if files else None
    if not fed_w:
        raise FileNotFoundError("연합 가중치 파일을 찾지 못했습니다.")

    fed_model = build_fed_head()
    fed_model.load_weights(fed_w)
    fed = eval_model(fed_model, valid_ds) | {"path": fed_w}

    # 요약 출력
    def pct(x): return f"{x*100:.2f}%"
    print("\n=== 전/후 성능 비교(동일 검증셋) ===")
    if base["acc"] is not None:
        print(f"- 연합 전 정확도: {pct(base['acc'])}  (source={base['source']})")
    else:
        print(f"- 연합 전 정확도: N/A")
    print(f"- 연합 후 정확도: {pct(fed['acc'])}")
    if base["acc"] is not None:
        print(f"- 차이(후-전):     {pct(fed['acc'] - base['acc'])}")

    # 저장
    out = {
        "baseline": base,
        "federated": fed,
        "csv": args.csv,
        "n_valid": int(len(valid_df)),
    }
    os.makedirs("./eval_out", exist_ok=True)
    with open("./eval_out/compare.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSaved: ./eval_out/compare.json")

if __name__ == "__main__":
    main()