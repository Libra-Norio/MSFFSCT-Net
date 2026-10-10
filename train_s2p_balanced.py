# train_s2p_balanced.py
# Seq2Point Balanced 模型训练，训练房屋1,3,4,5，测试房屋2，学习率0.001
# 评估时使用固定阈值（不搜索）

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import f1_score
import random

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEED = 2024
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

from UKDALE_Parser import UK_Dale_Parser
from config import get_args, update_preprocessing_parameters
from model_structure_seq2point_balanced import create_seq2point_balanced_model

# ========== 超参数 ==========
UKDALE_DATA_ROOT = "./data/UK_DALE"
OUTPUT_BASE = "./s2p_balanced_output"

WINDOW_SIZE = 480
STEP = 240
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

APPLIANCES = ['microwave', 'dishwasher', 'fridge', 'kettle', 'washing_machine']
TRAIN_HOUSES = [1, 3, 4, 5]
TEST_HOUSES = [2]

APPLIANCE_PARAMS = {
    'fridge':           {'pos_ratio': 0.5, 'weight_power': 1.5, 'status_weight': 1.0},
    'kettle':           {'pos_ratio': 0.5, 'weight_power': 1.0, 'status_weight': 1.0},
    'washing_machine':  {'pos_ratio': 0.55, 'weight_power': 2.0, 'status_weight': 2.0},
    'microwave':        {'pos_ratio': 0.65, 'weight_power': 2.5, 'status_weight': 3.0},
    'dishwasher':       {'pos_ratio': 0.5, 'weight_power': 1.5, 'status_weight': 2.0},
}
# =============================================

def compute_metrics(y_true, y_pred, threshold):
    mae = np.mean(np.abs(y_true - y_pred))
    sum_true = np.sum(y_true) + 1e-8
    nep = np.sum(np.abs(y_true - y_pred)) / sum_true
    bin_true = (y_true > threshold).astype(int)
    bin_pred = (y_pred > threshold).astype(int)
    f1 = f1_score(bin_true, bin_pred, zero_division=0)
    return {'f1': float(f1), 'mae': float(mae), 'nep': float(nep)}

def get_appliance_params(appliance):
    args = get_args()
    args.dataset_code = 'uk_dale'
    args = update_preprocessing_parameters(args)
    return args.cutoff[appliance], args.threshold[appliance]

def generate_windows(mains, target, window_size, step):
    mains = mains.flatten()
    target = target.flatten()
    half = window_size // 2
    X, y = [], []
    for start in range(0, len(mains) - window_size + 1, step):
        X.append(mains[start:start+window_size])
        y.append(target[start+half])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def oversample_windows(X, y, target_ratio, threshold):
    np.random.seed(SEED)
    if target_ratio <= 0 or len(X) == 0:
        return X, y
    pos_idx = np.where(y > threshold)[0]
    neg_idx = np.where(y <= threshold)[0]
    if len(pos_idx) == 0:
        return X, y
    total_neg = len(neg_idx)
    desired_pos = int(total_neg * target_ratio / (1 - target_ratio))
    if desired_pos <= len(pos_idx):
        return X, y
    dup_count = desired_pos - len(pos_idx)
    dup_idx = np.random.choice(pos_idx, size=dup_count, replace=True)
    X_rs = np.vstack([X, X[dup_idx]])
    y_rs = np.concatenate([y, y[dup_idx]])
    shuffle = np.random.permutation(len(X_rs))
    return X_rs[shuffle], y_rs[shuffle]

def main():
    output_root = Path(OUTPUT_BASE)
    output_root.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for appliance in APPLIANCES:
        print(f"\n>>>> 处理电器: {appliance} <<<<")
        max_power, on_threshold = get_appliance_params(appliance)
        conf = APPLIANCE_PARAMS[appliance]

        args = get_args()
        args.dataset_code = 'uk_dale'
        args = update_preprocessing_parameters(args)
        args.ukdale_location = UKDALE_DATA_ROOT
        args.appliance_names = [appliance]
        args.house_indicies = TRAIN_HOUSES
        parser_train = UK_Dale_Parser(args)
        X_train_raw, y_train_raw = generate_windows(parser_train.x, parser_train.y, WINDOW_SIZE, STEP)
        print(f"  训练集窗口数: {len(X_train_raw)}")

        split = int(0.85 * len(X_train_raw))
        X_train, X_val = X_train_raw[:split], X_train_raw[split:]
        y_train, y_val = y_train_raw[:split], y_train_raw[split:]
        print(f"  切分后训练集: {len(X_train)}, 验证集: {len(X_val)}")

        args.house_indicies = TEST_HOUSES
        parser_test = UK_Dale_Parser(args, (parser_train.x_mean, parser_train.x_std))
        X_test, y_test_true = generate_windows(parser_test.x, parser_test.y, WINDOW_SIZE, STEP)
        print(f"  测试集窗口数: {len(X_test)}")

        y_train_p = y_train / max_power
        y_val_p = y_val / max_power
        y_train_s = (y_train > on_threshold).astype(np.float32)
        y_val_s = (y_val > on_threshold).astype(np.float32)

        X_train_bal, y_train_bal = oversample_windows(X_train, y_train, conf['pos_ratio'], on_threshold)
        y_train_p = y_train_bal / max_power
        y_train_s = (y_train_bal > on_threshold).astype(np.float32)

        pos_count = np.sum(y_train_s == 1)
        neg_count = np.sum(y_train_s == 0)
        pos_weight = (neg_count / pos_count) ** conf['weight_power'] if pos_count > 0 else 1.0
        sample_weight = np.where(y_train_s == 1, pos_weight, 1.0).astype(np.float32)
        print(f"  正样本数: {pos_count}, 负样本数: {neg_count}, 权重: {pos_weight:.2f}")

        # ---- 构建 Seq2Point Balanced 模型 ----
        model = create_seq2point_balanced_model(input_window_length=WINDOW_SIZE)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss={'power_output': 'mse', 'status_output': 'binary_crossentropy'},
            loss_weights={'power_output': 1.0, 'status_output': conf['status_weight']},
            metrics={'power_output': 'mae'}
        )

        model_dir = output_root / appliance
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "best_model.keras"
        weight_path = model_dir / "best_weights.weights.keras"

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(str(model_path), save_best_only=True,
                                               monitor='val_power_output_mae', mode='min'),
            tf.keras.callbacks.ModelCheckpoint(str(weight_path), save_best_only=True,
                                               monitor='val_power_output_mae', mode='min', save_weights_only=True)
        ]

        print("  开始训练 Seq2Point Balanced 模型...")
        history = model.fit(
            X_train_bal,
            {'power_output': y_train_p, 'status_output': y_train_s},
            validation_data=(X_val, {'power_output': y_val_p, 'status_output': y_val_s}),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=2,
            sample_weight={'power_output': sample_weight},
            callbacks=callbacks
        )

        if model_path.exists():
            model = tf.keras.models.load_model(str(model_path))

        preds = model.predict(X_test, verbose=0)
        y_pred_power = preds[0].flatten() * max_power

        val_preds = model.predict(X_val, verbose=0)
        y_val_pred_power = val_preds[0].flatten() * max_power

        # ★★★ 使用固定阈值（不搜索）★★★
        best_t = on_threshold
        print(f"  使用固定阈值: {best_t:.2f}W")

        metrics = compute_metrics(y_test_true, y_pred_power, best_t)
        metrics['best_threshold'] = float(best_t)
        all_results[appliance] = metrics

        with open(model_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        pd.DataFrame(history.history).to_csv(model_dir / "history.csv")
        print(f"  {appliance}: F1={metrics['f1']:.4f}, MAE={metrics['mae']:.2f}W, NEP={metrics['nep']:.4f}")

    with open(output_root / "summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n全部完成！结果保存在:", OUTPUT_BASE)
    for app, met in all_results.items():
        print(f"{app}: F1={met['f1']:.4f}, MAE={met['mae']:.2f}W, NEP={met['nep']:.4f}")

if __name__ == "__main__":
    main()