# train_s2p_redd_fixed.py
# 纯 S2P 双输出（功率 + 状态），REDD 数据集
# 训练房屋 2‑6，测试房屋 1
# 包含过采样、固定阈值（无验证集搜索）、固定随机种子

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import f1_score
import random

# ---------- 环境与确定性 ----------
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEED = 2024
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

from REDD_Parser import Redd_Parser
from config import get_args, update_preprocessing_parameters
from model_structure_s2p import create_s2p_dual_model

# ---------- 超参数 ----------
REDD_DATA_ROOT = "./data/REDD"
OUTPUT_BASE = "./s2p_redd_fixed"

WINDOW_SIZE = 480
STEP = 240
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# 注意：REDD 标签中 dishwasher 写为 dishwaser，但 config 中键为 dishwasher
APPLIANCES = ['refrigerator', 'washer_dryer', 'microwave', 'dishwaser']
TRAIN_HOUSES = [2, 3, 4, 5, 6]
TEST_HOUSES = [1]

# ---------- 每个电器的过采样/损失参数 ----------
APPLIANCE_PARAMS = {
    'refrigerator':   {'pos_ratio': 0.5,  'weight_power': 1.5, 'status_weight': 1.0},
    'washer_dryer':   {'pos_ratio': 0.55, 'weight_power': 2.0, 'status_weight': 2.0},
    'microwave':      {'pos_ratio': 0.65, 'weight_power': 2.5, 'status_weight': 3.0},
    'dishwaser':      {'pos_ratio': 0.5,  'weight_power': 1.5, 'status_weight': 2.0},
}

# ---------- 工具函数 ----------
def compute_metrics(y_true, y_pred, threshold):
    mae = np.mean(np.abs(y_true - y_pred))
    sum_true = np.sum(y_true) + 1e-8
    nep = np.sum(np.abs(y_true - y_pred)) / sum_true
    bin_true = (y_true > threshold).astype(int)
    bin_pred = (y_pred > threshold).astype(int)
    f1 = f1_score(bin_true, bin_pred, zero_division=0)
    return {'f1': float(f1), 'mae': float(mae), 'nep': float(nep)}

def get_appliance_params(appliance):
    # 将 dishwaser 映射为 dishwasher 以读取 config 中的参数
    if appliance == 'dishwaser':
        key = 'dishwasher'
    else:
        key = appliance
    args = get_args()
    args.dataset_code = 'redd_lf'
    args = update_preprocessing_parameters(args)
    return args.cutoff[key], args.threshold[key]

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

# ---------- 主训练循环 ----------
def main():
    output_root = Path(OUTPUT_BASE)
    output_root.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for appliance in APPLIANCES:
        print(f"\n>>>> 处理电器: {appliance} <<<<")
        max_power, on_threshold = get_appliance_params(appliance)
        conf = APPLIANCE_PARAMS[appliance]

        # ---- 加载训练数据 ----
        args = get_args()
        args.dataset_code = 'redd_lf'
        args = update_preprocessing_parameters(args)
        args.redd_location = REDD_DATA_ROOT
        args.appliance_names = [appliance]
        args.house_indicies = TRAIN_HOUSES
        args.sampling = '6s'
        parser_train = Redd_Parser(args)
        X_train_raw, y_train_raw = generate_windows(parser_train.x, parser_train.y, WINDOW_SIZE, STEP)
        print(f"  训练集窗口数: {len(X_train_raw)}")

        split = int(0.85 * len(X_train_raw))
        X_train, X_val = X_train_raw[:split], X_train_raw[split:]
        y_train, y_val = y_train_raw[:split], y_train_raw[split:]
        print(f"  切分后训练集: {len(X_train)}, 验证集: {len(X_val)}")

        # ---- 加载测试数据 ----
        args.house_indicies = TEST_HOUSES
        parser_test = Redd_Parser(args, stats=(parser_train.x_mean, parser_train.x_std))
        X_test, y_test_true = generate_windows(parser_test.x, parser_test.y, WINDOW_SIZE, STEP)
        print(f"  测试集窗口数: {len(X_test)}")

        # ---- 归一化目标值并生成状态标签 ----
        y_train_p = y_train / max_power
        y_val_p   = y_val   / max_power
        y_train_s = (y_train > on_threshold).astype(np.float32)
        y_val_s   = (y_val   > on_threshold).astype(np.float32)

        # ---- 过采样 ----
        X_train_bal, y_train_bal = oversample_windows(X_train, y_train, conf['pos_ratio'], on_threshold)
        y_train_p = y_train_bal / max_power
        y_train_s = (y_train_bal > on_threshold).astype(np.float32)

        pos_count = np.sum(y_train_s == 1)
        neg_count = np.sum(y_train_s == 0)
        pos_weight = (neg_count / pos_count) ** conf['weight_power'] if pos_count > 0 else 1.0
        sample_weight = np.where(y_train_s == 1, pos_weight, 1.0).astype(np.float32)
        print(f"  正样本数: {pos_count}, 负样本数: {neg_count}, 权重: {pos_weight:.2f}")

        # ---- 构建模型 ----
        model = create_s2p_dual_model(input_window_length=WINDOW_SIZE)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss={'power_output': 'mse', 'status_output': 'binary_crossentropy'},
            loss_weights={'power_output': 1.0, 'status_output': conf['status_weight']},
            metrics={'power_output': 'mae'}
        )

        # ---- 保存路径 ----
        model_dir = output_root / appliance
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "best_model.keras"
        weight_path = model_dir / "best_weights.weights.keras"

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                str(model_path), save_best_only=True,
                monitor='val_power_output_mae', mode='min'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(weight_path), save_best_only=True,
                monitor='val_power_output_mae', mode='min', save_weights_only=True
            )
        ]

        # ---- 训练 ----
        print("  开始训练纯 S2P 模型...")
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

        # ---- 加载最佳模型 ----
        if model_path.exists():
            model = tf.keras.models.load_model(str(model_path))

        # ---- 测试预测 ----
        preds = model.predict(X_test, verbose=0)
        y_pred_power = preds[0].flatten() * max_power

        # ---- ★★★ 使用固定阈值（不进行验证集搜索） ★★★ ----
        best_t = on_threshold
        print(f"  固定阈值: {best_t:.2f}W")

        # ---- 评估 ----
        metrics = compute_metrics(y_test_true, y_pred_power, best_t)
        metrics['threshold'] = float(best_t)   # 记录使用的阈值
        all_results[appliance] = metrics

        with open(model_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        pd.DataFrame(history.history).to_csv(model_dir / "history.csv")
        print(f"  {appliance}: F1={metrics['f1']:.4f}, MAE={metrics['mae']:.2f}W, NEP={metrics['nep']:.4f}")

    # ---- 汇总 ----
    with open(output_root / "summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n全部完成！结果保存在:", OUTPUT_BASE)
    for app, met in all_results.items():
        print(f"{app}: F1={met['f1']:.4f}, MAE={met['mae']:.2f}W, NEP={met['nep']:.4f}")

if __name__ == "__main__":
    main()