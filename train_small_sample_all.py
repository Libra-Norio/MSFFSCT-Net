# train_small_sample_all.py
# 小样本实验：50%和100%训练数据，所有5个电器
# 固定阈值，每个比例跑3次取平均

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
from model_structure_s2p_cbam_transformer import (
    create_light_transformer_model, CBAM1D, StatFeatureExtractor, LightTransformer
)

CUSTOM_OBJECTS = {
    'CBAM1D': CBAM1D,
    'StatFeatureExtractor': StatFeatureExtractor,
    'LightTransformer': LightTransformer,
}

UKDALE_DATA_ROOT = "./data/UK_DALE"
OUTPUT_BASE = "./small_sample_all"

WINDOW_SIZE = 480
STEP = 240
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# ★ 全部5个电器
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

# ★ 只跑50%和100%
RATIOS = [0.5, 1.0]
N_RUNS = 3   # 每个比例跑3次


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
        X.append(mains[start:start + window_size])
        y.append(target[start + half])
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


def subsample(X, y, ratio, seed):
    """从训练集中随机采样一定比例"""
    np.random.seed(seed)
    n = len(X)
    n_sub = max(1, int(n * ratio))
    idx = np.random.choice(n, n_sub, replace=False)
    return X[idx], y[idx]


def main():
    output_root = Path(OUTPUT_BASE)
    output_root.mkdir(parents=True, exist_ok=True)
    all_results = []

    for appliance in APPLIANCES:
        print("\n" + "#" * 70)
        print(f"# 处理电器: {appliance}")
        print("#" * 70)

        max_power, on_threshold = get_appliance_params(appliance)
        conf = APPLIANCE_PARAMS[appliance]
        print(f"  cutoff={max_power}W, threshold={on_threshold}W")

        # ---- 加载数据（每个电器只加载一次）----
        args = get_args()
        args.dataset_code = 'uk_dale'
        args = update_preprocessing_parameters(args)
        args.ukdale_location = UKDALE_DATA_ROOT
        args.appliance_names = [appliance]
        args.house_indicies = TRAIN_HOUSES
        parser_train = UK_Dale_Parser(args)
        X_train_raw, y_train_raw = generate_windows(parser_train.x, parser_train.y, WINDOW_SIZE, STEP)

        split = int(0.85 * len(X_train_raw))
        X_train_full, X_val = X_train_raw[:split], X_train_raw[split:]
        y_train_full, y_val = y_train_raw[:split], y_train_raw[split:]

        args.house_indicies = TEST_HOUSES
        parser_test = UK_Dale_Parser(args, (parser_train.x_mean, parser_train.x_std))
        X_test, y_test_true = generate_windows(parser_test.x, parser_test.y, WINDOW_SIZE, STEP)

        y_val_p = y_val / max_power
        y_val_s = (y_val > on_threshold).astype(np.float32)

        print(f"  训练集(全量): {len(X_train_full)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
        print(f"  真实>阈值点数(测试集): {(y_test_true > on_threshold).sum()}")

        # ---- 遍历比例 ----
        for ratio in RATIOS:
            print(f"\n  ---- 训练数据比例: {ratio*100:.0f}% ----")

            for run in range(N_RUNS):
                seed = SEED + run
                print(f"    Run {run+1}/{N_RUNS} (seed={seed})")

                # 子采样
                X_train_sub, y_train_sub = subsample(X_train_full, y_train_full, ratio, seed)
                print(f"      子采样后训练集: {len(X_train_sub)}")

                # 过采样
                X_train_bal, y_train_bal = oversample_windows(
                    X_train_sub, y_train_sub, conf['pos_ratio'], on_threshold)
                y_train_p = y_train_bal / max_power
                y_train_s = (y_train_bal > on_threshold).astype(np.float32)

                pos = np.sum(y_train_s == 1)
                neg = np.sum(y_train_s == 0)
                pos_weight = (neg / pos) ** conf['weight_power'] if pos > 0 else 1.0
                sample_weight = np.where(y_train_s == 1, pos_weight, 1.0).astype(np.float32)

                print(f"      过采样后: 正样本={pos}, 负样本={neg}, 权重={pos_weight:.2f}")

                # 固定随机种子
                random.seed(seed)
                np.random.seed(seed)
                tf.random.set_seed(seed)

                # 构建模型
                model = create_light_transformer_model(window_size=WINDOW_SIZE, hidden=64)

                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss={'power_output': 'mse', 'status_output': 'binary_crossentropy'},
                    loss_weights={'power_output': 1.0, 'status_output': conf['status_weight']},
                    metrics={'power_output': 'mae'}
                )

                model_dir = output_root / appliance / f"ratio_{ratio}_run_{run}"
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / "best_model.keras"

                callbacks = [
                    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                    tf.keras.callbacks.ModelCheckpoint(str(model_path), save_best_only=True,
                                                       monitor='val_power_output_mae', mode='min',
                                                       overwrite=True)
                ]

                model.fit(
                    X_train_bal,
                    {'power_output': y_train_p, 'status_output': y_train_s},
                    validation_data=(X_val, {'power_output': y_val_p, 'status_output': y_val_s}),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=0,
                    sample_weight={'power_output': sample_weight},
                    callbacks=callbacks
                )

                # 加载最优模型
                if model_path.exists():
                    model = tf.keras.models.load_model(
                        str(model_path), custom_objects=CUSTOM_OBJECTS, compile=False)

                # 测试
                preds = model.predict(X_test, verbose=0)
                y_pred_power = preds[0].flatten() * max_power

                metrics = compute_metrics(y_test_true, y_pred_power, on_threshold)

                all_results.append({
                    'appliance': appliance,
                    'ratio': ratio,
                    'run': run,
                    'n_train': len(X_train_sub),
                    'f1': metrics['f1'],
                    'mae': metrics['mae'],
                    'nep': metrics['nep'],
                })

                print(f"      F1={metrics['f1']:.4f}, MAE={metrics['mae']:.2f}W, NEP={metrics['nep']:.4f}")

    # ---- 保存原始结果 ----
    df = pd.DataFrame(all_results)
    df.to_csv(output_root / "small_sample_results.csv", index=False)

    # ---- 汇总（按电器和比例取平均）----
    summary = df.groupby(['appliance', 'ratio']).agg(
        f1_mean=('f1', 'mean'),
        f1_std=('f1', 'std'),
        mae_mean=('mae', 'mean'),
        mae_std=('mae', 'std'),
        nep_mean=('nep', 'mean'),
    ).reset_index()
    summary.to_csv(output_root / "small_sample_summary.csv", index=False)

    print("\n" + "=" * 90)
    print("小样本实验汇总结果")
    print("=" * 90)
    print(summary.to_string(index=False))

    # ---- 分设备打印 ----
    print("\n【各设备在不同数据量下的F1】")
    f1_pivot = summary.pivot_table(index='appliance', columns='ratio',
                                    values='f1_mean', aggfunc='first')
    print(f1_pivot.to_string())

    print("\n【各设备在不同数据量下的MAE】")
    mae_pivot = summary.pivot_table(index='appliance', columns='ratio',
                                     values='mae_mean', aggfunc='first')
    print(mae_pivot.to_string())

    print(f"\n结果已保存到: {output_root}")


if __name__ == "__main__":
    main()