# train_hparam_washing.py
# 只跑洗衣机，7个配置，verbose=1看训练过程

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
from model_structure_hparam import (
    create_hparam_model, CBAM1D, StatFeatureExtractor, LightTransformer
)

CUSTOM_OBJECTS = {
    'CBAM1D': CBAM1D,
    'StatFeatureExtractor': StatFeatureExtractor,
    'LightTransformer': LightTransformer,
}

UKDALE_DATA_ROOT = "./data/UK_DALE"
OUTPUT_BASE = "./hparam_washing"

WINDOW_SIZE = 480
STEP = 240
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# ★ 只跑洗衣机
APPLIANCES = ['washing_machine']
TRAIN_HOUSES = [1, 3, 4, 5]
TEST_HOUSES = [2]

APPLIANCE_PARAMS = {
    'washing_machine':  {'pos_ratio': 0.55, 'weight_power': 2.0, 'status_weight': 2.0},
}

HPARAM_CONFIGS = [
    {'name': 'L1_H4_D64',  'num_layers': 1, 'num_heads': 4, 'hidden': 64},
    {'name': 'L2_H4_D64',  'num_layers': 2, 'num_heads': 4, 'hidden': 64},
    {'name': 'L3_H4_D64',  'num_layers': 3, 'num_heads': 4, 'hidden': 64},
    {'name': 'L1_H2_D64',  'num_layers': 1, 'num_heads': 2, 'hidden': 64},
    {'name': 'L1_H8_D64',  'num_layers': 1, 'num_heads': 8, 'hidden': 64},
    {'name': 'L1_H4_D32',  'num_layers': 1, 'num_heads': 4, 'hidden': 32},
    {'name': 'L1_H4_D128', 'num_layers': 1, 'num_heads': 4, 'hidden': 128},
]


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


def run_one_config(appliance, cfg, X_train_bal, y_train_p, y_train_s, sample_weight,
                   X_val, y_val_p, y_val_s, X_test, y_test_true, max_power, on_threshold, conf):
    cfg_name = cfg['name']
    print(f"\n  >>> 配置: {cfg_name}  (L={cfg['num_layers']}, "
          f"H={cfg['num_heads']}, D={cfg['hidden']})")

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # 构建模型
    model = create_hparam_model(
        window_size=WINDOW_SIZE,
        hidden=cfg['hidden'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
    )
    print(f"      模型参数量: {model.count_params():,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={'power_output': 'mse', 'status_output': 'binary_crossentropy'},
        loss_weights={'power_output': 1.0, 'status_output': conf['status_weight']},
        metrics={'power_output': 'mae'}
    )

    model_dir = Path(OUTPUT_BASE) / appliance / cfg_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "best_model.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(model_path), save_best_only=True,
                                           monitor='val_power_output_mae', mode='min',
                                           overwrite=True)
    ]

    # ★ verbose=1，能看到每轮 loss
    history = model.fit(
        X_train_bal,
        {'power_output': y_train_p, 'status_output': y_train_s},
        validation_data=(X_val, {'power_output': y_val_p, 'status_output': y_val_s}),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        sample_weight={'power_output': sample_weight},
        callbacks=callbacks
    )

    # ★ 直接加载，不先创建再覆盖
    if model_path.exists():
        model = tf.keras.models.load_model(
            str(model_path),
            custom_objects=CUSTOM_OBJECTS,
            compile=False
        )

    preds = model.predict(X_test, verbose=0)
    y_pred_power = preds[0].flatten() * max_power

    # 打印预测分布，确认模型输出正常
    print(f"      预测范围: [{y_pred_power.min():.2f}, {y_pred_power.max():.2f}]")
    print(f"      预测>阈值点数: {(y_pred_power > on_threshold).sum()}")
    print(f"      真实>阈值点数: {(y_test_true > on_threshold).sum()}")

    metrics = compute_metrics(y_test_true, y_pred_power, on_threshold)

    pd.DataFrame(history.history).to_csv(model_dir / "history.csv")

    print(f"      F1={metrics['f1']:.4f}, MAE={metrics['mae']:.2f}W, NEP={metrics['nep']:.4f}")

    return {
        'appliance': appliance,
        'config': cfg_name,
        'num_layers': cfg['num_layers'],
        'num_heads': cfg['num_heads'],
        'hidden': cfg['hidden'],
        'params': model.count_params(),
        'f1': metrics['f1'],
        'mae': metrics['mae'],
        'nep': metrics['nep'],
        'threshold': float(on_threshold),
    }


def main():
    output_root = Path(OUTPUT_BASE)
    output_root.mkdir(parents=True, exist_ok=True)
    all_results = []

    appliance = APPLIANCES[0]
    print("\n" + "#" * 70)
    print(f"# 处理电器: {appliance}")
    print("#" * 70)

    max_power, on_threshold = get_appliance_params(appliance)
    conf = APPLIANCE_PARAMS[appliance]
    print(f"  cutoff={max_power}W, threshold={on_threshold}W (固定阈值)")

    # 加载数据
    args = get_args()
    args.dataset_code = 'uk_dale'
    args = update_preprocessing_parameters(args)
    args.ukdale_location = UKDALE_DATA_ROOT
    args.appliance_names = [appliance]
    args.house_indicies = TRAIN_HOUSES
    parser_train = UK_Dale_Parser(args)
    X_train_raw, y_train_raw = generate_windows(parser_train.x, parser_train.y, WINDOW_SIZE, STEP)

    split = int(0.85 * len(X_train_raw))
    X_train, X_val = X_train_raw[:split], X_train_raw[split:]
    y_train, y_val = y_train_raw[:split], y_train_raw[split:]

    args.house_indicies = TEST_HOUSES
    parser_test = UK_Dale_Parser(args, (parser_train.x_mean, parser_train.x_std))
    X_test, y_test_true = generate_windows(parser_test.x, parser_test.y, WINDOW_SIZE, STEP)

    y_val_p = y_val / max_power
    y_val_s = (y_val > on_threshold).astype(np.float32)

    X_train_bal, y_train_bal = oversample_windows(X_train, y_train, conf['pos_ratio'], on_threshold)
    y_train_p = y_train_bal / max_power
    y_train_s = (y_train_bal > on_threshold).astype(np.float32)

    pos_count = np.sum(y_train_s == 1)
    neg_count = np.sum(y_train_s == 0)
    pos_weight = (neg_count / pos_count) ** conf['weight_power'] if pos_count > 0 else 1.0
    sample_weight = np.where(y_train_s == 1, pos_weight, 1.0).astype(np.float32)

    print(f"  训练集: {len(X_train_bal)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    print(f"  正样本: {pos_count}, 负样本: {neg_count}, 权重: {pos_weight:.2f}")
    print(f"  y_train_p 范围: [{y_train_p.min():.4f}, {y_train_p.max():.4f}]")
    print(f"  真实>阈值点数(测试集): {(y_test_true > on_threshold).sum()}")

    for cfg in HPARAM_CONFIGS:
        result = run_one_config(
            appliance, cfg,
            X_train_bal, y_train_p, y_train_s, sample_weight,
            X_val, y_val_p, y_val_s,
            X_test, y_test_true,
            max_power, on_threshold, conf
        )
        all_results.append(result)

    df = pd.DataFrame(all_results)
    df.to_csv(output_root / "hyperparam_results.csv", index=False)

    print("\n" + "=" * 90)
    print("超参数敏感性分析结果（洗衣机）")
    print("=" * 90)
    print(df.to_string(index=False))

    print("\n【层数敏感性】(H=4, D=64)")
    layer_df = df[df['config'].str.contains('H4_D64')]
    print(layer_df[['config', 'num_layers', 'f1', 'mae']].to_string(index=False))

    print("\n【头数敏感性】(L=1, D=64)")
    head_df = df[df['config'].str.match(r'L1_H\d+_D64')]
    print(head_df[['config', 'num_heads', 'f1', 'mae']].to_string(index=False))

    print("\n【隐藏维度敏感性】(L=1, H=4)")
    hidden_df = df[df['config'].str.match(r'L1_H4_D\d+')]
    print(hidden_df[['config', 'hidden', 'f1', 'mae']].to_string(index=False))


if __name__ == "__main__":
    main()