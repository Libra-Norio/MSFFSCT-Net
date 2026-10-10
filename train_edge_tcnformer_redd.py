# train_edge_tcnformer_redd.py
# EdgeTCNformer 在 REDD 数据集上的训练脚本（固定阈值，S2P 范式）
# 同时保存 .keras（完整模型）和 .weights.keras（权重）

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

from REDD_Parser import Redd_Parser
from config import get_args, update_preprocessing_parameters
from edge_tcnformer_tf import create_edge_tcnformer

REDD_DATA_ROOT = "./data/REDD"
OUTPUT_BASE = "./edge_tcnformer_redd"

WINDOW_SIZE = 480
STEP = 240
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

APPLIANCES = ['refrigerator', 'washer_dryer', 'microwave', 'dishwaser']
TRAIN_HOUSES = [2, 3, 4, 5, 6]
TEST_HOUSES = [1]

APPLIANCE_PARAMS = {
    'refrigerator':  {'pos_ratio': 0.5,  'weight_power': 1.5, 'status_weight': 1.0},
    'washer_dryer':  {'pos_ratio': 0.55, 'weight_power': 2.0, 'status_weight': 2.0},
    'microwave':     {'pos_ratio': 0.65, 'weight_power': 2.5, 'status_weight': 3.0},
    'dishwaser':     {'pos_ratio': 0.5,  'weight_power': 1.5, 'status_weight': 2.0},
}


# ============================================================
# 自定义回调：同时保存 .keras 和 .weights.keras
# ============================================================
class SaveBestBothFormats(tf.keras.callbacks.Callback):
    """
    在 monitor 指标最优时：
      1) 先保存 .weights.keras（稳）
      2) 再尝试保存 .keras（完整模型，可能因 Lambda/自定义层失败）
    失败不中断训练。
    """
    def __init__(self, keras_path, weights_path,
                 monitor='val_power_output_mae', mode='min', verbose=1):
        super().__init__()
        self.keras_path = str(keras_path)
        self.weights_path = str(weights_path)
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best = np.inf if mode == 'min' else -np.inf
        self.keras_ok = True
        self.first_save_done = False

    def _is_improved(self, current):
        if self.mode == 'min':
            return current < self.best
        return current > self.best

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            if self.verbose and not self.first_save_done:
                print(f"  [SaveCallback] 未找到 monitor='{self.monitor}'，"
                      f"可用 keys: {list(logs.keys())}")
            return

        if not self._is_improved(current):
            return

        self.best = current
        self.first_save_done = True

        # 1) 先保存权重（最稳）
        try:
            self.model.save_weights(self.weights_path)
            msg = f"  [保存] epoch {epoch+1}: 权重 -> {Path(self.weights_path).name}"
        except Exception as e:
            msg = f"  [保存失败] epoch {epoch+1}: 权重保存失败: {e}"

        # 2) 再尝试保存完整模型
        if self.keras_ok:
            try:
                self.model.save(self.keras_path)
                msg += f"\n  [保存] epoch {epoch+1}: 完整模型 -> {Path(self.keras_path).name}"
            except Exception as e:
                self.keras_ok = False
                msg += (f"\n  [警告] epoch {epoch+1}: .keras 保存失败，"
                        f"后续只保存权重。原因：{str(e)[:300]}")

        if self.verbose:
            print(msg)


# ============================================================
# 工具函数
# ============================================================
def compute_metrics(y_true, y_pred, threshold):
    mae = np.mean(np.abs(y_true - y_pred))
    sum_true = np.sum(y_true) + 1e-8
    nep = np.sum(np.abs(y_true - y_pred)) / sum_true
    bin_true = (y_true > threshold).astype(int)
    bin_pred = (y_pred > threshold).astype(int)
    f1 = f1_score(bin_true, bin_pred, zero_division=0)
    return {'f1': float(f1), 'mae': float(mae), 'nep': float(nep)}


def get_appliance_params(appliance):
    key = 'dishwasher' if appliance == 'dishwaser' else appliance
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


# ============================================================
# 主流程
# ============================================================
def main():
    output_root = Path(OUTPUT_BASE)
    output_root.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for appliance in APPLIANCES:
        print(f"\n{'='*60}")
        print(f">>>> 处理电器: {appliance} <<<<")
        print('='*60)

        max_power, on_threshold = get_appliance_params(appliance)
        conf = APPLIANCE_PARAMS[appliance]

        # ---------- 训练数据 ----------
        args = get_args()
        args.dataset_code = 'redd_lf'
        args = update_preprocessing_parameters(args)
        args.redd_location = REDD_DATA_ROOT
        args.appliance_names = [appliance]
        args.house_indicies = TRAIN_HOUSES
        args.sampling = '6s'
        parser_train = Redd_Parser(args)
        X_train_raw, y_train_raw = generate_windows(parser_train.x, parser_train.y,
                                                    WINDOW_SIZE, STEP)
        print(f"  训练集窗口数: {len(X_train_raw)}")

        split = int(0.85 * len(X_train_raw))
        X_train, X_val = X_train_raw[:split], X_train_raw[split:]
        y_train, y_val = y_train_raw[:split], y_train_raw[split:]
        print(f"  切分后训练集: {len(X_train)}, 验证集: {len(X_val)}")

        # ---------- 测试数据 ----------
        args.house_indicies = TEST_HOUSES
        parser_test = Redd_Parser(args, stats=(parser_train.x_mean,
                                               parser_train.x_std))
        X_test, y_test_true = generate_windows(parser_test.x, parser_test.y,
                                               WINDOW_SIZE, STEP)
        print(f"  测试集窗口数: {len(X_test)}")

        # ---------- 标签 ----------
        y_val_s = (y_val > on_threshold).astype(np.float32)

        X_train_bal, y_train_bal = oversample_windows(
            X_train, y_train, conf['pos_ratio'], on_threshold)

        y_train_p = (y_train_bal / max_power).reshape(-1, 1).astype(np.float32)
        y_train_s = (y_train_bal > on_threshold).astype(np.float32).reshape(-1, 1)

        y_val_p = (y_val / max_power).reshape(-1, 1).astype(np.float32)
        y_val_s = y_val_s.reshape(-1, 1)

        pos_count = int(np.sum(y_train_s == 1))
        neg_count = int(np.sum(y_train_s == 0))
        pos_weight = ((neg_count / pos_count) ** conf['weight_power']
                      if pos_count > 0 else 1.0)
        sample_weight = np.where(y_train_s == 1, pos_weight, 1.0).astype(np.float32)
        sample_weight = sample_weight.reshape(-1, 1)

        print(f"  正样本数: {pos_count}, 负样本数: {neg_count}, "
              f"权重: {pos_weight:.2f}")

        # ---------- 构建模型 ----------
        model = create_edge_tcnformer(window_size=WINDOW_SIZE,
                                      embed_dim=64, n_appliances=1)
        print(f"  模型参数量: {model.count_params():,}")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss={'power_output': 'mse', 'status_output': 'binary_crossentropy'},
            loss_weights={'power_output': 1.0,
                          'status_output': conf['status_weight']},
            metrics={'power_output': 'mae'}
        )

        model_dir = output_root / appliance
        model_dir.mkdir(parents=True, exist_ok=True)

        # ★ 两种文件的路径
        keras_path = model_dir / "best_model.keras"
        weights_path = model_dir / "best_weights.weights.keras"

        # ---------- 回调 ----------
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_power_output_mae',
                patience=15,
                mode='min',
                restore_best_weights=True
            ),
            SaveBestBothFormats(
                keras_path=keras_path,
                weights_path=weights_path,
                monitor='val_power_output_mae',
                mode='min',
                verbose=1
            ),
        ]

        # ---------- 训练 ----------
        print("  开始训练 EdgeTCNformer ...")
        history = model.fit(
            X_train_bal[..., np.newaxis],
            {'power_output': y_train_p, 'status_output': y_train_s},
            validation_data=(X_val[..., np.newaxis],
                             {'power_output': y_val_p, 'status_output': y_val_s}),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            sample_weight={'power_output': sample_weight},
            callbacks=callbacks
        )

        # ---------- 最终兜底保存 ----------
        try:
            model.save_weights(str(weights_path))
            print(f"  [最终保存] 权重 -> {weights_path.name}")
        except Exception as e:
            print(f"  [最终保存] 权重失败: {e}")

        if not keras_path.exists():
            try:
                model.save(str(keras_path))
                print(f"  [最终保存] 完整模型 -> {keras_path.name}")
            except Exception as e:
                print(f"  [最终保存] .keras 失败: {str(e)[:200]}")

        # ---------- 测试 ----------
        print("\n  >>> 加载最优模型进行测试 <<<")
        eval_model = None
        source = None

        if keras_path.exists():
            try:
                eval_model = tf.keras.models.load_model(
                    str(keras_path), compile=False)
                source = 'keras'
                print(f"  ✅ 加载完整模型: {keras_path.name}")
            except Exception as e:
                print(f"  ⚠️ 加载 .keras 失败: {str(e)[:200]}")
                eval_model = None

        if eval_model is None and weights_path.exists():
            eval_model = create_edge_tcnformer(
                window_size=WINDOW_SIZE, embed_dim=64, n_appliances=1)
            try:
                eval_model.load_weights(str(weights_path))
                source = 'weights'
                print(f"  ✅ 加载权重: {weights_path.name}")
            except Exception as e:
                print(f"  ⚠️ 直接加载失败，尝试 by_name: {str(e)[:200]}")
                eval_model.load_weights(str(weights_path),
                                        by_name=True, skip_mismatch=True)
                source = 'weights(skip)'

        if eval_model is None:
            print("  ❌ 无法加载模型，跳过测试")
            continue

        preds = eval_model.predict(X_test[..., np.newaxis], verbose=0)
        if isinstance(preds, (list, tuple)):
            y_pred_power = preds[0].flatten() * max_power
        elif isinstance(preds, dict):
            y_pred_power = np.asarray(preds['power_output']).flatten() * max_power
        else:
            y_pred_power = np.asarray(preds).flatten() * max_power
        y_pred_power = np.maximum(y_pred_power, 0)

        best_t = on_threshold
        print(f"  使用固定阈值: {best_t:.2f}W")

        metrics = compute_metrics(y_test_true, y_pred_power, best_t)
        metrics['threshold'] = float(best_t)
        metrics['model_source'] = source
        metrics['dataset'] = 'REDD'
        all_results[appliance] = metrics

        with open(model_dir / "metrics.json", 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        pd.DataFrame(history.history).to_csv(model_dir / "history.csv")
        print(f"  {appliance}: F1={metrics['f1']:.4f}, "
              f"MAE={metrics['mae']:.2f}W, NEP={metrics['nep']:.4f} "
              f"(来源={source})")

    with open(output_root / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("全部完成！结果保存在:", OUTPUT_BASE)
    for app, met in all_results.items():
        print(f"{app}: F1={met['f1']:.4f}, MAE={met['mae']:.2f}W, "
              f"NEP={met['nep']:.4f}")


if __name__ == "__main__":
    main()