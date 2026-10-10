# model_structure_seq2point_balanced.py
# TensorFlow/Keras 实现 Seq2Point Balanced (论文: Towards Robust NILM, 2025)
# 适配双输出：power_output + status_output

import tensorflow as tf
from tensorflow.keras import layers, Model


def create_seq2point_balanced_model(input_window_length=480):
    """
    构建 Seq2Point Balanced 模型（TF 版本）
    输入: (batch, input_window_length)
    输出: [power_output (batch, 1), status_output (batch, 1)]
    """
    inputs = layers.Input(shape=(input_window_length,), name='input')  # (B, L)

    # ---- 增加通道维度: (B, L, 1) ----
    x = layers.Reshape((input_window_length, 1))(inputs)

    # ---- Block 1: Conv + BN + ReLU + MaxPool ----
    x = layers.Conv1D(filters=16, kernel_size=8, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)  # L -> L/2

    # ---- Block 2 ----
    x = layers.Conv1D(filters=24, kernel_size=6, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)  # L/2 -> L/4

    # ---- Block 3 ----
    x = layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)  # L/4 -> L/8
    x = layers.Dropout(0.3)(x)

    # ---- Flatten + Dense ----
    x = layers.Flatten()(x)  # (B, 32 * (L/8))
    x = layers.Dense(768, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # ---- 共享特征 ----
    shared = x  # (B, 768)

    # ---- 回归分支 (功率) ----
    power_out = layers.Dense(1, activation='linear', name='power_output')(shared)

    # ---- 分类分支 (状态) ----
    status_out = layers.Dense(1, activation='sigmoid', name='status_output')(shared)

    model = Model(inputs=inputs, outputs=[power_out, status_out], name='Seq2Point_Balanced')
    return model