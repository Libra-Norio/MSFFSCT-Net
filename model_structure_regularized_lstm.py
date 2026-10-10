# model_structure_regularized_lstm.py
"""
Regularized LSTM 模型（Rafiq et al. SEGE 2018）
适配双输出：功率回归 + 状态分类
输入：(batch, window_size, 1)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

def create_regularized_lstm_model(window_size=480, dropout_rate=0.3):
    """
    构建双输出 Regularized LSTM 模型
    - Conv1D: 16 filters, kernel=4, activation='linear'
    - Dropout 0.3
    - BiLSTM 128 (return_sequences=True)
    - Dropout 0.3
    - BiLSTM 256 (return_sequences=False)
    - Dropout 0.3
    - Dense 128, tanh
    - 双输出：功率 (linear), 状态 (sigmoid)
    """
    inputs = layers.Input(shape=(window_size, 1), name='input')  # (B, W, 1)

    # Conv1D (activation='linear'，无 padding 则长度变化，但我们需要保持序列长度，使用 padding='same')
    x = layers.Conv1D(filters=16, kernel_size=4, strides=1, padding='same', activation='linear')(inputs)
    x = layers.Dropout(dropout_rate)(x)

    # BiLSTM 128, return sequences
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True), merge_mode='concat')(x)
    x = layers.Dropout(dropout_rate)(x)

    # BiLSTM 256, return sequences=False
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=False), merge_mode='concat')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Dense 128, tanh
    x = layers.Dense(128, activation='tanh')(x)

    # 共享特征后接双输出
    # 功率回归 (linear)
    power_out = layers.Dense(1, activation='linear', name='power_output')(x)
    # 状态分类 (sigmoid)
    status_out = layers.Dense(1, activation='sigmoid', name='status_output')(x)

    model = Model(inputs=inputs, outputs=[power_out, status_out], name='RegularizedLSTM')
    return model