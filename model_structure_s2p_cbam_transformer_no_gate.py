# model_structure_s2p_cbam_transformer_no_gate.py
# -*- coding: utf-8 -*-
"""
SCT-Net 消融版本：去掉门控自适应融合模块
- 保持 CBAM1D、StatFeatureExtractor、LightTransformer 与完整版一致
- 仅将门控融合替换为直接拼接（三路特征 concat）
- 用于证明门控机制的贡献
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

# ★ 复用完整版中已定义好的三个自定义层
from model_structure_s2p_cbam_transformer import (
    CBAM1D,
    StatFeatureExtractor,
    LightTransformer,
)


def create_light_transformer_model_no_gate(window_size=480, hidden=64):
    """
    无门控融合版本：
      完整版： gate * center_trans + (1 - gate) * center_cbam   →  再与 stat 拼接
      消融版： concat[center_cbam, center_trans]                →  再与 stat 拼接
    """
    inputs = layers.Input(shape=(window_size,), name="mains_input")

    # ---- 原始序列（用于统计特征）----
    raw_seq = layers.Reshape((window_size, 1))(inputs)

    # ---- S2P 卷积主干 ----
    x = layers.Reshape((window_size, 1))(inputs)
    x = layers.Conv1D(30, 10, padding="same", activation="relu")(x)
    x = layers.Conv1D(30, 8,  padding="same", activation="relu")(x)
    x = layers.Conv1D(40, 6,  padding="same", activation="relu")(x)
    x = layers.Conv1D(50, 5,  padding="same", activation="relu")(x)
    x = layers.Conv1D(hidden, 5, padding="same", activation="relu")(x)

    # ---- CBAM ----
    cbam_out = CBAM1D(channels=hidden)(x)            # (B, W, hidden)

    # ---- 统计特征 ----
    stat_feat = StatFeatureExtractor(hidden_dim=16)(raw_seq)   # (B, 16)

    # ---- 位置编码 + 轻量 Transformer ----
    T = window_size
    pos_emb = layers.Embedding(input_dim=T, output_dim=hidden)(tf.range(T))
    pos_emb = tf.expand_dims(pos_emb, axis=0)        # (1, T, hidden)
    x_trans = cbam_out + pos_emb
    x_trans = LightTransformer(embed_dim=hidden, num_heads=4,
                                ff_dim=hidden * 2, dropout=0.2)(x_trans)

    # ---- 取中心点 ----
    mid_idx = window_size // 2
    center_cbam  = layers.Lambda(lambda t: t[:, mid_idx, :],
                                 name="center_cbam")(cbam_out)     # (B, hidden)
    center_trans = layers.Lambda(lambda t: t[:, mid_idx, :],
                                 name="center_trans")(x_trans)     # (B, hidden)

    # ============================================================
    # ★ 消融点：去掉门控，改为直接拼接三路特征
    # ============================================================
    combined = tf.concat([center_cbam, center_trans, stat_feat], axis=-1)

    enhanced = layers.Dense(hidden, activation='relu')(combined)
    enhanced = layers.Dropout(0.2)(enhanced)

    # ---- 双输出头（与完整版一致）----
    power_out = layers.Dense(128, activation="tanh")(enhanced)
    power_out = layers.Dense(1, activation="linear",
                             name="power_output")(power_out)

    status_out = layers.Dense(128, activation="relu")(enhanced)
    status_out = layers.Dense(1, activation="sigmoid",
                              name="status_output")(status_out)

    model = Model(inputs=inputs,
                  outputs=[power_out, status_out],
                  name="S2P_CBAM_LightTransformer_NoGate")
    return model