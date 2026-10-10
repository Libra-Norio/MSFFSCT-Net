# model_structure_s2p_cbam_no_trans.py
# 消融实验专用模型：S2P + CBAM（无Transformer）
# 保留完整模型的一切组件（下采样、上采样、位置编码、双输出），仅移除Transformer块

import tensorflow as tf
from tensorflow.keras import layers, Model


# ==========================================
# 1. 1D-CBAM 模块（与完整模型完全一致）
# ==========================================
class CBAM1D(layers.Layer):
    def __init__(self, channels, ratio=8, **kwargs):
        super(CBAM1D, self).__init__(**kwargs)
        self.channels = channels
        self.ratio = ratio
        self.mlp = tf.keras.Sequential([
            layers.Dense(channels // ratio, activation='relu', use_bias=False),
            layers.Dense(channels, use_bias=False)
        ])
        self.spatial_conv = layers.Conv1D(filters=1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, inputs):
        # inputs shape: (Batch, Seq, Channels)
        # --- Channel Attention ---
        avg_p = layers.GlobalAveragePooling1D()(inputs)
        max_p = layers.GlobalMaxPooling1D()(inputs)
        ca_w = tf.nn.sigmoid(self.mlp(avg_p) + self.mlp(max_p))
        ca_w = tf.reshape(ca_w, [-1, 1, self.channels])
        x = inputs * ca_w

        # --- Spatial Attention ---
        sa_avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        sa_max = tf.reduce_max(x, axis=-1, keepdims=True)
        sa_w = self.spatial_conv(tf.concat([sa_avg, sa_max], axis=-1))
        return x * sa_w

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels, "ratio": self.ratio})
        return config


# ==========================================
# 2. 消融模型：S2P + CBAM（无Transformer）
# ==========================================
def create_no_trans_model(window_size=480, hidden=64):
    """
    输入形状: (Batch, Window_Size) -> 例如 (Batch, 480)
    输出: [power_output (回归), status_output (分类)]

    相较于完整模型 (S2P+CBAM+Transformer)：
    - 保留：卷积主干、CBAM、下采样(AvgPooling)、位置编码、上采样(Conv1DTranspose)、双输出头
    - 移除：TransformerBlock 的两层自注意力模块（仅此一处改动）
    """
    # [A] 输入阶段
    inputs = layers.Input(shape=(window_size,), name="mains_input")
    x = layers.Reshape((window_size, 1))(inputs)

    # [B] S2P Backbone: 5层卷积（与完整模型完全一致）
    x = layers.Conv1D(30, 10, padding="same", activation="relu")(x)  # (Batch, 480, 30)
    x = layers.Conv1D(30, 8, padding="same", activation="relu")(x)  # (Batch, 480, 30)
    x = layers.Conv1D(40, 6, padding="same", activation="relu")(x)  # (Batch, 480, 40)
    x = layers.Conv1D(50, 5, padding="same", activation="relu")(x)  # (Batch, 480, 50)
    x = layers.Conv1D(hidden, 5, padding="same", activation="relu")(x)  # (Batch, 480, 64)

    # [C] CBAM: 局部特征精炼（与完整模型完全一致）
    x = CBAM1D(channels=hidden, ratio=8, name="cbam_refiner")(x)  # (Batch, 480, 64)

    # [D] 下采样：缩放至潜空间，降低计算量（与完整模型完全一致）
    x_pooled = layers.AveragePooling1D(pool_size=2, strides=2, padding="same")(x)  # (Batch, 240, 64)

    # [E] 位置编码（与完整模型完全一致）
    pos_indices = tf.range(start=0, limit=tf.shape(x_pooled)[1], delta=1)
    pos_emb = layers.Embedding(input_dim=window_size // 2 + 1, output_dim=hidden)(pos_indices)
    x_trans = x_pooled + pos_emb  # (Batch, 240, 64)

    # ================================================================
    # 【唯一的消融改动】这里原本有 2 层 TransformerBlock，现已移除
    # 完整模型中此处为：
    # x_trans = TransformerBlock(hidden, num_heads=4, ff_dim=hidden*4)(x_trans)
    # x_trans = TransformerBlock(hidden, num_heads=4, ff_dim=hidden*4)(x_trans)
    # 现在直接跳过，x_trans 保持下采样后的特征
    # ================================================================

    # [F] 上采样：恢复至原始时间分辨率（与完整模型完全一致）
    x_up = layers.Conv1DTranspose(filters=hidden, kernel_size=4, strides=2, padding="same")(x_trans)  # (Batch, 480, 64)

    # [G] Seq2Point 输出头：取窗口中心点特征（与完整模型完全一致）
    mid_idx = window_size // 2
    center_feat = layers.Lambda(lambda t: t[:, mid_idx, :], name="midpoint_feature")(x_up)  # (Batch, 64)

    # [H] 双输出头：回归 + 分类（与完整模型完全一致）
    # 回归分支：预测功率值
    power_out = layers.Dense(128, activation="tanh")(center_feat)
    power_out = layers.Dense(1, activation="linear", name="power_output")(power_out)

    # 分类分支：预测开关状态（辅助提升 F1）
    status_out = layers.Dense(128, activation="relu")(center_feat)
    status_out = layers.Dense(1, activation="sigmoid", name="status_output")(status_out)

    # 构建模型
    model = Model(inputs=inputs, outputs=[power_out, status_out], name="S2P_CBAM_NO_TRANS")
    return model


# ==========================================
# 3. 测试模型结构（验证形状）
# ==========================================
if __name__ == "__main__":
    # 使用与训练脚本相同的窗口大小 WINDOW_SIZE=480
    model = create_no_trans_model(window_size=480, hidden=64)
    model.summary()

    # 打印输出形状验证
    print("\n输入形状: (None, 480)")
    print("输出形状:")
    for out in model.outputs:
        print(f"  - {out.name}: {out.shape}")