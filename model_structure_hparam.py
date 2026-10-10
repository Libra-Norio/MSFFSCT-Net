# model_structure_hparam.py
# 支持可变Transformer超参数的SCT-Net模型
# 用于超参数敏感性分析：层数、注意力头数、隐藏维度

import tensorflow as tf
from tensorflow.keras import layers, Model


# ============================================================
# 模块1：CBAM1D
# ============================================================
@tf.keras.utils.register_keras_serializable()
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
        avg_p = tf.reduce_mean(inputs, axis=1)
        max_p = tf.reduce_max(inputs, axis=1)
        ca_w = tf.nn.sigmoid(self.mlp(avg_p) + self.mlp(max_p))
        ca_w = tf.reshape(ca_w, [-1, 1, self.channels])
        x = inputs * ca_w
        sa_avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        sa_max = tf.reduce_max(x, axis=-1, keepdims=True)
        sa_w = self.spatial_conv(tf.concat([sa_avg, sa_max], axis=-1))
        return x * sa_w

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels, "ratio": self.ratio})
        return config


# ============================================================
# 模块2：统计特征提取器
# ============================================================
@tf.keras.utils.register_keras_serializable()
class StatFeatureExtractor(layers.Layer):
    def __init__(self, hidden_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dense = layers.Dense(hidden_dim, activation='relu')

    def call(self, raw_input):
        mean = tf.reduce_mean(raw_input, axis=1)
        std = tf.math.reduce_std(raw_input, axis=1)
        maxv = tf.reduce_max(raw_input, axis=1)
        minv = tf.reduce_min(raw_input, axis=1)
        sign_changes = tf.abs(tf.sign(raw_input[:, 1:, :]) - tf.sign(raw_input[:, :-1, :]))
        zcr = tf.reduce_sum(sign_changes, axis=1) / tf.cast(tf.shape(raw_input)[1], tf.float32)
        stats = tf.concat([mean, std, maxv, minv, zcr], axis=-1)
        return self.dense(stats)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


# ============================================================
# 模块3：轻量Transformer（单层，可堆叠）
# ============================================================
@tf.keras.utils.register_keras_serializable()
class LightTransformer(layers.Layer):
    def __init__(self, embed_dim, num_heads=4, ff_dim=128, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim)
        ])
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=True):
        attn_out = self.att(x, x, training=training)
        x = self.ln1(x + self.dropout1(attn_out, training=training))
        ffn_out = self.ffn(x, training=training)
        x = self.ln2(x + self.dropout2(ffn_out, training=training))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate
        })
        return config


# ============================================================
# 完整模型（支持可变超参数）
# ============================================================
def create_hparam_model(window_size=480, hidden=64, num_layers=1, num_heads=4):
    """
    支持可变Transformer超参数的SCT-Net

    参数：
    - window_size: 输入序列长度（固定480）
    - hidden: 隐藏维度（可变：32/64/128）
    - num_layers: Transformer层数（可变：1/2/3）
    - num_heads: 注意力头数（可变：2/4/8）
    """
    inputs = layers.Input(shape=(window_size,), name="mains_input")
    raw_seq = layers.Reshape((window_size, 1))(inputs)

    # ---- S2P 卷积 ----
    x = layers.Reshape((window_size, 1))(inputs)
    x = layers.Conv1D(30, 10, padding="same", activation="relu")(x)
    x = layers.Conv1D(30, 8, padding="same", activation="relu")(x)
    x = layers.Conv1D(40, 6, padding="same", activation="relu")(x)
    x = layers.Conv1D(50, 5, padding="same", activation="relu")(x)
    x = layers.Conv1D(hidden, 5, padding="same", activation="relu")(x)

    # ---- CBAM ----
    cbam_out = CBAM1D(channels=hidden)(x)

    # ---- 统计特征 ----
    stat_feat = StatFeatureExtractor(hidden_dim=16)(raw_seq)

    # ---- 位置编码 ----
    T = window_size
    pos_emb = layers.Embedding(input_dim=T, output_dim=hidden)(tf.range(T))
    pos_emb = tf.expand_dims(pos_emb, axis=0)
    x_trans = cbam_out + pos_emb

    # ---- ★ 可变层数的Transformer ----
    for layer_idx in range(num_layers):
        x_trans = LightTransformer(
            embed_dim=hidden,
            num_heads=num_heads,
            ff_dim=hidden * 2,
            dropout=0.2,
            name=f"light_transformer_{layer_idx}"
        )(x_trans)

    # ---- 取中心点 ----
    mid_idx = window_size // 2
    center_cbam = layers.Lambda(lambda t: t[:, mid_idx, :])(cbam_out)
    center_trans = layers.Lambda(lambda t: t[:, mid_idx, :])(x_trans)

    # ---- 自适应融合 ----
    gate_concat = tf.concat([center_cbam, center_trans, stat_feat], axis=-1)
    gate = layers.Dense(1, activation='sigmoid')(gate_concat)
    fused = gate * center_trans + (1 - gate) * center_cbam

    # ---- 融合统计特征 ----
    combined = tf.concat([fused, stat_feat], axis=-1)
    enhanced = layers.Dense(hidden, activation='relu')(combined)
    enhanced = layers.Dropout(0.2)(enhanced)

    # ---- 双输出头 ----
    power_out = layers.Dense(128, activation="tanh")(enhanced)
    power_out = layers.Dense(1, activation="linear", name="power_output")(power_out)

    status_out = layers.Dense(128, activation="relu")(enhanced)
    status_out = layers.Dense(1, activation="sigmoid", name="status_output")(status_out)

    model = Model(inputs=inputs, outputs=[power_out, status_out],
                  name=f"SCT_Net_L{num_layers}_H{num_heads}_D{hidden}")
    return model