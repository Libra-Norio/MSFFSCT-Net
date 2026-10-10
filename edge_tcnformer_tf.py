# edge_tcnformer_tf.py
# TensorFlow/Keras 实现 EdgeTCNformer
# 输入: (B, T, 1)  输出: power (B, n_appliances, T), state (B, n_appliances, T)

import tensorflow as tf
from tensorflow.keras import layers, Model


class DepthwiseSeparableConv1D(layers.Layer):
    """深度可分离一维卷积（Depthwise + Pointwise）"""
    def __init__(self, out_channels, kernel_size, dilation, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.depthwise = layers.DepthwiseConv1D(
            kernel_size=kernel_size,
            padding='same',
            dilation_rate=dilation,
            depth_multiplier=1,
            activation=None
        )
        self.pointwise = layers.Conv1D(
            filters=out_channels,
            kernel_size=1,
            padding='same',
            activation=None
        )

    def call(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        return x


class AttentionNorm(layers.Layer):
    """注意力归一化：全局平均池化 → Dense → Sigmoid → 逐通道相乘"""
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.gap = layers.GlobalAveragePooling1D()
        self.fc = layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        # inputs: (B, T, C)
        weights = self.gap(inputs)            # (B, C)
        weights = self.fc(weights)             # (B, C)
        weights = tf.expand_dims(weights, 1)   # (B, 1, C)
        return inputs * weights


class DualDilatedTCNBlock(layers.Layer):
    """多分支扩张卷积 TCN 块"""
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 dilations=(1, 2, 4), **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.branches = []
        for d in dilations:
            self.branches.append(
                DepthwiseSeparableConv1D(out_channels, kernel_size, d)
            )
        self.norm = layers.LayerNormalization(axis=-1)
        self.relu = layers.ReLU()

    def call(self, inputs):
        # inputs: (B, T, C)
        branch_outs = [branch(inputs) for branch in self.branches]
        out = tf.add_n(branch_outs) / len(branch_outs)
        out = self.norm(out)
        out = self.relu(out)
        return out


class LightweightTransformerBlock(layers.Layer):
    """轻量 Transformer 块（Post-Norm）"""
    def __init__(self, embed_dim, n_heads=2, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        # ★ 修复：MultiHeadAttention，H和A都大写
        self.attn = layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=embed_dim // n_heads
        )
        self.ff = tf.keras.Sequential([
            layers.Dense(embed_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.norm1 = layers.LayerNormalization(axis=-1)
        self.norm2 = layers.LayerNormalization(axis=-1)

    def call(self, inputs, training=False):
        # inputs: (B, T, C)
        attn_out = self.attn(inputs, inputs, training=training)
        x = self.norm1(inputs + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class MultiTaskDecoder(layers.Layer):
    """多任务解码器：每个电器一个功率头 + 一个状态头"""
    def __init__(self, in_channels, n_appliances, **kwargs):
        super().__init__(**kwargs)
        self.n_appliances = n_appliances
        self.power_heads = []
        self.state_heads = []
        for i in range(n_appliances):
            self.power_heads.append(
                layers.Dense(1, activation=None, name=f'power_head_{i}')
            )
            self.state_heads.append(
                layers.Dense(1, activation='sigmoid', name=f'state_head_{i}')
            )

    def call(self, inputs):
        # inputs: (B, T, C)
        power_out = []
        state_out = []
        for i in range(self.n_appliances):
            p = self.power_heads[i](inputs)   # (B, T, 1)
            p = tf.squeeze(p, axis=-1)        # (B, T)
            power_out.append(p)

            s = self.state_heads[i](inputs)   # (B, T, 1)
            s = tf.squeeze(s, axis=-1)        # (B, T)
            state_out.append(s)

        power_out = tf.stack(power_out, axis=1)  # (B, n_appliances, T)
        state_out = tf.stack(state_out, axis=1)  # (B, n_appliances, T)
        return power_out, state_out


class EdgeTCNformer(Model):
    """EdgeTCNformer 主体"""
    def __init__(self, input_channels=1, embed_dim=64, n_appliances=1, **kwargs):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.n_appliances = n_appliances

        self.encoder = tf.keras.Sequential([
            DualDilatedTCNBlock(input_channels, embed_dim),
            AttentionNorm(embed_dim),
            DualDilatedTCNBlock(embed_dim, embed_dim),
            AttentionNorm(embed_dim)
        ])
        self.transformer = LightweightTransformerBlock(embed_dim, n_heads=2)
        self.decoder = MultiTaskDecoder(embed_dim, n_appliances)

    def call(self, inputs, training=False):
        # inputs: (B, T, 1)
        x = self.encoder(inputs, training=training)
        x = self.transformer(x, training=training)
        power, state = self.decoder(x)
        return power, state


def create_edge_tcnformer(window_size=480, embed_dim=64, n_appliances=1):
    """创建 EdgeTCNformer 模型，并在输出端取中心点（兼容 S2P 训练）"""
    inputs = layers.Input(shape=(window_size, 1), name='input')
    model = EdgeTCNformer(input_channels=1, embed_dim=embed_dim,
                          n_appliances=n_appliances)
    power, state = model(inputs)

    mid_idx = window_size // 2
    # 取第一个电器的中心点（n_appliances=1 时就是唯一电器）
    power_center = layers.Lambda(lambda t: t[:, 0, mid_idx],
                                 name='power_output')(power)
    state_center = layers.Lambda(lambda t: t[:, 0, mid_idx],
                                 name='status_output')(state)

    final_model = Model(inputs=inputs, outputs=[power_center, state_center],
                        name='EdgeTCNformer')
    return final_model