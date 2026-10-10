# model_structure_auglpn.py
# AugLPN-NILM: Augmented Lightweight Parallel Network
# 适配双输出：power_output + status_output
# 输入 shape: (batch, window_size)
# 输出: [power_output (batch, 1), status_output (batch, 1)]

import tensorflow as tf
from tensorflow.keras import layers, Model


# ============ 自定义辅助层 ============

def C21(x, filters, stride):
    """Conv2D + BN + ReLU，kernel_size=(1,3)，stride 可变"""
    x = layers.Conv2D(filters=filters, kernel_size=(1, 3),
                      strides=(1, stride), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def C_lateral(x, filters):
    """1×1 卷积降维，BN+ReLU"""
    x = layers.Conv2D(filters=filters, kernel_size=(1, 1),
                      strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def C_up(x, filters):
    """上采样 (高度不变, 宽度×2) + 卷积 + BN + ReLU"""
    x = layers.UpSampling2D(size=(1, 2))(x)
    x = layers.Conv2D(filters=filters, kernel_size=(1, 3),
                      strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


# ============ 主模型构建函数 ============

def create_auglpn_model(window_size=480):
    input_tensor = layers.Input(shape=(window_size,), name='input')
    x = layers.Reshape((1, window_size, 1))(input_tensor)

    channel = 32

    # ----- 编码器 -----
    layer1 = layers.Conv2D(filters=channel, kernel_size=(1, 3),
                           strides=(1, 2), padding='same', activation='relu')(x)

    layer2 = C21(layer1, channel, 2)
    layer3 = C21(layer2, channel * 2, 2)
    layer4 = C21(layer3, channel * 4, 2)

    # ----- 解码器（横向连接） -----
    layer4_lateral = C_lateral(layer4, channel)
    UP1 = C_up(layer4_lateral, channel)
    layer3_lateral = C_lateral(layer3, channel)
    layer3_ = layers.Add()([UP1, layer3_lateral])

    layer2_lateral = C_lateral(layer2, channel)
    UP2 = layers.UpSampling2D((1, 2))(layer3_)
    layer2_ = layers.Add()([UP2, layer2_lateral])

    layer1_lateral = C_lateral(layer1, channel)
    UP3 = layers.UpSampling2D((1, 2))(layer2_)
    layer1_ = layers.Add()([UP3, layer1_lateral])

    # ----- 注意力路径 -----
    layer1_temp_condv = layers.Conv2D(filters=channel, kernel_size=(1, 1),
                                      strides=(1, 1))(layer1_)
    layer1_temp_condv = tf.nn.l2_normalize(layer1_temp_condv, axis=2)

    max_pool = layers.GlobalMaxPooling2D()(layer1_)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = layers.Conv2D(filters=channel, kernel_size=(1, 1),
                             strides=(1, 1), activation='relu')(max_pool)
    max_pool = layers.Activation('hard_sigmoid')(max_pool)
    layer1_mul = layers.multiply([max_pool, layer1_temp_condv])

    layer1_b3 = layers.Conv2D(filters=channel, kernel_size=(1, 3),
                              strides=(1, 1), padding='same')(layer1_)
    result_1 = tf.nn.l2_normalize(layer1_b3, axis=2)
    layer1_mul = layers.multiply([layer1_mul, result_1])

    # ----- PAN 路径 -----
    layer2_up = layers.Conv2D(filters=channel * 2, kernel_size=(1, 2),
                              strides=(1, 2))(layer1_mul)
    layer2_pan_ = layers.Conv2D(filters=channel * 2, kernel_size=(1, 3),
                                strides=(1, 1), padding='same', activation='relu')(layer2_)
    layer3_pan = layers.Add()([layer2_up, layer2_pan_])

    layer3_pan_ = layers.Conv2D(filters=channel, kernel_size=(1, 3),
                                strides=(1, 2), padding='same', activation='relu')(layer3_pan)
    pan_Preoutput = layers.Add()([layer3_pan_, layer3_])

    # 扩张卷积 + 池化
    layer3_pan = layers.Conv2D(filters=96, kernel_size=(1, 3), strides=(1, 1),
                               padding='same', activation='relu', dilation_rate=2)(layer3_pan)
    layer3_pan = layers.Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1),
                               padding='same', activation='relu', dilation_rate=4)(layer3_pan)
    layer3_pan = layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(layer3_pan)
    layer3_pan = layers.Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1),
                               padding='same', activation='relu')(layer3_pan)
    pan_Preoutput = layers.Add()([layer3_pan, pan_Preoutput])

    # ----- GRU 分支 -----
    seq_len = window_size // 8  # 经过3次 stride=2 下采样

    # 分支1: 从 layer3
    sep1 = layers.SeparableConv2D(filters=128, kernel_size=(1, 2),
                                  strides=(1, 1), padding='same', activation='relu')(layer3)
    sep2 = layers.SeparableConv2D(filters=128, kernel_size=(1, 3),
                                  strides=(1, 1), padding='same', activation='relu')(layer3)
    sep_add = layers.Add()([sep1, sep2])
    gru_input = layers.Reshape((seq_len, 128))(sep_add)
    gru_out = layers.Bidirectional(layers.GRU(64, return_sequences=True))(gru_input)
    gru_out = layers.Reshape((-1, seq_len, 128))(gru_out)
    # ★★★ 修正：将通道降为 64 ★★★
    gru_out = layers.Conv2D(filters=64, kernel_size=(1, 2),
                            strides=(1, 1), padding='same', activation='relu')(gru_out)

    # 分支2: 从 layer2
    sep1_2 = layers.SeparableConv2D(filters=64, kernel_size=(1, 2),
                                    strides=(1, 1), padding='same', activation='relu')(layer2)
    sep2_2 = layers.SeparableConv2D(filters=64, kernel_size=(1, 3),
                                    strides=(1, 1), padding='same', activation='relu')(layer2)
    sep_add_2 = layers.Add()([sep1_2, sep2_2])
    sep_add_2 = layers.Conv2D(filters=64, kernel_size=(1, 2),
                              strides=(1, 2), padding='same', activation='relu')(sep_add_2)
    gru_input_2 = layers.Reshape((seq_len, 64))(sep_add_2)
    gru_out_2 = layers.Bidirectional(layers.GRU(32, return_sequences=True))(gru_input_2)
    gru_out_2 = layers.Reshape((-1, seq_len, 64))(gru_out_2)

    # ★★★ 现在两个分支通道数都是 64，可以 Add ★★★
    right_output = layers.Add()([gru_out, gru_out_2])

    # ----- 融合并输出 -----
    out1 = layers.Concatenate()([right_output, pan_Preoutput])
    flat = layers.Flatten(name='flatten')(out1)
    dense1 = layers.Dense(56, activation='relu', name='dense1')(flat)
    dense1 = layers.Dense(10, activation='relu', name='dense2')(dense1)

    # 双输出
    power_out = layers.Dense(1, activation='linear', name='power_output')(dense1)
    status_out = layers.Dense(1, activation='sigmoid', name='status_output')(dense1)

    model = Model(inputs=input_tensor, outputs=[power_out, status_out], name='AugLPN_NILM')
    return model