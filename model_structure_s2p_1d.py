# model_structure_s2p_1d.py
# 正确的1D S2P模型，与2D版滤波器数量完全一致

import tensorflow as tf
from tensorflow.keras import layers, Model


def create_s2p_1d_model(window_size=480, hidden=64):
    """
    1D S2P模型（正确版本）
    - 5层1D卷积，与2D版滤波器数量一致：30,30,40,50,64
    - 取中心点（S2P核心）
    - 双输出：功率回归 + 状态分类
    - 无Flatten，无大Dense层
    """
    inputs = layers.Input(shape=(window_size,), name="mains_input")

    # Reshape: (B, 480) -> (B, 480, 1)
    x = layers.Reshape((window_size, 1))(inputs)

    # 5层1D卷积（与2D版滤波器数量完全一致）
    x = layers.Conv1D(30, 10, padding='same', activation='relu')(x)
    x = layers.Conv1D(30, 8, padding='same', activation='relu')(x)
    x = layers.Conv1D(40, 6, padding='same', activation='relu')(x)
    x = layers.Conv1D(50, 5, padding='same', activation='relu')(x)
    x = layers.Conv1D(hidden, 5, padding='same', activation='relu')(x)  # (B, 480, 64)

    # S2P核心：取中心点（第240个采样点）
    mid_idx = window_size // 2
    x = layers.Lambda(lambda t: t[:, mid_idx, :])(x)  # (B, 64)

    # 双输出
    power_out = layers.Dense(1, activation='linear', name='power_output')(x)
    status_out = layers.Dense(1, activation='sigmoid', name='status_output')(x)

    model = Model(inputs=inputs, outputs=[power_out, status_out], name="S2P_1D")
    return model


# 验证参数量
if __name__ == "__main__":
    model = create_s2p_1d_model(window_size=480, hidden=64)
    model.summary()

    params = model.count_params()
    print(f"\n参数量: {params:,} = {params / 1_000_000:.2f} M")