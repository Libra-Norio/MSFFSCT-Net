# model_structure_s2p_dual.py
import tensorflow as tf
from tensorflow.keras import layers, Model

def create_s2p_dual_model(input_window_length):
    input_layer = tf.keras.layers.Input(shape=(input_window_length,))
    reshape_layer = tf.keras.layers.Reshape((1, input_window_length, 1))(input_layer)

    conv_1 = layers.Conv2D(30, (10, 1), padding='same', activation='relu')(reshape_layer)
    conv_2 = layers.Conv2D(30, (8, 1), padding='same', activation='relu')(conv_1)
    conv_3 = layers.Conv2D(40, (6, 1), padding='same', activation='relu')(conv_2)
    conv_4 = layers.Conv2D(50, (5, 1), padding='same', activation='relu')(conv_3)
    conv_5 = layers.Conv2D(50, (5, 1), padding='same', activation='relu')(conv_4)

    flatten = layers.Flatten()(conv_5)
    shared = layers.Dense(1024, activation='relu')(flatten)

    # 回归输出
    power_out = layers.Dense(1, activation='linear', name='power_output')(shared)
    # 分类输出
    status_out = layers.Dense(128, activation='relu')(shared)
    status_out = layers.Dropout(0.3)(status_out)
    status_out = layers.Dense(1, activation='sigmoid', name='status_output')(status_out)

    model = Model(inputs=input_layer, outputs=[power_out, status_out])
    return model