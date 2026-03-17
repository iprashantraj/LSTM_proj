import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer

original_dense_from_config = Dense.from_config
def patched_dense_from_config(cls, config):
    config.pop("quantization_config", None)
    return original_dense_from_config(config)
Dense.from_config = classmethod(patched_dense_from_config)

original_input_from_config = InputLayer.from_config
def patched_input_from_config(cls, config):
    config.pop("optional", None)
    if "batch_shape" in config:
        config["batch_input_shape"] = config.pop("batch_shape")
    return original_input_from_config(config)
InputLayer.from_config = classmethod(patched_input_from_config)

input_layer = InputLayer.from_config({"batch_shape": [None, 100, 1], "optional": False})
print("InputLayer:", input_layer)
