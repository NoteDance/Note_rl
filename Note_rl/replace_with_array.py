import tensorflow as tf


def replace_with_array_k(model: tf.keras.Model, shared_params: list):
    shared_iter = iter(shared_params)
    for layer in model.layers:
        var_to_attr = {
            id(v): k
            for k, v in layer.__dict__.items()
            if isinstance(v, tf.Variable)
        }
        for var in layer.trainable_weights:
            arr = next(shared_iter)
            attr_name = var_to_attr.get(id(var))
            if attr_name is not None:
                object.__setattr__(layer, attr_name, arr)