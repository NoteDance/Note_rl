import tensorflow as tf
import torch


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


def replace_with_array_p(model: torch.nn.Module, shared_params: list):
    shared_iter = iter(shared_params)
    for module in model.modules():
        for attr_name in list(module._parameters):
            if module._parameters[attr_name] is not None:
                arr = next(shared_iter)
                object.__setattr__(module, attr_name, torch.from_numpy(arr))
