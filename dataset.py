import numpy as np
import tensorflow as tf


def get_response_fn(gradient=1., bias=0., noise_sd=0.1):
    def response_fn(x):
        return x * gradient + bias + np.random.normal(scale=noise_sd)

    return response_fn


def get_generator(gradient=1., bias=0., noise_sd=0.1):
    response_fn = get_response_fn(
        gradient=gradient, bias=bias, noise_sd=noise_sd)

    def generator():
        while True:
            x = np.random.uniform(low=-1., high=+1.)
            yield {'x': x}, response_fn(x)

    return generator


def get_dataset(generator, batch_size=128, name="dataset"):
    with tf.name_scope(name):
        dataset = tf.data.Dataset.from_generator(
            generator=generator,
            output_types=({'x': tf.float32}, tf.float32),
            output_shapes=({'x': tf.TensorShape([])},
                           tf.TensorShape([])))
        dataset = dataset.batch(batch_size=batch_size)
    return dataset
