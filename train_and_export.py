"""
Train a simple linear regression model
"""
import tensorflow as tf

from dataset import get_dataset
from dataset import get_generator
from model import get_model_fn

tf.logging.set_verbosity(tf.logging.DEBUG)

flags = tf.flags

# Run config
flags.DEFINE_string(
    name="model_dir",
    default='/tmp/my_model/trained',
    help="Directory to put the trained model.")
flags.DEFINE_string(
    name="export_dir",
    default='/tmp/my_model/exported',
    help="Directory to put the exported model.")

flags.DEFINE_integer(
    name="random_seed",
    default=42,
    help="The random seed.")

# Hyperparameters
flags.DEFINE_float(name="learning_rate",
                   default=0.01,
                   help="The learning rate.")

flags.DEFINE_integer(name="batch_size",
                     default=32,
                     help="The batch size.")
flags.DEFINE_integer(name="max_steps",
                     default=100,
                     help="The maximum number of steps to train for.")


FLAGS = flags.FLAGS
FLAGS.mark_as_parsed()


def architecture_fn(features, training, params):
    predicted_response = tf.keras.layers.Dense(units=1)(features)
    return predicted_response


def train():
    params = tf.contrib.training.HParams(
        learning_rate=FLAGS.learning_rate,
        batch_size=FLAGS.batch_size,
        max_steps=FLAGS.max_steps)

    generator = get_generator()
    dataset = get_dataset(generator, batch_size=params.batch_size)

    def input_fn():
        with tf.name_scope("input"):
            iterator = dataset.make_one_shot_iterator()
            next_items = iterator.get_next()

        return next_items

    config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        tf_random_seed=FLAGS.random_seed)

    model_fn = get_model_fn(architecture_fn=architecture_fn)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, config=config, params=params)

    return estimator.train(input_fn=input_fn, max_steps=FLAGS.max_steps)


def serving_input_receiver_fn():
    receiver_tensors = {'x': tf.placeholder(tf.float32, [None, 1])}
    features = receiver_tensors.copy()

    return tf.estimator.export.ServingInputReceiver(
        receiver_tensors=receiver_tensors, features=features)


def train_and_export(unused_argv):
    estimator = train()
    estimator.export_saved_model(
        export_dir_base=FLAGS.export_dir,
        serving_input_receiver_fn=serving_input_receiver_fn)


if __name__ == '__main__':
    tf.app.run(main=train_and_export)
