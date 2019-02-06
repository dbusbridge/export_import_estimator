import tensorflow as tf

from tensorflow.python.estimator.model_fn import ModeKeys


def get_eval_metric_ops(labels, predictions):
    metrics = dict()
    with tf.name_scope("eval_metric_ops"):
        metrics.update(
            {"mse": tf.metrics.mean_squared_error(
                labels=labels, predictions=predictions, name='mse')})

    return metrics


def get_model_fn(architecture_fn):
    def model_fn(features, labels, mode, params):
        features = tf.reshape(features['x'], [-1, 1], name="features_reshape")

        training = mode == ModeKeys.TRAIN

        predicted_response = architecture_fn(features, training, params)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predicted_response = tf.reshape(predicted_response, [-1])

            return tf.estimator.EstimatorSpec(
                mode=mode, predictions={'response': predicted_response})

        labels = tf.reshape(labels, [-1, 1], name="labels")

        loss = tf.losses.mean_squared_error(
            labels=labels, predictions=predicted_response)

        eval_metric_ops = get_eval_metric_ops(
            labels=labels, predictions=predicted_response)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        assert mode == tf.estimator.ModeKeys.TRAIN

        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        global_step = tf.train.get_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return model_fn
