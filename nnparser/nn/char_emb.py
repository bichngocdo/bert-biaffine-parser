import tensorflow as tf

import nnparser.nn


def character_based_embeddings(inputs, input_dim, hidden_dim, output_dim, lengths=None,
                               is_training=False,
                               input_keep_prob=1.,
                               output_keep_prob=1.,
                               state_keep_prob=1.,
                               variational_recurrent=False):
    with tf.variable_scope('character_based_embeddings'):
        with tf.variable_scope('input'):
            ndim = len(inputs.shape)
            input_original_shape = tf.shape(inputs)
            if ndim > 3:
                dims = tf.unstack(tf.shape(inputs))
                batch_dim = 1
                for dim in dims[:-2]:
                    batch_dim *= dim
                time_dim = dims[-2]
                feature_dim = dims[-1]
                inputs = tf.reshape(inputs, (batch_dim, time_dim, feature_dim))
                if lengths is not None:
                    lengths = tf.reshape(lengths, (batch_dim,))
                inputs.set_shape((None, None, input_dim))

        fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_dim)
        bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_dim)
        if is_training:
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell,
                                                    input_keep_prob=input_keep_prob,
                                                    state_keep_prob=state_keep_prob,
                                                    variational_recurrent=variational_recurrent,
                                                    input_size=input_dim,
                                                    dtype=tf.float32)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell,
                                                    input_keep_prob=input_keep_prob,
                                                    state_keep_prob=state_keep_prob,
                                                    variational_recurrent=variational_recurrent,
                                                    input_size=input_dim,
                                                    dtype=tf.float32)
        with tf.variable_scope('lstm'):
            (fw, bw), (fw_states, bw_states) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs,
                                                                               sequence_length=lengths,
                                                                               dtype=tf.float32)
        with tf.variable_scope('attention'):
            hidden = tf.concat([fw, bw], axis=-1)
            if is_training and 0 <= output_keep_prob < 1:
                hidden = tf.nn.dropout(hidden, keep_prob=output_keep_prob,
                                       noise_shape=nnparser.nn.noise_shape(hidden, (None, 1, None)))
            attention_weight = tf.layers.dense(hidden, 1, activation=None, use_bias=False)
            attention_weight = tf.nn.softmax(attention_weight, axis=-2)
            weighted_hidden = attention_weight * hidden
            attention = tf.reduce_sum(weighted_hidden, axis=-2)
            cell_state = tf.concat([fw_states.c, bw_states.c], axis=-1)

        with tf.variable_scope('output'):
            output = tf.concat([attention, cell_state], axis=-1)
            if ndim > 3:
                output_original_shape = tf.concat([input_original_shape[:-2], [4 * hidden_dim]], axis=-1)
                output = tf.reshape(output, output_original_shape)

            if is_training and 0 <= output_keep_prob < 1:
                output = tf.nn.dropout(output, keep_prob=output_keep_prob,
                                       noise_shape=nnparser.nn.noise_shape(output, (None, 1, None)))
            output = tf.layers.dense(output, output_dim, activation=None, use_bias=True)

        return output
