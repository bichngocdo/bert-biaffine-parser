import os

import core.nn
import numpy as np
import tensorflow as tf
from bert.modeling import BertConfig, BertModel

import nnparser.bert.model as bert_modeling
from nnparser.nn import character_based_embeddings


class BiaffineParserNetwork(object):
    def __init__(self, args):
        self.ignore_variables = list()

        self._build_embeddings(args)

        self.train, self.forward, self.backward = self._build_train_functions(args)

        self.eval = self._build_eval_function(args)

        self.make_train_summary = self._build_train_summary_function()
        self.make_dev_summary = self._build_dev_summary_function()

    def initialize_global_variables(self, session):
        feed_dict = dict()
        if self.word_pt_embeddings is not None:
            feed_dict[self.word_pt_embeddings_ph] = self._word_pt_embeddings
        if self.tag_pt_embeddings is not None:
            feed_dict[self.tag_pt_embeddings_ph] = self._tag_pt_embeddings
        session.run(tf.global_variables_initializer(), feed_dict=feed_dict)

    def _build_train_summary_function(self):
        with tf.variable_scope('train_summary/'):
            x_uas = tf.placeholder(tf.float32,
                                   shape=None,
                                   name='x_uas')
            x_las = tf.placeholder(tf.float32,
                                   shape=None,
                                   name='x_las')

            tf.summary.scalar('train_acc', x_uas, collections=['train_summary'])
            tf.summary.scalar('train_uas', x_las, collections=['train_summary'])

            summary = tf.summary.merge_all(key='train_summary')

        def f(session, uas, las):
            feed_dict = {
                x_uas: uas,
                x_las: las
            }
            return session.run(summary, feed_dict=feed_dict)

        return f

    def _build_dev_summary_function(self):
        with tf.variable_scope('dev_summary/'):
            x_loss = tf.placeholder(tf.float32,
                                    shape=None,
                                    name='x_loss')
            x_uas = tf.placeholder(tf.float32,
                                   shape=None,
                                   name='x_uas')
            x_las = tf.placeholder(tf.float32,
                                   shape=None,
                                   name='x_las')

            tf.summary.scalar('dev_loss', x_loss, collections=['dev_summary'])
            tf.summary.scalar('dev_uas', x_uas, collections=['dev_summary'])
            tf.summary.scalar('dev_acc', x_las, collections=['dev_summary'])

            summary = tf.summary.merge_all(key='dev_summary')

        def f(session, loss, uas, las):
            feed_dict = {
                x_loss: loss,
                x_uas: uas,
                x_las: las
            }
            return session.run(summary, feed_dict=feed_dict)

        return f

    def _build_placeholders(self, args):
        # x_word has shape (batch_size, max_length)
        x_word = tf.placeholder(tf.int32,
                                shape=(None, None),
                                name='x_word')
        if self.word_pt_embeddings is not None:
            x_pt_word = tf.placeholder(tf.int32,
                                       shape=(None, None),
                                       name='x_pt_word')
        else:
            x_pt_word = None

        # x_tag has shape (batch_size, max_length)
        x_tag = tf.placeholder(tf.int32,
                               shape=(None, None),
                               name='x_tag')
        if self.tag_pt_embeddings is not None:
            x_pt_tag = tf.placeholder(tf.int32,
                                      shape=(None, None),
                                      name='x_pt_tag')
        else:
            x_pt_tag = None

        # x_char has shape (batch_size, max_length, max_char_length)
        if self.character_embeddings is not None:
            x_char = tf.placeholder(tf.int32,
                                    shape=(None, None, None),
                                    name='x_char')
        else:
            x_char = None

        # x_bert_ids has shape (batch_size, max_token_length)
        # x_bert_mask has shape (batch_size, max_token_length)
        # x_bert_types has shape (batch_size, max_token_length)
        # x_bert_indices has shape (batch_size, max_length, max_subword_length)
        if args.bert_path is not None:
            x_bert_id = tf.placeholder(tf.int32,
                                       shape=(None, None),
                                       name='x_bert_ids')
            x_bert_mask = tf.placeholder(tf.int32,
                                         shape=(None, None),
                                         name='x_bert_mask')
            x_bert_type = tf.placeholder(tf.int32,
                                         shape=(None, None),
                                         name='x_bert_types')
            x_bert_index = tf.placeholder(tf.int32,
                                          shape=(None, None, None),
                                          name='x_bert_indices')
        else:
            x_bert_id = None
            x_bert_mask = None
            x_bert_type = None
            x_bert_index = None

        # y_head has shape (batch_size, max_length)
        y_edge = tf.placeholder(tf.int32,
                                shape=(None, None),
                                name='y_edge')

        # y_label_in has shape (batch_size, max_length)
        y_label = tf.placeholder(tf.int32,
                                 shape=(None, None),
                                 name='y_label')

        return [x_word, x_pt_word, x_tag, x_pt_tag, x_char,
                x_bert_id, x_bert_mask, x_bert_type, x_bert_index,
                y_edge, y_label]

    def _build_embeddings(self, args):
        with tf.variable_scope('embeddings'):
            if not args.bert_path:
                if args.word_embeddings is not None:
                    self.word_embeddings = tf.get_variable('word_embeddings',
                                                           shape=(args.no_words, args.word_dim),
                                                           dtype=tf.float32,
                                                           initializer=tf.zeros_initializer,
                                                           regularizer=tf.contrib.layers.l2_regularizer(args.word_l2))
                    self.word_pt_embeddings_ph = tf.placeholder(tf.float32,
                                                                shape=args.word_embeddings.shape,
                                                                name='word_pt_embeddings_ph')
                    self.word_pt_embeddings = tf.Variable(self.word_pt_embeddings_ph,
                                                          name='word_pt_embeddings',
                                                          trainable=False)
                    self._word_pt_embeddings = args.word_embeddings
                else:
                    self.word_embeddings = \
                        tf.get_variable('word_embeddings',
                                        shape=(args.no_words, args.word_dim),
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer,
                                        regularizer=tf.contrib.layers.l2_regularizer(args.word_l2))
                    self.word_pt_embeddings = None
            else:
                if args.word_dim > 0:
                    self.word_embeddings = tf.get_variable('word_embeddings',
                                                           shape=(args.no_words, args.word_dim),
                                                           dtype=tf.float32,
                                                           initializer=tf.random_normal_initializer,
                                                           regularizer=tf.contrib.layers.l2_regularizer(args.word_l2))
                else:
                    self.word_embeddings = None
                self.word_pt_embeddings = None

            if args.tag_embeddings is not None:
                self.tag_embeddings = tf.get_variable('tag_embeddings',
                                                      shape=(args.no_tags, args.tag_dim),
                                                      dtype=tf.float32,
                                                      initializer=tf.zeros_initializer,
                                                      regularizer=tf.contrib.layers.l2_regularizer(args.tag_l2))
                self.tag_pt_embeddings_ph = tf.placeholder(tf.float32,
                                                           shape=args.tag_embeddings.shape,
                                                           name='tag_pt_embeddings_ph')
                self.tag_pt_embeddings = tf.Variable(self.tag_pt_embeddings_ph,
                                                     name='tag_pt_embeddings',
                                                     trainable=False)
                self._tag_pt_embeddings = args.tag_embeddings
            else:
                self.tag_embeddings = \
                    tf.get_variable('tag_embeddings',
                                    shape=(args.no_tags, args.tag_dim),
                                    dtype=tf.float32,
                                    initializer=tf.random_normal_initializer,
                                    regularizer=tf.contrib.layers.l2_regularizer(args.tag_l2))
                self.tag_pt_embeddings = None

            if not args.bert_path:
                if args.character_embeddings:
                    self.character_embeddings = \
                        tf.get_variable('characters_embeddings',
                                        shape=(args.no_chars, args.char_dim),
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer,
                                        regularizer=tf.contrib.layers.l2_regularizer(args.char_l2))
                else:
                    self.character_embeddings = None
            else:
                self.character_embeddings = None

    def _build_input_layers(self, args, is_training):
        def f(x_word, x_pt_word, x_tag, x_pt_tag, x_char,
              x_bert_id, x_bert_mask, x_bert_type, x_bert_index):

            if not args.bert_path:
                # Word embeddings
                e_word = tf.nn.embedding_lookup(self.word_embeddings, x_word)

                # Word pre-trained embeddings
                if self.word_pt_embeddings is not None:
                    e_pt_word = tf.nn.embedding_lookup(self.word_pt_embeddings, x_pt_word)
                    e_word += e_pt_word

                if is_training:
                    e_word = tf.nn.dropout(e_word,
                                           keep_prob=1 - args.input_dropout,
                                           noise_shape=core.nn.noise_shape(e_word, (None, None, 1)))

                # Character-based word embeddings
                if args.character_embeddings:
                    index_mask = tf.greater(x_char, 0)
                    lengths = tf.reduce_sum(tf.cast(index_mask, tf.int32), axis=-1)
                    e_char = tf.nn.embedding_lookup(self.character_embeddings, x_char)
                    e_char_word = character_based_embeddings(e_char,
                                                             args.char_dim,
                                                             args.char_hidden_dim,
                                                             args.word_dim,
                                                             lengths, is_training,
                                                             input_keep_prob=1 - args.char_input_dropout,
                                                             state_keep_prob=1 - args.char_recurrent_dropout,
                                                             output_keep_prob=1 - args.dropout,
                                                             variational_recurrent=True)
                    e_word += e_char_word
            else:
                # Word embeddings
                if self.word_embeddings is not None:
                    e_word = tf.nn.embedding_lookup(self.word_embeddings, x_word)
                else:
                    e_word = None

                # BERT embeddings
                bert_layer = self._build_bert_layer(args, is_training,
                                                    trainable=args.bert_fine_tuning)
                e_subword = bert_layer(x_bert_id, x_bert_mask, x_bert_type)

                batch_size, max_length, max_word_length = tf.unstack(tf.shape(x_bert_index))
                dim = e_subword.get_shape()[-1]

                index_mask = tf.greater_equal(x_bert_index, 0)
                indices = tf.where(index_mask, x_bert_index, tf.zeros_like(x_bert_index))

                indices = tf.reshape(indices, (batch_size, max_length * max_word_length))
                e_subword = tf.batch_gather(e_subword, indices)
                e_subword = tf.reshape(e_subword, (batch_size, max_length, max_word_length, dim))
                subword_mask = tf.tile(tf.expand_dims(index_mask, -1), (1, 1, 1, dim))
                e_subword = tf.where(subword_mask, e_subword, tf.zeros_like(e_subword))
                no_subwords = tf.reduce_sum(tf.cast(subword_mask, tf.float32), axis=-2)
                e_word_bert = tf.reduce_sum(e_subword, axis=-2) / no_subwords
                e_word_bert = tf.where(tf.not_equal(no_subwords, 0), e_word_bert, tf.zeros_like(e_word_bert))

                if self.word_embeddings is not None:
                    e_word = tf.concat([e_word, e_word_bert], axis=-1)
                else:
                    e_word = e_word_bert

                if is_training:
                    e_word = tf.nn.dropout(e_word,
                                           keep_prob=1 - args.input_dropout,
                                           noise_shape=core.nn.noise_shape(e_word, (None, None, 1)))

            # Tag embeddings
            e_tag = tf.nn.embedding_lookup(self.tag_embeddings, x_tag)

            # Tag pre-trained embeddings
            if self.tag_pt_embeddings is not None:
                e_pt_tag = tf.nn.embedding_lookup(self.tag_pt_embeddings, x_pt_tag)
                e_tag += e_pt_tag

            if is_training:
                # TODO: Important!!! Dropout whole words/tags, dropout twice
                e_tag = tf.nn.dropout(e_tag,
                                      keep_prob=1 - args.input_dropout,
                                      noise_shape=core.nn.noise_shape(e_tag, (None, None, 1)))

            input = tf.concat([e_word, e_tag], axis=-1)

            return input

        return f

    def _build_bert_layer(self, args, is_training, trainable=True):
        def f(x_bert_id, x_bert_mask, x_bert_type):
            bert_config_file = os.path.join(args.bert_path, 'bert_config.json')
            bert_checkpoint = os.path.join(args.bert_path, 'bert_model.ckpt')
            bert_config = BertConfig.from_json_file(bert_config_file)
            with tf.variable_scope('bert'):
                model = BertModel(bert_config, is_training=is_training,
                                  input_ids=x_bert_id, input_mask=x_bert_mask, token_type_ids=x_bert_type)
                bert_modeling.initialize_from_checkpoint(bert_checkpoint)
                if not trainable:
                    self.ignore_variables += tf.trainable_variables(tf.get_variable_scope().name)
            all_layers = model.get_all_encoder_layers()
            selected_layers = [all_layers[i] for i in args.bert_layers]
            return tf.add_n(selected_layers) / len(args.bert_layers)

        return f

    def _build_lstm_layers(self, args, is_training):
        def f(input, lengths):
            hidden = input
            with tf.variable_scope('lstms'):
                for i in range(args.num_lstms):
                    # TODO: Important!!! Don't use TF DropoutWrapper input and output dropouts
                    # because it applies the same mask (aka positions) for the whole batch,
                    # which leads to worse performance.
                    if is_training:
                        hidden = tf.nn.dropout(hidden,
                                               keep_prob=1 - args.dropout,
                                               noise_shape=core.nn.noise_shape(hidden, (None, 1, None)))
                    fw_cell = tf.nn.rnn_cell.LSTMCell(args.hidden_dim,
                                                      initializer=tf.initializers.orthogonal)
                    if is_training:
                        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell,
                                                                state_keep_prob=1 - args.recurrent_dropout,
                                                                # output_keep_prob=1 - args.dropout,
                                                                variational_recurrent=True,
                                                                dtype=tf.float32)

                    bw_cell = tf.nn.rnn_cell.LSTMCell(args.hidden_dim,
                                                      initializer=tf.initializers.orthogonal)
                    if is_training:
                        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell,
                                                                state_keep_prob=1 - args.recurrent_dropout,
                                                                # output_keep_prob=1 - args.dropout,
                                                                variational_recurrent=True,
                                                                dtype=tf.float32)
                    with tf.variable_scope('lstm%d' % i):
                        (fw, bw), (fw_s, bw_s) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, hidden,
                                                                                 sequence_length=lengths,
                                                                                 dtype=tf.float32)
                    hidden = tf.concat([fw, bw], axis=-1)

            with tf.variable_scope('mlp'):
                if is_training:
                    hidden = tf.nn.dropout(hidden,
                                           keep_prob=1 - args.dropout,
                                           noise_shape=core.nn.noise_shape(hidden, (None, 1, None)))
                for _ in range(args.num_mlps):
                    kernel = orthogonal((2 * args.hidden_dim, args.edge_mlp_dim + args.label_mlp_dim))
                    kernel = np.concatenate([kernel, kernel], axis=-1)
                    hidden = tf.layers.dense(hidden,
                                             units=2 * (args.edge_mlp_dim + args.label_mlp_dim),
                                             activation=tf.nn.leaky_relu,
                                             kernel_initializer=tf.initializers.constant(kernel),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(args.l2))
                    if is_training:
                        hidden = tf.nn.dropout(hidden,
                                               keep_prob=1 - args.dropout,
                                               noise_shape=core.nn.noise_shape(hidden, (None, 1, None)))

                h_dep, h_head = tf.split(hidden, 2, axis=-1)

            return h_dep, h_head

        return f

    def _build_edge_classifier(self, args, is_training):
        def f(h_dep, h_head):
            with tf.variable_scope('edge_classifier'):
                weight = tf.get_variable('weight',
                                         shape=(args.edge_mlp_dim + 1, args.edge_mlp_dim),
                                         dtype=tf.float32,
                                         initializer=tf.zeros_initializer,
                                         regularizer=tf.contrib.layers.l2_regularizer(args.l2))
                h_dep = tf.pad(h_dep, [[0, 0], [0, 0], [0, 1]],
                               mode='constant', constant_values=1.)
                return core.nn.bilinear(h_dep, h_head, weight)

        return f

    def _build_label_classifier(self, args, is_training):
        def f(h_dep, h_head):
            with tf.variable_scope('label_classifier'):
                weight = tf.get_variable('weight',
                                         shape=(args.no_labels, args.label_mlp_dim + 2, args.label_mlp_dim + 2),
                                         dtype=tf.float32,
                                         initializer=tf.zeros_initializer,
                                         regularizer=tf.contrib.layers.l2_regularizer(args.l2))
                h_dep = tf.pad(h_dep, [[0, 0], [0, 0], [0, 2]],
                               mode='constant', constant_values=1.)
                h_head = tf.pad(h_head, [[0, 0], [0, 0], [0, 2]],
                                mode='constant', constant_values=1.)
                y = core.nn.bilinear(h_dep, h_head, weight)
                y = tf.transpose(y, (0, 1, 3, 2))
                return y

        return f

    def _build(self, args, is_training):
        with tf.variable_scope('placeholders'):
            x_word, x_pt_word, x_tag, x_pt_tag, x_char, \
                x_bert_id, x_bert_mask, x_bert_type, x_bert_index, \
                y_edge, y_label = self._build_placeholders(args)

        input_layers = self._build_input_layers(args, is_training)
        lstm_layers = self._build_lstm_layers(args, is_training)
        edge_classifier = self._build_edge_classifier(args, is_training)
        label_classifier = self._build_label_classifier(args, is_training)

        with tf.variable_scope('input_layers', reuse=tf.AUTO_REUSE):
            inputs = input_layers(x_word, x_pt_word, x_tag, x_pt_tag, x_char,
                                  x_bert_id, x_bert_mask, x_bert_type, x_bert_index)
            input_mask = tf.greater(x_word, 0)

            lengths = tf.reduce_sum(tf.cast(input_mask, tf.int32), axis=-1)

        with tf.variable_scope('hidden_layers', reuse=tf.AUTO_REUSE):
            h_dep, h_head = lstm_layers(inputs, lengths)
            h_dep_edge, h_dep_label = tf.split(h_dep, [args.edge_mlp_dim, args.label_mlp_dim], axis=-1)
            h_head_edge, h_head_label = tf.split(h_dep, [args.edge_mlp_dim, args.label_mlp_dim], axis=-1)

            output_mask = input_mask[:, 1:]
            edge_output_mask = tf.logical_and(tf.expand_dims(input_mask, 2),
                                              tf.expand_dims(input_mask, 1))
            edge_output_mask = edge_output_mask[:, 1:, :]
            label_output_mask = tf.tile(tf.expand_dims(edge_output_mask, -1),
                                        (1, 1, 1, args.no_labels))

            with tf.variable_scope('scoring_layer', reuse=tf.AUTO_REUSE):
                edge_logits = edge_classifier(h_dep_edge, h_head_edge)
                edge_logits = edge_logits[:, 1:, :]
                edge_logits = tf.where(edge_output_mask, edge_logits, tf.ones_like(edge_logits) * float('-inf'))
                edge_scores = tf.nn.softmax(edge_logits, axis=-1)

                label_logits = label_classifier(h_dep_label, h_head_label)
                label_logits = label_logits[:, 1:, :]
                label_logits = tf.where(label_output_mask, label_logits, tf.ones_like(label_logits) * float('-inf'))
                label_scores = tf.nn.softmax(label_logits, axis=-1)

        with tf.variable_scope('output_layers', reuse=tf.AUTO_REUSE):
            edge_predictions = tf.cast(tf.argmax(edge_scores, axis=-1), tf.int32)

            gold_tree_label_logits = _extract_label_logits(label_logits, y_edge)
            pred_tree_label_scores = _extract_label_logits(label_scores, edge_predictions)
            label_predictions = tf.cast(tf.argmax(pred_tree_label_scores, axis=-1), tf.int32)

        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            edge_loss = _loss_with_mask(y_edge, edge_logits, output_mask)
            label_loss = _loss_with_mask(y_label, gold_tree_label_logits, output_mask)
            loss = edge_loss + label_loss

        inputs = [x_word, x_pt_word, x_tag, x_pt_tag, x_char,
                  x_bert_id, x_bert_mask, x_bert_type, x_bert_index,
                  y_edge, y_label]
        outputs = {
            'edge_scores': edge_scores,
            'label_scores': pred_tree_label_scores,
            'edge_predictions': edge_predictions,
            'label_predictions': label_predictions
        }

        return inputs, outputs, loss

    def _build_train_function(self, args):
        with tf.name_scope('train'):
            inputs, outputs, loss = self._build(args, is_training=True)
            trainable_variables = [v for v in tf.trainable_variables()
                                   if v not in self.ignore_variables]

            self.iteration = tf.Variable(-1, name='iteration', trainable=False)
            with tf.variable_scope('learning_rate'):
                self.learning_rate = tf.train.exponential_decay(learning_rate=args.learning_rate,
                                                                global_step=self.iteration,
                                                                decay_steps=args.decay_step,
                                                                decay_rate=args.decay_rate)

            with tf.variable_scope('train_summary/'):
                tf.summary.scalar('lr', self.learning_rate, collections=['train_instant_summary'])
                tf.summary.scalar('train_loss', loss, collections=['train_instant_summary'])
                summary = tf.summary.merge_all(key='train_instant_summary')

            optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                               beta2=0.9)

            gradients_vars = optimizer.compute_gradients(loss, trainable_variables)
            if args.max_norm is not None:
                with tf.variable_scope('max_norm'):
                    gradients = [gv[0] for gv in gradients_vars]
                    gradients, _ = tf.clip_by_global_norm(gradients, args.max_norm)
                    gradients_vars = [(g, gv[1]) for g, gv in zip(gradients, gradients_vars)]

            with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
                train_step = optimizer.apply_gradients(gradients_vars, global_step=self.iteration)

        def f(vars, session):
            feed_dict = dict()
            for input, var in zip(inputs, vars):
                if input is not None:
                    feed_dict[input] = var
            _, output_, loss_, summary_ = session.run([train_step, outputs, loss, summary],
                                                      feed_dict=feed_dict)
            return output_, loss_, summary_

        return f

    def _build_train_functions(self, args):
        with tf.name_scope('train'):
            inputs, outputs, loss = self._build(args, is_training=True)
            trainable_variables = [v for v in tf.trainable_variables()
                                   if v not in self.ignore_variables]

            with tf.variable_scope('accumulation'):
                acc_loss = tf.Variable(0., name='acc_loss', trainable=False)
                acc_gradients = [tf.Variable(tf.zeros_like(var), trainable=False)
                                 for var in trainable_variables]
                acc_counter = tf.Variable(0., name='acc_counter', trainable=False)

            self.iteration = tf.Variable(-1, name='iteration', trainable=False)
            with tf.variable_scope('learning_rate'):
                self.learning_rate = tf.train.exponential_decay(learning_rate=args.learning_rate,
                                                                global_step=self.iteration,
                                                                decay_steps=args.decay_step,
                                                                decay_rate=args.decay_rate)

            with tf.variable_scope('train_summary/'):
                tf.summary.scalar('lr', self.learning_rate, collections=['train_instant_summary'])
                tf.summary.scalar('train_loss', acc_loss, collections=['train_instant_summary'])
                summary = tf.summary.merge_all(key='train_instant_summary')

            optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                               beta2=0.9)

            gradients_vars = optimizer.compute_gradients(loss, trainable_variables)
            for i in range(len(gradients_vars)):
                g, v = gradients_vars[i]
                if g is None:
                    gradients_vars[i] = (tf.zeros_like(v), v)
            gradients = [gv[0] for gv in gradients_vars]
            with tf.variable_scope('accumulation'):
                acc_gradients_ops = [tf.assign_add(acc_g, g)
                                     for acc_g, g in zip(acc_gradients, gradients)]
                acc_loss_ops = tf.assign_add(acc_loss, loss)
                acc_counter_ops = tf.assign_add(acc_counter, 1.)
                zero_gradients_ops = [tf.assign(acc_g, tf.zeros_like(v))
                                      for acc_g, v in zip(acc_gradients, trainable_variables)]
                zero_loss_ops = tf.assign(acc_loss, 0.)
                zero_counter_ops = tf.assign(acc_counter, 0.)

            acc_gradients_vars = [(g, v) for g, v in zip(acc_gradients, trainable_variables)]
            if args.max_norm is not None:
                with tf.variable_scope('max_norm'):
                    clipped_acc_gradients, _ = tf.clip_by_global_norm(acc_gradients, args.max_norm)
                    acc_gradients_vars = [(g, v) for g, v in zip(clipped_acc_gradients, trainable_variables)]
                    clipped_gradients, _ = tf.clip_by_global_norm(gradients, args.max_norm)
                    gradients_vars = [(g, gv[1]) for g, gv in zip(clipped_gradients, gradients_vars)]

            with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
                train_step = optimizer.apply_gradients(gradients_vars, global_step=self.iteration)
                forward_ops = [acc_loss_ops, acc_gradients_ops, acc_counter_ops]
                backward_ops = [tf.assign(acc_loss, acc_loss / acc_counter),
                                [tf.assign(g, (g / acc_counter)) for g in acc_gradients],
                                optimizer.apply_gradients(acc_gradients_vars, global_step=self.iteration)]
                reset_ops = [zero_loss_ops, zero_gradients_ops, zero_counter_ops]

        def f_train(vars, session):
            feed_dict = dict()
            for input, var in zip(inputs, vars):
                if input is not None:
                    feed_dict[input] = var
            _, output_, loss_, summary_ = session.run([train_step, outputs, loss, summary],
                                                      feed_dict=feed_dict)
            return output_, loss_, summary_

        def f_forward(vars, session):
            feed_dict = dict()
            for input, var in zip(inputs, vars):
                if input is not None:
                    feed_dict[input] = var
            _, output_, loss_, = session.run([forward_ops, outputs, loss],
                                             feed_dict=feed_dict)
            return output_, loss_

        def f_backward(session):
            _, loss_, summary_ = session.run([backward_ops, acc_loss, summary])
            session.run(reset_ops)
            return loss_, summary_

        return f_train, f_forward, f_backward

    def _build_eval_function(self, args):
        with tf.name_scope('eval'):
            inputs, outputs, loss = self._build(args, is_training=False)

        def f(vars, session):
            feed_dict = dict()
            for input, var in zip(inputs, vars):
                if input is not None:
                    feed_dict[input] = var
            return session.run([outputs, loss], feed_dict=feed_dict)

        return f


def _extract_label_logits(label_logits, heads):
    batch_size, no_deps, no_heads, no_labels = tf.unstack(tf.shape(label_logits))
    label_logits = tf.reshape(label_logits, (-1, no_heads, no_labels))
    indices = tf.concat([tf.reshape(tf.range(batch_size * no_deps), (-1, 1)),
                         tf.reshape(heads, (-1, 1))], axis=-1)
    return tf.reshape(tf.gather_nd(label_logits, indices), (batch_size, no_deps, no_labels))


def _loss_with_mask(gold_labels, logits, mask):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gold_labels, logits=logits)
    loss = tf.reduce_mean(tf.boolean_mask(cross_entropy, mask))
    return loss


def orthogonal(shape):
    num_rows = 1
    for dim in shape[:-1]:
        num_rows *= dim
    num_cols = shape[-1]
    flat_shape = (num_cols, num_rows) if num_rows < num_cols else (num_rows, num_cols)

    # Generate a random matrix
    a = np.random.normal(0., 1., flat_shape)
    # Compute the qr factorization
    q, r = np.linalg.qr(a, mode='reduced')
    # Make Q uniform
    d = np.diag(r)
    ph = d / np.abs(d)
    q *= ph
    if num_rows < num_cols:
        q = np.transpose(q)
    return np.reshape(q, shape)
