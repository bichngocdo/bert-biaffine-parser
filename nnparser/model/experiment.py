import argparse
import ast
import configparser
import os.path
import pickle
import random
import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from core.data import Dataset
from core.data.batching import Bucketing, SimpleBatch

import nnparser.model.data_reader as data_reader
import nnparser.model.data_writer as data_writer
from nnparser.model.data_converter import DataConverter
from nnparser.model.data_encoder import DataEncoder
from nnparser.model.evaluator import Stats
from nnparser.model.network import BiaffineParserNetwork


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def dump_config(f, args):
    config_parser = configparser.ConfigParser()
    config_parser.optionxform = str
    for k, v in args.__dict__.items():
        if k != 'word_embeddings' and k != 'tag_embeddings':
            config_parser.defaults()[k] = v
    config_parser.write(f)


def load_config(fp):
    config_parser = configparser.ConfigParser()
    config_parser.optionxform = str
    args = argparse.Namespace()
    args_dict = vars(args)

    if config_parser.read(fp):
        for k, v in config_parser.defaults().items():
            try:
                args_dict[k] = ast.literal_eval(v)
            except:
                args_dict[k] = v
    return args


class Experiment(object):
    def __init__(self, args):
        self.args = args

        self.data_encoder = None
        self.network = None

        self.epoch = 0
        self.iteration = 0
        self.best_iteration = 0
        self.best_score = -1.

        self.train_stats = Stats('train')
        self.dev_stats = Stats('dev')

        self.config_path = os.path.join(args.model_dir, 'config.cfg')
        self.experiment_path = os.path.join(args.model_dir, 'experiment.pkl')
        self.data_model_path = os.path.join(args.model_dir, 'data.pkl')
        self.network_path = os.path.join(args.model_dir, 'model')
        self.best_network_path = os.path.join(args.model_dir, 'best')
        self.summary_path = os.path.join(args.model_dir, 'log')

        self.saver = None
        self.best_saver = None
        self.summary_writer = None

    def save(self):
        with open(self.experiment_path, 'wb') as f:
            pickle.dump(self.iteration, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.epoch, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.best_iteration, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.best_score, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(random.getstate(), f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(np.random.get_state(), f, pickle.HIGHEST_PROTOCOL)
        with open(self.config_path, 'w') as f:
            dump_config(f, self.args)

    def save_data_model(self):
        with open(self.data_model_path, 'wb') as f:
            pickle.dump(self.data_encoder, f, pickle.HIGHEST_PROTOCOL)

    def save_network(self, session):
        vars_path = os.path.join(self.network_path, 'network')
        self.saver.save(session, vars_path, global_step=self.network.iteration)

    def save_best_network(self, session):
        vars_path = os.path.join(self.best_network_path, 'network')
        self.best_saver.save(session, vars_path, global_step=self.network.iteration)

    def restore(self):
        with open(self.experiment_path, 'rb') as f:
            self.iteration = pickle.load(f)
            self.epoch = pickle.load(f)
            self.best_iteration = pickle.load(f)
            self.best_score = pickle.load(f)
            state = pickle.load(f)
            random.setstate(state)
            state = pickle.load(f)
            np.random.set_state(state)
        new_args = self.args
        self.update_arguments(new_args)

    def update_arguments(self, new_args):
        old_args = load_config(self.config_path)
        old_args.command = new_args.command
        if new_args.command == 'train':
            if new_args.max_epoch > old_args.max_epoch:
                old_args.max_epoch = new_args.max_epoch
        self.args = old_args

    def restore_data_model(self):
        with open(self.data_model_path, 'rb') as f:
            self.data_encoder = pickle.load(f)
        self.load_embeddings()

    def restore_network(self, session, is_best=True):
        self.network.initialize_global_variables(session)
        path = self.best_network_path if is_best else self.network_path
        vars_path = tf.train.latest_checkpoint(path)
        self.saver.restore(session, vars_path)
        print('Restore network from file %s' % vars_path)

    def load_embeddings(self):
        if self.args.word_emb_file is not None and self.args.bert_path is None:
            self.args.word_embeddings = self.data_encoder.load_word_embeddings(self.args.word_emb_file)
            assert self.args.word_dim == self.args.word_embeddings.shape[1]
        else:
            self.args.word_embeddings = None
        if self.args.tag_emb_file is not None:
            self.args.tag_embeddings = self.data_encoder.load_tag_embeddings(self.args.tag_emb_file)
            assert self.args.tag_dim == self.args.tag_embeddings.shape[1]
        else:
            self.args.tag_embeddings = None
        if self.args.bert_path:
            self.data_encoder.load_bert_encoder(self.args.bert_path, self.args.bert_lowercase)

    def init_data_model(self, raw_data):
        self.data_encoder = DataEncoder()
        self.data_encoder.word_cutoff_threshold = self.args.word_cutoff if self.args.word_cutoff >= 0 else float('inf')
        self.data_encoder.lowercase = args.lowercase
        self.data_encoder.init(raw_data)
        self.load_embeddings()

        self.args.no_words = len(self.data_encoder.word_encoder)
        self.args.no_tags = len(self.data_encoder.tag_encoder)
        self.args.no_chars = len(self.data_encoder.char_encoder)
        self.args.no_labels = len(self.data_encoder.label_encoder)

        print('No. words:        %d' % self.args.no_words)
        print('No. tags:         %d' % self.args.no_tags)
        print('No. characters:   %d' % self.args.no_chars)
        print('No. labels:       %d' % self.args.no_labels)

        self.save_data_model()

    def init_network(self, session):
        self.network = BiaffineParserNetwork(self.args)
        self.network.initialize_global_variables(session)
        var_list = [var for var in tf.global_variables() if 'pt_embeddings' not in var.name]
        var_list = [var for var in var_list if var not in self.network.ignore_variables]
        # for v in var_list:
        #     print(v.name)
        # exit()
        with tf.variable_scope('model_save'):
            self.saver = tf.train.Saver(var_list, max_to_keep=1)
        with tf.variable_scope('best_model_save'):
            self.best_saver = tf.train.Saver(var_list, max_to_keep=1)
        if self.args.command == 'train':
            self.summary_writer = tf.summary.FileWriter(self.summary_path, session.graph)

    def load_data(self, raw_data):
        return self.data_encoder.encode(raw_data)

    def __train_old(self, train_dataset, dev_dataset, interval,
                    train_batch_size, validate_batch_size, working_batch_size,
                    session):
        batches = train_dataset.get_batches(train_batch_size, shuffle=self.args.shuffle)

        data_converter = DataConverter()
        for batch in batches:
            vars = data_converter.convert(batch)
            self.iteration += 1
            start = time.time()
            outputs, loss, summary = self.network.train(vars, session)
            self.summary_writer.add_summary(summary=summary, global_step=self.iteration)
            end = time.time()
            self.train_stats.update(loss, end - start,
                                    batch[-2], batch[-1],
                                    outputs['edge_predictions'], outputs['label_predictions'])

            if self.iteration % interval == 0:
                self.__validate(dev_dataset, validate_batch_size, session)

    def __train(self, train_dataset, dev_dataset, interval,
                train_batch_size, validate_batch_size, working_batch_size,
                session):
        batches = train_dataset.get_batches(train_batch_size, shuffle=self.args.shuffle)

        data_converter = DataConverter()
        for batch in batches:
            self.iteration += 1
            lengths = [len(item) for item in batch[0]]
            mini_dataset = Dataset(batch,
                                   batch_generator=Bucketing(1, lengths))
            mini_batches = mini_dataset.get_batches(working_batch_size, shuffle=False)
            start = time.time()
            outputs = dict()
            for mini_batch in mini_batches:
                vars = data_converter.convert(mini_batch)
                mini_outputs, mini_loss = self.network.forward(vars, session)
                for k, v in mini_outputs.items():
                    outputs.setdefault(k, list()).extend(v)

            loss, summary = self.network.backward(session)
            self.summary_writer.add_summary(summary=summary, global_step=self.iteration)
            end = time.time()
            self.train_stats.update(loss, end - start,
                                    batch[-2], batch[-1],
                                    outputs['edge_predictions'], outputs['label_predictions'])

            if self.iteration % interval == 0:
                self.__validate(dev_dataset, validate_batch_size, session)

    def __validate(self, dev_dataset, validate_batch_size, session):
        batches = dev_dataset.get_batches(validate_batch_size, shuffle=False)
        data_converter = DataConverter()
        for batch in batches:
            vars = data_converter.convert(batch)
            start = time.time()
            outputs, loss = self.network.eval(vars, session)
            end = time.time()
            self.dev_stats.update(loss, end - start,
                                  batch[-2], batch[-1],
                                  outputs['edge_predictions'], outputs['label_predictions'])

        stats = OrderedDict()
        stats['lr'] = session.run(self.network.learning_rate)
        stats.update(self.train_stats.aggregate())
        stats.update(self.dev_stats.aggregate())
        print('Epoch %d, iteration %s:' % (self.epoch, self.iteration))
        for key, value in stats.items():
            print(key, '=', value)

        summary = self.network.make_train_summary(session, stats['train_uas'], stats['train_las'])
        self.summary_writer.add_summary(summary, global_step=self.iteration)
        summary = self.network.make_dev_summary(session, stats['dev_loss'], stats['dev_uas'], stats['dev_las'])
        self.summary_writer.add_summary(summary, global_step=self.iteration)

        self.save()
        self.save_network(session)

        if stats['dev_uas'] > self.best_score:
            self.best_score = stats['dev_uas']
            self.best_iteration = self.iteration
            self.save_best_network(session)
            print('Save the best model')
        else:
            print('Best score was %.5f at iteration %d' % (self.best_score, self.best_iteration))

    def __evaluate(self, dev_dataset, eval_batch_size, session):
        batches = dev_dataset.get_batches(eval_batch_size, shuffle=False)
        data_converter = DataConverter()
        results = dict()
        results['edge_predictions'] = list()
        results['label_predictions'] = list()
        for batch in batches:
            vars = data_converter.convert(batch)
            start = time.time()
            outputs, loss = self.network.eval(vars, session)
            end = time.time()
            self.dev_stats.update(loss, end - start,
                                  batch[-2], batch[-1],
                                  outputs['edge_predictions'], outputs['label_predictions'])
            results['edge_predictions'].extend(outputs['edge_predictions'].tolist())
            results['label_predictions'].extend(outputs['label_predictions'].tolist())

        stats = self.dev_stats.aggregate()
        for key, value in stats.items():
            print(key, '=', value)

        results['label_predictions'] = self.data_encoder.label_encoder.decode(results['label_predictions'])

        return results

    def train(self, session, train_data, dev_data,
              max_epoch=250, interval=100,
              train_batch_size=5000, validate_batch_size=10000, working_batch_size=5000):
        train_lengths = [len(item) for item in train_data[0]]
        train_dataset = Dataset(train_data,
                                batch_generator=Bucketing(args.num_train_buckets, train_lengths))
        dev_lengths = [len(item) for item in dev_data[0]]
        dev_dataset = Dataset(dev_data,
                              batch_generator=Bucketing(args.num_dev_buckets, dev_lengths))

        while self.epoch < max_epoch:
            self.epoch += 1
            self.__train(train_dataset, dev_dataset, interval,
                         train_batch_size, validate_batch_size,
                         working_batch_size,
                         session)

        if self.iteration % interval != 0:
            self.__validate(dev_dataset, validate_batch_size, session)

    def evaluate(self, session, dev_data, eval_batch_size=50):
        dev_dataset = Dataset(dev_data,
                              batch_generator=SimpleBatch(len(dev_data[0])))
        results = self.__evaluate(dev_dataset, eval_batch_size, session)

        return results


MAX_UNSIGNED_INT = 4294967295


def train(args):
    if args.seed is None:
        args.seed = random.randint(0, MAX_UNSIGNED_INT)
    if args.np_seed is None:
        args.np_seed = random.randint(0, MAX_UNSIGNED_INT)
    if args.tf_seed is None:
        args.tf_seed = random.randint(0, MAX_UNSIGNED_INT)
    random.seed(args.seed)
    np.random.seed(args.np_seed)
    tf.set_random_seed(args.tf_seed)

    experiment_dir = args.model_dir
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    experiment = Experiment(args)

    train_raw_data = data_reader.read_conllx(args.train_file)
    print('Load data from file %s, no. sentences: %d' % (args.train_file, len(train_raw_data['words'])))

    dev_raw_data = data_reader.read_conllx(args.dev_file)
    print('Load data from file %s, no. sentences: %d' % (args.dev_file, len(dev_raw_data['words'])))

    if args.test_file is not None:
        test_raw_data = data_reader.read_conllx(args.test_file)
        print('Load data from file %s, no. sentences: %d' % (args.test_file, len(test_raw_data['words'])))
    else:
        test_raw_data = None

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        if args.continue_training:
            experiment.restore()
            experiment.restore_data_model()
            experiment.init_network(session)
            experiment.restore_network(session, is_best=False)

            print('Current iteration: %7d' % experiment.iteration)
            print('Current epoch    : %7d' % experiment.epoch)
            print('Best iteration   : %7d' % experiment.best_iteration)
            print('Best score       : %6.5f' % experiment.best_score)
        else:
            experiment.init_data_model(train_raw_data)
            experiment.init_network(session)

        train_data = experiment.load_data(train_raw_data)
        dev_data = experiment.load_data(dev_raw_data)
        if test_raw_data is not None:
            test_data = experiment.load_data(test_raw_data)
        else:
            test_data = None

        experiment.train(session, train_data, dev_data,
                         max_epoch=args.max_epoch,
                         interval=args.interval,
                         train_batch_size=args.train_batch_size,
                         validate_batch_size=args.validate_batch_size,
                         working_batch_size=args.working_batch_size)

        if test_data is not None:
            experiment.restore_network(session, is_best=True)
            experiment.evaluate(session, test_data, args.eval_batch_size)


def evaluate(args):
    experiment = Experiment(args)

    dev_raw_data = data_reader.read_conllx(args.test_file)
    print('Load data from file %s, no. sentences: %d' % (args.test_file, len(dev_raw_data['words'])))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        experiment.restore()
        experiment.restore_data_model()
        experiment.init_network(session)
        experiment.restore_network(session, is_best=True)
        dev_data = experiment.load_data(dev_raw_data)
        results = experiment.evaluate(session, dev_data, args.batch_size)

    if args.output_file is not None:
        data_writer.write_conllx(args.output_file, dev_raw_data, results)


def infer(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='experiment.py',
                                     description='Biaffine Parser')
    subparsers = parser.add_subparsers()

    #################################################################################################
    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(command='train')

    # Input and output
    parser_train.add_argument('--train_file', type=str, required=True,
                              help='training data')
    parser_train.add_argument('--dev_file', type=str, required=True,
                              help='development data')
    parser_train.add_argument('--test_file', type=str, default=None,
                              help='test data')
    parser_train.add_argument('--word_emb_file', type=str, default=None,
                              help='pre-trained word embedding file')
    parser_train.add_argument('--tag_emb_file', type=str, default=None,
                              help='pre-trained tag embedding file')
    parser_train.add_argument('--bert_path', type=str, default=None,
                              help='bert model path')
    parser_train.add_argument('--bert_lowercase', type=str2bool, default=True,
                              help='bert model uncased')
    parser_train.add_argument('--model_dir', type=str, default='model',
                              help='folder to save the model')

    # Lexicon
    group = parser_train.add_argument_group('lexicon')
    group.add_argument('--word_cutoff', type=int, default=2,
                       help='word cutoff threshold')
    group.add_argument('--lowercase', type=str2bool, default=True,
                       help='lowercasing words')

    # Training hyperparameters
    group = parser_train.add_argument_group('training hyperparameters')
    group.add_argument('--seed', type=int, default=None,
                       help='random seed')
    group.add_argument('--np_seed', type=int, default=None,
                       help='Numpy random seed')
    group.add_argument('--tf_seed', type=int, default=None,
                       help='Tensorflow random seed')
    group.add_argument('--continue_training', action='store_true',
                       help='continue training')
    group.add_argument('--learning_rate', type=float, default=2e-3,
                       help='learning rate')
    group.add_argument('--decay_rate', type=float, default=0.75,
                       help='decay rate')
    group.add_argument('--decay_step', type=int, default=5000,
                       help='decay step')
    group.add_argument('--shuffle', type=str2bool, default=True,
                       help='shuffle data while training')
    group.add_argument('--train_batch_size', type=int, default=5000,
                       help='mini-batch size in number of tokens')
    group.add_argument('--validate_batch_size', type=int, default=5000,
                       help='mini-batch size in number of tokens')
    group.add_argument('--eval_batch_size', type=int, default=50,
                       help='mini-batch size in number of sentences')
    group.add_argument('--working_batch_size', type=int, default=5000,
                       help='mini-batch size in number of tokens')
    group.add_argument('--max_epoch', type=int, default=200,
                       help='number of training epochs')
    group.add_argument('--interval', type=int, default=100,
                       help='iteration interval to print info, evaluate and save')
    group.add_argument('--num_train_buckets', type=int, default=40,
                       help='number of length buckets')
    group.add_argument('--num_dev_buckets', type=int, default=10,
                       help='number of length buckets')

    # Network hyperparameters
    group = parser_train.add_argument_group('network hyperparameters')
    group.add_argument('--word_dim', type=int, default=100,
                       help='word embedding dimension')
    group.add_argument('--tag_dim', type=int, default=100,
                       help='tag embedding dimension')
    group.add_argument('--num_lstms', type=int, default=3,
                       help='number of LSTM layers')
    group.add_argument('--hidden_dim', type=int, default=400,
                       help='LTSM hidden dimension')
    group.add_argument('--num_mlps', type=int, default=1,
                       help='number of MLP layers')
    group.add_argument('--edge_mlp_dim', type=int, default=500,
                       help='edge MLP dimension')
    group.add_argument('--label_mlp_dim', type=int, default=100,
                       help='label MLP dimension')
    group.add_argument('--character_embeddings', type=str2bool, default=False,
                       help='use character-based word embeddings')
    group.add_argument('--char_dim', type=int, default=100,
                       help='character embedding dimension')
    group.add_argument('--char_hidden_dim', type=int, default=400,
                       help='character LSTM hidden dimension')

    group.add_argument('--bert_layers', type=int, nargs='+', default=[-1, -2],
                       help='layers used for embeddings in BERT')
    group.add_argument('--bert_fine_tuning', type=str2bool, default=True,
                       help='fine tuning BERT')

    # Optimization hyperparameters
    # group = parser_train.add_argument_group('optimization params')
    # group.add_argument('--loss', choices=[],
    #                    default='ranking_max', help='loss function')
    # group.add_argument('--optimizer', choices=['sgd', 'rmsprop', 'adam', 'adagrad', 'adadelta'],
    #                    default='adam', help='optimizer')

    # Regularization hyperparameters
    group = parser_train.add_argument_group('regularization hyperparameters')
    group.add_argument('--input_dropout', type=float, default=0.33,
                       help='input dropout probability')
    group.add_argument('--dropout', type=float, default=0.33,
                       help='dropout prob probability')
    group.add_argument('--recurrent_dropout', type=float, default=0.33,
                       help='recurrent dropout probability')
    group.add_argument('--char_input_dropout', type=float, default=0.,
                       help='character input dropout probability')
    group.add_argument('--char_recurrent_dropout', type=float, default=0.33,
                       help='character recurrent dropout probability')
    group.add_argument('--word_l2', type=float, default=0.,
                       help='word L2 regularization')
    group.add_argument('--tag_l2', type=float, default=0.,
                       help='tag L2 regularization')
    group.add_argument('--char_l2', type=float, default=0.,
                       help='word L2 regularization')
    group.add_argument('--l2', type=float, default=0.,
                       help='L2 regularization')
    group.add_argument('--max_norm', type=float, default=5.,
                       help='max norm')

    #################################################################################################
    parser_eval = subparsers.add_parser('eval')
    parser_eval.set_defaults(command='eval')

    parser_eval.add_argument('--test_file', type=str, required=True,
                             help='test data')
    parser_eval.add_argument('--output_file', type=str, default=None,
                             help='output file')
    parser_eval.add_argument('--model_dir', type=str, default='model',
                             help='folder of the model')

    parser_eval.add_argument('--batch_size', type=int, default=50,
                             help='mini-batch size in number of sentences')

    #################################################################################################
    parser_infer = subparsers.add_parser('infer')
    parser_infer.set_defaults(command='infer')

    parser_infer.add_argument('--test_file', type=str, required=True,
                              help='test data')
    parser_infer.add_argument('--output_file', type=str, default=None,
                              help='output file')
    parser_infer.add_argument('--model_dir', type=str, default='model',
                              help='folder of the model')

    parser_infer.add_argument('--batch_size', type=int, default=50,
                              help='mini-batch size in number of sentences')

    #################################################################################################
    args = parser.parse_args()
    print(args)

    if args.command == 'train':
        train(args)
    elif args.command == 'eval':
        evaluate(args)
    elif args.command == 'infer':
        infer(args)
