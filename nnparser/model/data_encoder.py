import numpy as np
from core.data.encoders import BertEncoder, VocabEncoder, build_vocab
from core.fileio import read_embeddings


class DataEncoder(object):
    def __init__(self):
        self.word_encoder = None
        self.tag_encoder = None
        self.char_encoder = None
        self.label_encoder = None

        self.pt_word_encoder = None
        self.pt_tag_encoder = None
        self.bert_encoder = None

        self.word_cutoff_threshold = 0
        self.lowercase = False

        self.NONE = '[NONE]'
        self.UNK = '[UNK]'
        self.ROOT = '[ROOT]'

    def init(self, raw_data):
        words = raw_data['words']
        chars = raw_data['chars']
        tags = raw_data['tags']
        labels = raw_data['labels']

        if self.lowercase:
            old_words = words
            words = list()
            for item in old_words:
                if isinstance(item, list):
                    words.append([word.lower() for word in item])
                else:
                    words.append(item.lower())

        word_vocab = [self.NONE, self.ROOT, self.UNK] + build_vocab(words, cutoff_threshold=self.word_cutoff_threshold)
        tag_vocab = [self.NONE, self.ROOT, self.UNK] + build_vocab(tags)
        char_vocab = [self.NONE, self.ROOT, self.UNK] + build_vocab(chars)
        label_vocab = [self.NONE, self.UNK] + build_vocab(labels)

        self.word_encoder = VocabEncoder(word_vocab, oov_token=self.UNK)
        self.tag_encoder = VocabEncoder(tag_vocab, oov_token=self.UNK)
        self.char_encoder = VocabEncoder(char_vocab, oov_token=self.UNK)
        self.label_encoder = VocabEncoder(label_vocab, oov_token=self.UNK)

    def __load_pretrained_embeddings(self, encoder_type, path, have_unk=False):
        vocab, embeddings = read_embeddings(path, mode='txt')
        vocab = [self.NONE, self.ROOT, self.UNK] + vocab
        if have_unk:
            embeddings = np.pad(embeddings, ((2, 0), (0, 0)), 'constant', constant_values=0)
        else:
            embeddings = np.pad(embeddings, ((3, 0), (0, 0)), 'constant', constant_values=0)
        embeddings /= np.std(embeddings)
        encoder = VocabEncoder(vocab, oov_token=self.UNK)
        self.__setattr__(encoder_type, encoder)
        return embeddings

    def load_word_embeddings(self, path):
        return self.__load_pretrained_embeddings('pt_word_encoder', path)

    def load_tag_embeddings(self, path):
        return self.__load_pretrained_embeddings('pt_tag_encoder', path)

    def load_bert_encoder(self, path, lowercase):
        self.bert_encoder = BertEncoder(path, lowercase)

    def encode(self, raw_data):
        results = list()

        words = raw_data['words']
        chars = raw_data['chars']
        tags = raw_data['tags']
        heads = raw_data['heads']
        labels = raw_data['labels']

        if self.lowercase:
            old_words = words
            words = list()
            for item in old_words:
                if isinstance(item, list):
                    words.append([word.lower() for word in item])
                else:
                    words.append(item.lower())

        words_ = list()
        for sentence in words:
            words_.append([self.ROOT] + sentence)
        words = words_
        tags_ = list()
        for sentence in tags:
            tags_.append([self.ROOT] + sentence)
        tags = tags_
        chars_ = list()
        for sentence in chars:
            chars_.append([[self.ROOT]] + sentence)
        chars = chars_

        results.append(self.word_encoder.encode(words))
        if self.pt_word_encoder:
            results.append(self.pt_word_encoder.encode(words))
        else:
            results.append(None)

        results.append(self.tag_encoder.encode(tags))
        if self.pt_tag_encoder:
            results.append(self.pt_tag_encoder.encode(tags))
        else:
            results.append(None)

        results.append(self.char_encoder.encode(chars))

        if self.bert_encoder:
            results.extend(self.bert_encoder.encode(words))
        else:
            results.extend([None] * 4)

        results.append(heads)
        results.append(self.label_encoder.encode(labels))

        return results
