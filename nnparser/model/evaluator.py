from collections import OrderedDict


class Stats(object):
    def __init__(self, name):
        self.name = name

        self.loss = 0.
        self.time = 0.
        self.num_iterations = 0
        self.num_sentences = 0
        self.num_tokens = 0
        self.num_correct_sentences = 0
        self.num_correct_heads = 0
        self.num_correct_labels = 0

    def reset(self):
        self.loss = 0.
        self.time = 0.
        self.num_iterations = 0
        self.num_sentences = 0
        self.num_tokens = 0
        self.num_correct_heads = 0
        self.num_correct_labels = 0

    def update(self, loss, time_elapsed, batch_gold_heads, batch_gold_labels, batch_heads, batch_labels):
        self.loss += loss
        self.time += time_elapsed
        self.num_iterations += 1

        for gold_heads, gold_labels, heads, labels in \
                zip(batch_gold_heads, batch_gold_labels, batch_heads, batch_labels):
            self.num_sentences += 1
            for ghead, glabel, head, label in \
                    zip(gold_heads, gold_labels, heads, labels):
                self.num_tokens += 1
                if ghead == head:
                    self.num_correct_heads += 1
                    if glabel == label:
                        self.num_correct_labels += 1

    def aggregate(self):
        results = OrderedDict()
        results['%s_loss' % self.name] = self.loss / self.num_iterations
        results['%s_rate' % self.name] = self.num_sentences / self.time
        results['%s_uas' % self.name] = self.num_correct_heads / self.num_tokens \
            if self.num_tokens > 0 else float('NaN')
        results['%s_las' % self.name] = self.num_correct_labels / self.num_tokens \
            if self.num_tokens > 0 else float('NaN')
        self.reset()
        return results
