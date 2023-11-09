from core.fileio import CoNLLFile


def read_conll(fp, word_col, tag_col, head_col, label_col):
    with open(fp, 'r', encoding='utf-8') as f:
        f = CoNLLFile(f)
        results = {
            'words': list(),
            'tags': list(),
            'chars': list(),
            'heads': list(),
            'labels': list()
        }
        for block in f:
            words = list()
            tags = list()
            chars = list()
            heads = list()
            labels = list()

            for items in block:
                words.append(items[word_col])
                tags.append(items[tag_col])
                chars.append([c for c in items[word_col]])
                heads.append(int(items[head_col]))
                labels.append(items[label_col])
            results['words'].append(words)
            results['tags'].append(tags)
            results['chars'].append(chars)
            results['heads'].append(heads)
            results['labels'].append(labels)

        return results


def read_conllx(fp):
    return read_conll(fp, word_col=1, tag_col=4, head_col=6, label_col=7)
