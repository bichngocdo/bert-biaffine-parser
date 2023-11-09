def write_conllx(fp, raw_data, results):
    with open(fp, 'w') as f:
        all_words = raw_data['words']
        all_tags = raw_data['tags']
        all_heads = raw_data['heads']
        all_labels = raw_data['labels']
        pred_heads = results['edge_predictions']
        pred_labels = results['label_predictions']
        for words, tags, heads, labels, pred_heads, pred_labels in \
                zip(all_words, all_tags, all_heads, all_labels, pred_heads, pred_labels):
            for i, (word, tag, head, label, pred_head, pred_label) in \
                    enumerate(zip(words, tags, heads, labels, pred_heads, pred_labels)):
                f.write(f'{i + 1}\t{word}\t_\t_\t{tag}\t_\t{pred_head}\t{pred_label}\t_\t_\n')
            f.write('\n')
