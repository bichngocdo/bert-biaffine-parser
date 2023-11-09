from core.data.converters import PaddedTensorConverter


class DataConverter(PaddedTensorConverter):
    def __init__(self):
        # x_word, x_pt_word, x_tag, x_pt_tag, x_char,
        # x_bert_ids, x_bert_mask, x_bert_types, x_bert_indices
        # y_edge, y_label
        size = 11
        padding_values = [0, 0, 0, 0, 0,
                          0, 0, 0, -1,
                          0, 0]
        types = 'int32'
        super(DataConverter, self).__init__(size, padding_values, types)
