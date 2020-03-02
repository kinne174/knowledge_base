from transformers import BertTokenizer, AlbertTokenizer
import pickle
import os
import logging

# logging
logger = logging.getLogger(__name__)

transformer_tokenizer_classes = {
    'bert': BertTokenizer,
    'albert': AlbertTokenizer,
}


class MyTokenizer(object):
    def __init__(self, args, old_node_indices_dict=None):
        tokenizer_class = transformer_tokenizer_classes[args.tokenizer_model]

        self._bert = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)
        self.bert_to_node_indices = {} if old_node_indices_dict is None else old_node_indices_dict

    def encode(self, sentence, add_special_tokens, max_length):

        inputs = self._bert.encode_plus(sentence,
                            add_special_tokens=add_special_tokens,
                            max_length=max_length)

        _bert_input_ids = inputs['input_ids']

        node_ids = []
        for id in _bert_input_ids:
            if id not in self.bert_to_node_indices:
                current_word_id = len(self.bert_to_node_indices)
                self.bert_to_node_indices[id] = current_word_id

            node_ids.append(self.bert_to_node_indices[id])

        inputs['input_ids'] = node_ids

        return inputs

    def save_tokenizer(self, args):
        save_file = os.path.join(args.cache_dir, '{}_{}_{}.py'.format(args.tokenizer_model, args.tokenizer_name, '-'.join(args.domain_words)))
        logger.info('Saving tokenizer with {} ids to {}'.format(len(self.bert_to_node_indices), save_file))

        with open(save_file, 'w'):
            pickle.dump(obj=self.bert_to_node_indices, file=self.bert_to_node_indices)

    @classmethod
    def load_tokenizer(cls, args):
        load_file = os.path.join(args.cache_dir, '{}_{}_{}.py'.format(args.tokenizer_model, args.tokenizer_name, '-'.join(args.domain_words)))
        if os.path.exists(load_file):
            logging.info('Loding pretrained tokenizer from {}'.format(load_file))

            with open(load_file, 'r'):
                saved_node_indices_dict = pickle.load(load_file)

            return cls(args, old_node_indices_dict=saved_node_indices_dict)

        return cls(args)
