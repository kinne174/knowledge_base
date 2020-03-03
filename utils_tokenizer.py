from transformers import BertTokenizer, AlbertTokenizer
import pickle
import os
import logging
from utils_real_data import ArcExample
from collections import Counter

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

        self.myind_to_word = {}

    def encode(self, sentence, add_special_tokens, max_length):

        inputs = self._bert.encode_plus(sentence,
                            add_special_tokens=add_special_tokens,
                            max_length=max_length)

        _bert_input_ids = inputs['input_ids']

        node_ids = []
        for id in _bert_input_ids:
            try:
                node_ids.append(self.bert_to_node_indices[id])
            except KeyError:
                raise Exception('Id {} not found in tokenizer which is word {}!'.format(id, self._bert.convert_ids_to_tokens([id])))

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
            logger.info('Loding pretrained tokenizer from {}'.format(load_file))

            with open(load_file, 'r'):
                saved_node_indices_dict = pickle.load(load_file)

            return cls(args, old_node_indices_dict=saved_node_indices_dict)

        return cls(args)

    def build_tokenizer(self, args, examples):
        counter = None

        for example_ind, example in enumerate(examples):
            assert isinstance(example, ArcExample)

            for sentence_feature in example.sentence_features:
                sentence = sentence_feature['sentence']

                try:
                    inputs = self._bert.encode(
                        sentence,
                        add_special_tokens=True,
                        max_length=args.max_length
                    )
                except AssertionError as err_msg:
                    logger.info('Assertion error at example id {}: {}.\n -The sentence is {}'.format(example_ind, err_msg, sentence))
                    continue

                _bert_input_ids = inputs['input_ids']

                for id in _bert_input_ids:
                    if id not in self.bert_to_node_indices:
                        current_word_id = len(self.bert_to_node_indices)
                        self.bert_to_node_indices[id] = current_word_id

                if counter is None:
                    counter = Counter(_bert_input_ids)
                else:
                    counter.update(_bert_input_ids)

        # want to replace less common words with a single [UNK] id in node_indices dict
        # want to only have common words in the new dict

        vocabulary_size = len(self.bert_to_node_indices)
        for id, count in counter.items():
            if count <= args.common_word_threshold:
                self.bert_to_node_indices[id] = vocabulary_size

        node_indices_list = list(self.bert_to_node_indices.items())
        node_indices_list.sort(key=lambda t: t[1], reverse=False)
        first_unk_index = [t[1] for t in node_indices_list].index(vocabulary_size)

        for ind, (bert_ind, _) in enumerate(node_indices_list):
            if ind < first_unk_index:
                self.myind_to_word[ind] = self._bert.convert_ids_to_tokens([bert_ind])
            self.bert_to_node_indices[bert_ind] = min(ind, first_unk_index)

        self.myind_to_word[len(self.myind_to_word)] = '[UNK]'

        return -1

    def vocabulary(self):
        vocab_tuples = list(self.myind_to_word.items())

        vocab_tuples.sort(key=lambda t: t[0], reverse=False)

        return [t[1] for t in vocab_tuples]
