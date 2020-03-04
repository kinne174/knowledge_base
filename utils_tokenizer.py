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
    def __init__(self, args, old_node_indices_dict=None, old_vocabulary_dict=None):
        tokenizer_class = transformer_tokenizer_classes[args.tokenizer_model]

        self._bert = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)
        self.bert_to_node_indices = {} if old_node_indices_dict is None else old_node_indices_dict

        self.myind_to_word = {} if old_vocabulary_dict is None else old_vocabulary_dict

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
        node_indices_file = os.path.join(args.cache_dir, 'tokenizerDict_{}.py'.format(args.tokenizer_name))
        logger.info('Saving tokenizer with {} ids to {}'.format(len(self.bert_to_node_indices), node_indices_file))

        with open(node_indices_file, 'wb') as tokenizer_writer:
            pickle.dump(obj=self.bert_to_node_indices, file=tokenizer_writer)

        vocabulary_file = os.path.join(args.cache_dir, 'vocabulary_{}.py'.format(args.tokenizer_name))
        logger.info('Saving vocabulary to {}'.format(vocabulary_file))

        with open(vocabulary_file, 'wb') as vocabulary_writer:
            pickle.dump(obj=self.myind_to_word, file=vocabulary_writer)

        return -1

    @classmethod
    def load_tokenizer(cls, args):
        node_indices_file = os.path.join(args.cache_dir, 'tokenizerDict_{}.py'.format(args.tokenizer_name))
        vocabulary_file = os.path.join(args.cache_dir, 'vocabulary_{}.py'.format(args.tokenizer_name))

        if os.path.exists(node_indices_file) and os.path.exists(vocabulary_file) and not args.overwrite_cache_dir:
            logger.info('Loding pretrained tokenizer from {} and {}'.format(node_indices_file, vocabulary_file))

            with open(node_indices_file, 'rb') as tokenizer_reader:
                saved_node_indices_dict = pickle.load(tokenizer_reader)

            with open(vocabulary_file, 'rb') as vocabulary_reader:
                saved_vocabulary_dict = pickle.load(vocabulary_reader)

            return cls(args, old_node_indices_dict=saved_node_indices_dict, old_vocabulary_dict=saved_vocabulary_dict)

        return cls(args)

    def build_and_save_tokenizer(self, args, examples):
        counter = None

        for example_ind, example in enumerate(examples):
            assert isinstance(example, ArcExample)

            for sentence_feature in example.sentence_features:
                sentence = sentence_feature['sentence']

                try:
                    inputs = self._bert.encode_plus(
                        sentence,
                        add_special_tokens=False,
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

        vocabulary_size = len(self.bert_to_node_indices)
        num_thrown_to_unk = 0
        for id, count in counter.items():
            if count <= args.common_word_threshold:
                self.bert_to_node_indices[id] = vocabulary_size
                num_thrown_to_unk += 1

        logger.info('The number of words replaced by [UNK] when building vocabulary is {}'.format(num_thrown_to_unk))

        node_indices_list = list(self.bert_to_node_indices.items())
        # TODO there's a faster way to do this by summing, rather than sorting, then can just compare for 'for loop'
        node_indices_list.sort(key=lambda t: t[1], reverse=False)
        first_unk_index = [t[1] for t in node_indices_list].index(vocabulary_size)

        for ind, (bert_ind, _) in enumerate(node_indices_list):
            if ind < first_unk_index:
                self.myind_to_word[ind+2] = self._bert.convert_ids_to_tokens(bert_ind, skip_special_tokens=True)
                self.bert_to_node_indices[bert_ind] = ind+2
            else:
                self.bert_to_node_indices[bert_ind] = 1

        self.myind_to_word[1] = '[UNK]'
        self.myind_to_word[0] = '[PAD]'

        assert self.save_tokenizer(args) == -1

        return -1

    def vocabulary(self):
        vocab_tuples = list(self.myind_to_word.items())

        vocab_tuples.sort(key=lambda t: t[0], reverse=False)

        return [t[1] for t in vocab_tuples]
