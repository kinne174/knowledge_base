import pickle
import os
import logging
from utils_real_data import ArcExample
from collections import Counter
import nltk
from nltk import word_tokenize

# logging
logger = logging.getLogger(__name__)


class MyTokenizer(object):
    def __init__(self, args, old_node_indices_dict=None, old_vocabulary_dict=None):

        # mirror dicts to help with transition from token to word and vice versa
        self.word_to_my_ind = {} if old_node_indices_dict is None else old_node_indices_dict
        self.myind_to_word = {} if old_vocabulary_dict is None else old_vocabulary_dict

    def encode(self, sentence, do_lower_case):
        # expects one sentence at a time

        # seperate words/ punctuation within the sentence
        words = word_tokenize(sentence)

        # assign each word the proper token or the [UNK] if not in the token dict
        node_ids = []
        for word in words:
            word = word.lower() if do_lower_case else word
            try:
                node_ids.append(self.word_to_my_ind[word])
            except KeyError:
                node_ids.append(1)
                self.word_to_my_ind[word] = 1
                logger.info('Word {} not found in initial tokenization! Mapping to [UNK].'.format(word))

        return node_ids

    def save_tokenizer(self, args):
        # save the two necessary dicts
        node_indices_file = os.path.join(args.cache_dir, 'tokenizerDict.py')
        logger.info('Saving tokenizer with {} ids to {}'.format(len(self.word_to_my_ind), node_indices_file))

        with open(node_indices_file, 'wb') as tokenizer_writer:
            pickle.dump(obj=self.word_to_my_ind, file=tokenizer_writer)

        vocabulary_file = os.path.join(args.cache_dir, 'vocabulary.py')
        logger.info('Saving vocabulary to {}'.format(vocabulary_file))

        with open(vocabulary_file, 'wb') as vocabulary_writer:
            pickle.dump(obj=self.myind_to_word, file=vocabulary_writer)

        return -1

    @classmethod
    def load_tokenizer(cls, args, evaluating):
        node_indices_file = os.path.join(args.cache_dir, 'tokenizerDict.py')
        vocabulary_file = os.path.join(args.cache_dir, 'vocabulary.py')

        # if the files exist and not overwriting cache load pre trained tokenizer, otherwise overwrite
        if os.path.exists(node_indices_file) and os.path.exists(vocabulary_file) and (not args.overwrite_cache_dir or evaluating):
            logger.info('Loding pretrained tokenizer from {} and {}'.format(node_indices_file, vocabulary_file))

            with open(node_indices_file, 'rb') as tokenizer_reader:
                saved_node_indices_dict = pickle.load(tokenizer_reader)

            with open(vocabulary_file, 'rb') as vocabulary_reader:
                saved_vocabulary_dict = pickle.load(vocabulary_reader)

            return cls(args, old_node_indices_dict=saved_node_indices_dict, old_vocabulary_dict=saved_vocabulary_dict)

        return cls(args)

    def build_and_save_tokenizer(self, args, examples):
        # create tokenizer from the examples, each word gets a unique id
        counter = Counter()
        temp_word_to_my_ind = {}

        for example_ind, example in enumerate(examples):
            assert isinstance(example, ArcExample)

            for sentence in example.sentences:

                # separate words/ punctuation
                words = word_tokenize(sentence)

                # for each word if an id has not been created, create one based on the number of words currently in dict
                for word in words:
                    word = word.lower() if args.do_lower_case else word
                    if word not in temp_word_to_my_ind:
                        current_word_myid = len(temp_word_to_my_ind)
                        temp_word_to_my_ind[word] = current_word_myid

                # to help with words being mapped to [UNK] create a counter so can threshold words with small counts
                counter.update(words)

        # reassign low threshold words to a random number not yet seen (len vocabulary in this instance)
        vocabulary_size = len(temp_word_to_my_ind)
        num_thrown_to_unk = 0
        for word, count in counter.items():
            if count <= args.common_word_threshold:
                temp_word_to_my_ind[word] = vocabulary_size
                num_thrown_to_unk += 1

        logger.info('The number of words replaced by [UNK] when building vocabulary is {}'.format(num_thrown_to_unk))

        # order by count and find where the cutoff for minimal used words
        node_indices_list = list(temp_word_to_my_ind.items())
        # TODO there's a faster way to do this by summing, rather than sorting, then can just compare for 'for loop'
        node_indices_list.sort(key=lambda t: t[1], reverse=False)
        first_unk_index = [t[1] for t in node_indices_list].index(vocabulary_size)

        # re assign ids based on count and assign all minimum use words to 1<->[UNK]
        for ind, (word, _) in enumerate(node_indices_list):
            if ind < first_unk_index:
                self.myind_to_word[ind+2] = word
                self.word_to_my_ind[word] = ind+2
            else:
                self.word_to_my_ind[word] = 1

        # give 0 to [PAD] and 1 to [UNK]
        self.myind_to_word[1] = '[UNK]'
        self.myind_to_word[0] = '[PAD]'

        # save tokenizer dicts
        assert self.save_tokenizer(args) == -1

        return -1

    def vocabulary(self):
        # return list of all words in vocabulary
        vocab_tuples = list(self.myind_to_word.items())

        vocab_tuples.sort(key=lambda t: t[0], reverse=False)

        return [t[1] for t in vocab_tuples]
