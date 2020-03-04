import getpass
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import os, glob
import argparse
import numpy as np
import random

from utils_tokenizer import MyTokenizer
from utils_real_data import examples_loader
from utils_embedding_model import features_loader
from utils_kb import build_graph, load_graph, save_graph

# logging
logger = logging.getLogger(__name__)


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)

# return if there is a gpu available
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


# returns a list of length num_choices with each entry a length four list with a 1 in the correct response spot and 0s elsewhere
def label_map(labels, num_choices):

    def label_list(target, new_target, switch_index, num_choices):
        l = [target]*num_choices
        l[switch_index] = new_target
        return l

    answers = [label_list(0, 1, lab, num_choices) for lab in labels]
    return answers


# from hf, returns a list of lists from features of a selected field within the choices_features list of dicts
def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.input_features
        ]
        for feature in features
    ]


def load_and_cache_(args):
    my_tokenizer = MyTokenizer.load_tokenizer(args)

    only_context_str = '_ONLYCONTEXT' if args.only_context else ''
    cutoff_str = '' if args.cutoff is None else '_cutoff{}'.format(args.cutoff)

    features_filename = os.path.join(args.cache_dir, 'features_{}{}'.format(only_context_str, cutoff_str))
    # tokenizer should already be loaded but this is just making damn sure
    tokenizer_filename = os.path.join(args.cache_dir, 'tokenizerDict_{}.py'.format(args.tokenizer_name))
    vocabulary_file = os.path.join(args.cache_dir, 'vocabulary_{}.py'.format(args.tokenizer_name))

    if os.path.exists(features_filename) and os.path.exists(tokenizer_filename) and os.path.exists(vocabulary_file) and not args.overwrite_cache_dir:
        logger.info('Loading features from ({})'.format(features_filename))
        features = torch.load(features_filename)
    else:
        logger.info('Creating features')
        examples = examples_loader(args)

        assert my_tokenizer.build_and_save_tokenizer(args, examples) == -1

        features = features_loader(args, my_tokenizer, examples)

        logger.info('Saving features into {}'.format(features_filename))
        torch.save(features, features_filename)

    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_sentence_type = torch.tensor([f.sentence_type for f in features], dtype=torch.long).unsqueeze(1)
    all_labels = torch.tensor(label_map([f.label for f in features], num_choices=4), dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_sentence_type, all_labels)

    graph_filename = os.path.join(args.cache_dir, 'graph{}.py'.format(cutoff_str))
    if os.path.exists(graph_filename) and not args.overwrite_cache_dir:
        knowledge_base = load_graph(args)
    else:
        vocabulary = my_tokenizer.vocabulary()
        knowledge_base = build_graph(args, dataset, vocabulary)

        assert save_graph(args, knowledge_base) == -1

    return my_tokenizer, dataset, knowledge_base


def train(args, dataset, knowledge_base, model):

    # set up dataset in a sampler

    # initialize optimizers

    # start training

        # get batch

        # send through model

        # backwards pass

        # depending on args do evaluation

    pass


def evaluate(args):
    pass


def main():
    parser = argparse.ArgumentParser()


    if not getpass.getuser() == 'Mitch':
        args = parser.parse_args()
    else:
        class Args(object):
            def __init__(self):
                self.data_dir = '../ARC/ARC-with-context/'
                self.output_dir = 'output/'
                self.cache_dir = 'saved/'
                self.tokenizer_name = 'bert-base-uncased'
                self.tokenizer_model = 'bert'
                self.cutoff = 50
                self.overwrite_output_dir = True
                self.overwrite_cache_dir = False
                self.seed = 1234
                self.max_length = 128
                self.do_lower_case = True

                self.domain_words = ['moon', 'earth']
                self.only_context = True
                self.pmi_threshold = 0.4
                self.common_word_threshold = 3


        args = Args()

    # Setup logging
    num_logging_files = len(glob.glob('logging/logging_*'))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename='logging/logging_{}'.format(num_logging_files))

    if not os.path.exists(args.output_dir):
        raise Exception('Output directory does not exist here ({})'.format(args.output_dir))
    if not os.path.exists(args.cache_dir):
        raise Exception('Cache directory does not exist here ({})'.format(args.cache_dir))
    if not os.path.exists(args.data_dir):
        raise Exception('Data directory does not exist here ({})'.format(args.data_dir))

    # within output and saved folders create a folder with domain words to keep output and saved objects
    folder_name = '-'.join(args.domain_words)
    proposed_output_dir = os.path.join(args.output_dir, folder_name)
    if not os.path.exists(proposed_output_dir):
        os.makedirs(proposed_output_dir)
    else:
        if os.listdir(proposed_output_dir) and not args.overwrite_output_dir:
            raise Exception(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    proposed_output_dir))

    args.output_dir = proposed_output_dir

    proposed_cache_dir = os.path.join(args.cache_dir, folder_name)
    if not os.path.exists(proposed_cache_dir):
        os.makedirs(proposed_cache_dir)
    # else:
    #     if os.listdir(proposed_cache_dir) and not args.overwrite_cache_dir:
    #         raise Exception(
    #             "Cache directory ({}) already exists and is not empty. Use --overwrite_cache_dir to overcome.".format(
    #                 proposed_cache_dir))

    args.cache_dir = proposed_cache_dir

    for arg, value in sorted(vars(args).items()):
        logger.info("Argument {}: {}".format(arg, value))

    # Set seed
    set_seed(args)

    # load the examples and features, build the tokenizer if not already implemented, and the knowledge base
    my_tokenizer, dataset, knowledge_base = load_and_cache_(args)

    # Load models or randomly initialize, everything after the randomization should be doable in a single function
    model = MyModel()

    # do training here
    train(args, dataset, knowledge_base, model)

    # do evaluation here

    # do results here


if __name__ == '__main__':
    main()



