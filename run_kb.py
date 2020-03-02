import getpass
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import os, glob
import argparse
import numpy as np

from utils_tokenizer import MyTokenizer
from utils_real_data import examples_loader
from utils_embedding_model import features_loader

# logging
logger = logging.getLogger(__name__)

# hugging face transformers default models, can use pretrained ones though too


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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


def load_and_cache_features(args, tokenizer):
    only_context_str = '_ONLYCONTEXT' if args.only_context else ''
    cutoff_str = '' if args.cutoff is None else '_cutoff{}'.format(args.cutoff)

    features_filename = '{}{}{}'.format('_'.join(args.domain_words), only_context_str, cutoff_str)

    if os.path.exists(features_filename):
        logger.info('Loading features from ({})'.format(features_filename))
        features = torch.load(features_filename)

    else:
        logger.info('Creating features')
        examples = examples_loader(args)
        features = features_loader(args, tokenizer, examples)

        logger.info('Saving features into {}'.format(features_filename))
        torch.save(features, features_filename)

    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_sentence_type = torch.tensor([f.sentence_type for f in features], dtype=torch.long)
    all_labels = torch.tensor(label_map([f.label for f in features], num_choices=4), dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_sentence_type, all_labels)

    return dataset


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
                pass

        args = Args()

    # Setup logging
    num_logging_files = len(glob.glob('logging/logging_*'))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename='logging/logging_{}'.format(num_logging_files))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise Exception("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        raise Exception('Output directory does not exist here ({})'.format(args.output_dir))
    if not os.path.exists(args.cache_dir):
        raise Exception('Cache directory does not exist here ({})'.format(args.cache_dir))
    if not os.path.exists(args.data_dir):
        raise Exception('Data directory does not exist here ({})'.format(args.data_dir))

    for arg, value in sorted(vars(args).items()):
        logging.info("Argument {}: {}".format(arg, value))

    # Set seed
    set_seed(args)

    # my tokenizer to put data into useable format
    # only need the sentences to be used in training
    # should only have to do this once, save tokenized data
    # should output a list of objects with tokenized sentences
    my_tokenizer = MyTokenizer.load_tokenizer(args)

    # TODO might have to build the tokenizer first from examples to get the vocabulary, and remove low frequency words

    dataset = load_and_cache_features(args, my_tokenizer)

    # if loading graph make sure it exists, otherwise initialize the graph with the tokenized data
    knowledge_base = MyKnowledgeBase.load_knowledge_base(args, dataset)

    # Load models or randomly initialize, everything after the randomization should be doable in a single function

    # do training here

    # do evaluation here

    # do results here






