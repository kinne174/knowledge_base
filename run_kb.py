import getpass
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch.optim as optim
from tqdm import tqdm, trange
import os, glob
import argparse
import numpy as np
import random

from utils_tokenizer import MyTokenizer
from utils_real_data import examples_loader
from utils_embedding_model import features_loader
from utils_kb import build_graph, load_graph, save_graph
from utils_models import GraphBlock

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


def train(args, dataset, knowledge_base, model, optimizer):

    # set up dataset in a sampler
    # use pytorch data loaders to cycle through the data,
    train_sampler = RandomSampler(dataset, replacement=False)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size)

    num_training_correct = 0
    num_training_seen = 0

    train_iterator = trange(int(args.epochs), desc="Epoch")
    # start training
    logger.info('Starting to train!')
    logger.info('There are {} examples.'.format(len(dataset)))
    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration, batch size {}".format(args.batch_size))
        for iterate, batch in enumerate(epoch_iterator):
            logger.info('Epoch: {}, Batch: {}'.format(epoch, iterate))

            model.zero_grad()

            # get batch
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'input_mask': batch[1],
                      'sentence_type': batch[2],
                      'labels': batch[3],
                      }

            # send through model
            model.train()
            error, predictions = model(knowledge_base, training=True, **inputs)

            # backwards pass
            error.backward()
            optimizer.step()

            # print('model params')
            # for i in range(len(list(model.parameters()))):
            #     print(i)
            #     print(list(model.parameters())[i].grad)
            #     print(torch.max(list(model.parameters())[i].grad))

            logger.info('The error is {}'.format(error))

            num_training_seen += int(inputs['labels'].shape[0])
            num_training_correct += int(sum([inputs['labels'][i, p].item() for i, p in zip(range(inputs['labels'].shape[0]), predictions)]))
            logger.info('The training total correct is {} out of {} for a percentage of {}'.format(
                num_training_correct, num_training_seen, round(num_training_correct/num_training_seen, 2)))

            # TODO depending on args do evaluation


def evaluate(args):
    pass


def main():
    parser = argparse.ArgumentParser()

    if not getpass.getuser() == 'Mitch':

        # Required
        parser.add_argument('--domain_words', default=None, nargs='+', required=True,
                            help='Domain words to search for')

        # Optional
        parser.add_argument('--data_dir', default='../ARC/ARC-with-context/', type=str,
                            help='Data directory where question answers with context resides, train/test/dev .jsonl')
        parser.add_argument('--output_dir', default='output/', type=str,
                            help='Directory where output should be written')
        parser.add_argument('--cache_dir', default='saved/', type=str,
                            help='Directory where saved parameters should be written')
        parser.add_argument('--tokenizer_name', default='bert-base-uncased', type=str,
                            help='Name of the tokenizer to be used in tokenizing the strings')
        parser.add_argument('--tokenizer_model', default='bert', type=str,
                            help='Model of the tokenizer to be used in tokenizing the strings (bert, albert etc.)')
        parser.add_argument('--cutoff', default=None, type=int,
                            help='Number of examples to cutoff at if testing code')
        parser.add_argument('--overwrite_output_dir', action='store_true',
                            help='bool used to make sure user wants to overwrite any output sharing name of domain words in use')
        parser.add_argument('--overwrite_cache_dir', action='store_true',
                            help='bool used to overwrite any saved parameters sharing name of domain words in use')
        parser.add_argument('--seed', default=1234, type=int,
                            help='Seed for consistent randomization')
        parser.add_argument('--max_length', default=128, type=int,
                            help='maximum length of tokens in a sentence, anything longer is cutoff')
        parser.add_argument('--do_lower_case', action='store_true',
                            help='Convert all tokens to lower case')
        parser.add_argument('--no_gpu', action='store_true',
                            help='Use cpu even if gpu is available')
        parser.add_argument('--batch_size', default=25, type=int,
                            help='Batch size of each iteration')
        parser.add_argument('--epochs', default=10, type=int,
                            help='Number of epochs')
        parser.add_argument('--only_context', action='store_true',
                            help='Only look at context to pick out sentences')
        parser.add_argument('--pmi_threshold', default=0.4, type=float,
                            help='Only accept connections in GN above this threshold')
        parser.add_argument('--common_word_threshold', default=2, type=int,
                            help='If number of words observed is at or below this threshold assign it [UNK] token')
        parser.add_argument('--lstm_hidden_dim', default=256, type=int,
                            help='Dimension size of hiden layer of LSTM')
        parser.add_argument('--mlp_hidden_dim', default=256, type=int,
                            help='Dimension size of fully connected layer of MLP')
        parser.add_argument('--essential_terms_hidden_dim', default=512, type=int,
                            help='Dimension size of hidden layer within train_noise model')
        parser.add_argument('--edge_parameter', default=0.1, type=float,
                            help='Hyperparameter to calculate how likely an edge is to exist in GN')
        parser.add_argument('--word_embedding_dim', default=300, type=int,
                            help='This probably will not change, embedding dimension of word vectors')
        parser.add_argument('--attention_window_size', default=3, type=int,
                            help='Number of words to replace in sentences with negative sampling')

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
                self.overwrite_cache_dir = True
                self.seed = 1234
                self.max_length = 128
                self.do_lower_case = True
                self.no_gpu = True
                self.batch_size = 2
                self.epochs = 3

                self.domain_words = ['moon', 'earth']
                self.only_context = True
                self.pmi_threshold = 0.4
                self.common_word_threshold = 1
                self.lstm_hidden_dim = 25
                self.edge_parameter = 0.1
                self.word_embedding_dim = 300
                self.mlp_hidden_dim = 100
                self.essential_terms_hidden_dim = 100
                self.attention_window_size = 3


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

    assert 0 <= np.sqrt(args.edge_parameter/args.pmi_threshold) <= 1, "Edge parameter and pmi threshold won't work together"
    assert 0 <= np.sqrt(args.edge_parameter) <= 1, "Edge parameter won't work, needs to be less"

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

    # get whether running on cpu or gpu
    device = torch.device('cpu') if args.no_gpu else get_device()
    args.device = device
    logger.info('Using device {}'.format(args.device))

    for arg, value in sorted(vars(args).items()):
        logger.info("Argument {}: {}".format(arg, value))

    # Set seed
    set_seed(args)

    # load the examples and features, build the tokenizer if not already implemented, and the knowledge base
    my_tokenizer, dataset, knowledge_base = load_and_cache_(args)

    # Load models or randomly initialize, everything after the randomization should be doable in a single function
    model = GraphBlock(args)

    # throw model to device
    model.to(args.device)

    # intiailize optimizer
    optimizer = optim.Adam(params=model.parameters())

    # do training here
    train(args, dataset, knowledge_base, model, optimizer)

    # do evaluation here
    # TODO better logic for evaluation

    # do results here


if __name__ == '__main__':
    main()



