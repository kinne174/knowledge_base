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
from utils_ablation import ablation

# logging
logger = logging.getLogger(__name__)

# set seed for randomization purposes
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


def load_and_cache_evaluation(args, subset, all_models):
    # when evaluating find questions in a specific subset that contains the domain words

    # make sure these files exist before progressing since, training is not guaranteed
    tokenizer_filename = os.path.join(args.cache_dir, 'tokenizerDict.py')
    assert os.path.exists(tokenizer_filename), 'Cannot find tokenizer filename, this must exist to evaluate'
    vocabulary_filename = os.path.join(args.cache_dir, 'vocabulary.py')
    assert os.path.exists(vocabulary_filename), 'Cannot find vocabulary filename, this must exist to evaluate'

    logging.info('For evaluating {} subset loading tokenizer from {} and {}'.format(subset,
                                                                                    tokenizer_filename,
                                                                                    vocabulary_filename))
    # load the tokenizer, files are loaded from within class
    my_tokenizer = MyTokenizer.load_tokenizer(args, evaluating=True)

    # if features already exist then fetch otherwise create/ save
    feautres_filename = os.path.join(args.cache_dir, 'features_{}'.format(subset))
    if os.path.exists(feautres_filename):
        logger.info('Loading {} features'.format(subset))
        features = torch.load(feautres_filename)
    else:
        logger.info('Could not find features file, creating {} features'.format(subset))
        examples = examples_loader(args, evaluate_subset=subset)
        features = features_loader(args, my_tokenizer, examples)

        logger.info('Saving {} features'.format(subset))
        torch.save(features, feautres_filename)

    # create tensors from data to load into a dataset
    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_sentence_type = torch.tensor([f.sentence_type for f in features], dtype=torch.long).unsqueeze(1)
    all_labels = torch.tensor(label_map([f.label for f in features], num_choices=4), dtype=torch.float)

    # make sure all sentence types are that of a question
    assert all(all_sentence_type == torch.ones_like(all_sentence_type)), 'Not all sentences are a question in {}'.format(subset)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_labels)

    # make sure the graph file exists and load it
    cutoff_str = '' if args.cutoff is None else '_cutoff{}'.format(args.cutoff)
    graph_filename = os.path.join(args.cache_dir, 'graph{}.py'.format(cutoff_str))
    assert os.path.exists(graph_filename)

    logger.info('Evaluating {} subset, Loading graph from {}'.format(subset,
                                                                     graph_filename))
    knowledge_base = load_graph(args)

    # load model
    model_folders = glob.glob(os.path.join(args.output_dir, 'model_checkpoint_*'))
    assert len(model_folders) > 0, 'No model parameters found'
    index_ = len(model_folders[0]) - model_folders[0][::-1].index('_')

    checkpoints = [int(mf[index_:]) for mf in model_folders]

    # if all the models are not being evaluted only grab the latest one
    if not all_models:
        max_checkpoint = max(checkpoints)
        model_folders_checkpoints = [(os.path.join(args.output_dir, 'model_checkpoint_{}'.format(max_checkpoint)), max_checkpoint)]
    else:
        model_folders_checkpoints = list(map(list, zip(model_folders, checkpoints)))

    # load each model and combine with its checkpoint
    models_checkpoints = []
    for (mf, cp) in model_folders_checkpoints:

        logger.info('Loading model using checkpoint {}'.format(cp))
        model_filename = os.path.join(mf, 'model_parameters.py')
        good_counter_filename = os.path.join(mf, 'good_counter.py')
        all_counter_filename = os.path.join(mf, 'all_counter.py')

        model = GraphBlock.from_pretrained(args, knowledge_base, good_counter_filename, all_counter_filename)
        model.load_state_dict(torch.load(model_filename))
        model.eval()

        models_checkpoints.append((model, cp))

    return dataset, models_checkpoints


def load_and_cache_training(args):
    # create/ save or load for training, main difference from evaluation is
    my_tokenizer = MyTokenizer.load_tokenizer(args, evaluating=False)

    # label features if not using the corpus and/or if a cutoff is being used for code-testing purposes
    only_context_str = '_ONLYCONTEXT' if args.only_context else ''
    cutoff_str = '' if args.cutoff is None else '_cutoff{}'.format(args.cutoff)

    features_filename = os.path.join(args.cache_dir, 'features{}{}{}'.format(only_context_str, cutoff_str, 'train'))

    # tokenizer should already be loaded but this is just making damn sure
    tokenizer_filename = os.path.join(args.cache_dir, 'tokenizerDict.py')
    vocabulary_filename = os.path.join(args.cache_dir, 'vocabulary.py')

    # fetch features or create/save them
    if os.path.exists(features_filename) and os.path.exists(tokenizer_filename) and os.path.exists(vocabulary_filename) and not args.overwrite_cache_dir:
        logger.info('Loading features from ({})'.format(features_filename))
        features = torch.load(features_filename)
    else:
        logger.info('Creating features')
        examples = examples_loader(args)

        assert my_tokenizer.build_and_save_tokenizer(args, examples) == -1

        features = features_loader(args, my_tokenizer, examples)

        logger.info('Saving features into {}'.format(features_filename))
        torch.save(features, features_filename)

    # create tensors from data to load into a dataset
    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_sentence_type = torch.tensor([f.sentence_type for f in features], dtype=torch.long).unsqueeze(1)
    all_labels = torch.tensor(label_map([f.label for f in features], num_choices=4), dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_sentence_type, all_labels)

    # fetch or create/save graph
    graph_filename = os.path.join(args.cache_dir, 'graph{}.py'.format(cutoff_str))
    if os.path.exists(graph_filename) and not args.overwrite_cache_dir:
        knowledge_base = load_graph(args)
    else:
        vocabulary = my_tokenizer.vocabulary()
        knowledge_base = build_graph(args, dataset, vocabulary)

        assert save_graph(args, knowledge_base) == -1

    return dataset, knowledge_base


def train(args, dataset, model, optimizer):

    # set up dataset in a sampler
    # use pytorch data loaders to cycle through the data,
    train_sampler = RandomSampler(dataset, replacement=False)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size)

    global_step = 0

    train_iterator = trange(int(args.epochs), desc="Epoch")
    # start training
    logger.info('Starting to train!')
    logger.info('There are {} examples.'.format(len(dataset)))
    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration, batch size {}".format(args.batch_size))

        # index 0 is context, index 1 is questions
        num_training_correct = np.array([0]*2)
        num_training_seen = np.array([0]*2)
        total_error = 0

        for iterate, batch in enumerate(epoch_iterator):
            logger.info('Epoch: {}, Batch: {}'.format(epoch, iterate))

            # zero out previous runs
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
            error, softmaxed_scores = model(training=True, **inputs)

            # backwards pass
            error.backward()
            optimizer.step()

            # print('model params')
            # for i in range(len(list(model.parameters()))):
            #     print(i)
            #     print(list(model.parameters())[i].grad)
            #     print(torch.max(list(model.parameters())[i].grad))

            total_error += error.item()

            logger.info('The error for this epoch is {}'.format(round(total_error/(iterate + 1), 4)))

            # helper function to update the correct responses in questions and contexts
            def correct_update(sentence_type, correct):
                temp = [0, 0]
                temp[sentence_type] = int(correct)
                return temp

            # calculate prediction
            predictions = torch.argmax(softmaxed_scores, dim=1)

            # num training seen/ correct divided up by context and questions
            num_training_seen = np.add(num_training_seen, np.array([inputs['sentence_type'].shape[0] - sum(inputs['sentence_type']), sum(inputs['sentence_type'])], dtype=int), dtype=int)
            correct_list = [inputs['labels'][i, p].item() for i, p in zip(range(inputs['labels'].shape[0]), predictions)]
            num_training_correct = np.add(num_training_correct, np.sum([correct_update(st, c) for st, c in zip(inputs['sentence_type'], correct_list)], axis=0), dtype=int)

            logger.info('The training total for this epoch correct is {} out of {} for a percentage of {}'.format(
                sum(num_training_correct), sum(num_training_seen), round(sum(num_training_correct)/float(sum(num_training_seen)), 3)))

            for nind, (nc, ns) in enumerate(zip(num_training_correct, num_training_seen)):
                if ns == 0:
                    logger.info('No examples of type "{}" seen during this epoch'.format(['context', 'mc question'][nind]))
                    continue
                logger.info('The training total correct for type {} is {} out of {} for a percentage of {}'.format(
                    ['context', 'mc question'][nind], nc, ns, round(nc/float(ns), 3)
                ))

            # save model and evaluate depending on args
            if global_step is not 0 and global_step % args.global_save_step == 0:
                assert save_model_params(args=args, checkpoint=global_step, model=model) == -1

                if args.evaluate_during_training:
                    assert evaluate(args, subset='dev', all_models=False) == -1

            global_step += 1

    # save model at the end
    assert save_model_params(args=args, checkpoint=global_step, model=model) == -1


def evaluate(args, subset, all_models=False):
    assert subset in ['dev', 'test'], 'subset must be one of "test" or "dev"'

    # get questions from appropriate subset, loads latest model
    dataset, models_checkpoints = load_and_cache_evaluation(args, subset, all_models)

    for (model, checkpoint) in models_checkpoints:

        model = model.to(args.device)

        logging.info('Begining to evaluate {} subset using model from checkpoint {}'.format(subset, checkpoint))

        # should be whole thing
        batch = dataset.tensors

        # get batch
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'input_mask': batch[1],
                  'labels': None,
                  }
        labels = batch[2]

        _, softmaxed_scores = model(training=False, **inputs)

        # calculate prediction
        predictions = torch.argmax(softmaxed_scores, dim=1)

        num_seen = int(labels.shape[0])
        num_correct = int(
            sum([labels[i, p].item() for i, p in zip(range(labels.shape[0]), predictions)]))
        logger.info('In {}: The number total correct is {} out of {} for a percentage of {}'.format(subset,
            num_correct, num_seen, round(num_correct / num_seen, 2)))

        # depending on args do an ablation study
        if args.do_ablation:
            assert ablation(args, subset, model, checkpoint, dataset) == -1

    return -1

def save_model_params(args, checkpoint, model):
    # each checkpoint should have a different folder
    model_save_folder = os.path.join(args.output_dir, 'model_checkpoint_{}'.format(checkpoint))

    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    model_save_file = os.path.join(model_save_folder, 'model_parameters.py')
    good_counter_save_file = os.path.join(model_save_folder, 'good_counter.py')
    all_counter_save_file = os.path.join(model_save_folder, 'all_counter.py')

    # save the parameters and the counters for posterior edge calculations
    with open(model_save_file, 'wb') as mf:
        torch.save(model.state_dict(), mf)

    with open(good_counter_save_file, 'wb') as gf:
        torch.save(model.good_edge_connections, gf)

    with open(all_counter_save_file, 'wb') as af:
        torch.save(model.all_edge_connections, af)

    return -1


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
        parser.add_argument('--cutoff', default=None, type=int,
                            help='Number of examples to cutoff at if testing code')
        parser.add_argument('--overwrite_output_dir', action='store_true',
                            help='bool used to make sure user wants to overwrite any output sharing name of domain words in use')
        parser.add_argument('--clear_output_dir', action='store_true',
                            help='Clear all files in output directory')
        parser.add_argument('--overwrite_cache_dir', action='store_true',
                            help='bool used to overwrite any saved parameters sharing name of domain words in use')
        parser.add_argument('--seed', default=1234, type=int,
                            help='Seed for consistent randomization')
        parser.add_argument('--max_length', default=75, type=int,
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
        parser.add_argument('--common_word_threshold', default=4, type=int,
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
        parser.add_argument('--global_save_step', default=100, type=int,
                            help='Save model parameters when modulus this step is zero')
        parser.add_argument('--evaluate_during_training', action='store_true',
                            help='While saving perform evaluation of model on dev set')
        parser.add_argument('--train', action='store_true',
                            help='Perform training of model')
        parser.add_argument('--evaluate_dev', action='store_true',
                            help='Evaluate latest model on dev set')
        parser.add_argument('--evaluate_test', action='store_true',
                            help='Evaluate latest model on test set')
        parser.add_argument('--evaluate_all_models', action='store_true',
                            help='Evaluate all checkpoints in relevant output directory')
        parser.add_argument('--do_ablation', action='store_true',
                            help='Within evaluate do an ablation study.')

        args = parser.parse_args()

    else:
        class Args(object):
            def __init__(self):
                self.data_dir = '../ARC/ARC-with-context/'
                self.output_dir = 'output/'
                self.cache_dir = 'saved/'
                self.cutoff = 500
                self.overwrite_output_dir = True
                self.overwrite_cache_dir = False
                self.seed = 1234
                self.max_length = 75
                self.do_lower_case = True
                self.no_gpu = False
                self.batch_size = 25
                self.epochs = 3
                self.global_save_step = 2
                self.evaluate_during_training = False
                self.train = True
                self.evaluate_dev = True
                self.evaluate_test = False
                self.clear_output_dir = False
                self.do_ablation = True
                self.evaluate_all_models = False

                self.domain_words = ['moon', 'earth']
                self.only_context = True
                self.pmi_threshold = 0.4
                self.common_word_threshold = 1
                self.lstm_hidden_dim = 25
                self.edge_parameter = 0.1
                self.word_embedding_dim = 300
                self.mlp_hidden_dim = 100
                self.essential_terms_hidden_dim = 512
                self.attention_window_size = 3

        args = Args()

    # Setup logging
    num_logging_files = len(glob.glob('logging/logging_{}_*'.format('-'.join(args.domain_words))))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename='logging/logging_{}_{}'.format('-'.join(args.domain_words), num_logging_files))

    # initial checks
    if not os.path.exists(args.output_dir):
        raise Exception('Output directory does not exist here ({})'.format(args.output_dir))
    if not os.path.exists(args.cache_dir):
        raise Exception('Cache directory does not exist here ({})'.format(args.cache_dir))
    if not os.path.exists(args.data_dir):
        raise Exception('Data directory does not exist here ({})'.format(args.data_dir))
    if not args.train and (args.evaluate_test or args.evaluate_dev) and args.clear_output_dir:
        raise Exception('You are clearing the output directory without training and asking to evaluate on the test and/or dev set!\n'
                        'Fix one of --train, --evaluate_test, --evaluate_dev, or --clear_output_dir')

    assert 0 <= np.sqrt(args.edge_parameter/args.pmi_threshold) <= 1, "Edge parameter and pmi threshold won't work together"
    assert 0 <= np.sqrt(args.edge_parameter) <= 1, "Edge parameter won't work, needs to be less"

    # within output and saved folders create a folder with domain words to keep output and saved objects
    folder_name = '-'.join(args.domain_words)
    proposed_output_dir = os.path.join(args.output_dir, folder_name)
    if not os.path.exists(proposed_output_dir):
        os.makedirs(proposed_output_dir)
    else:
        if os.listdir(proposed_output_dir):
            if not args.overwrite_output_dir:
                raise Exception(
                    "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                        proposed_output_dir))
            elif args.clear_output_dir:
                for folder in os.listdir(proposed_output_dir):
                    filenames = os.listdir(os.path.join(proposed_output_dir, folder))
                    for filename in filenames:
                        file_path = os.path.join(proposed_output_dir, folder, filename)
                        try:
                            os.unlink(file_path)
                        except Exception as e:
                            logger.info('Failed to delete {}. Reason: {}'.format(file_path, e))
                    os.rmdir(os.path.join(proposed_output_dir, folder))
    if not args.overwrite_output_dir and args.clear_output_dir:
        logger.info('If you want to clear the output directory make sure to set --overwrite_output_dir too')


    args.output_dir = proposed_output_dir

    # reassign cache dir based on domain words
    proposed_cache_dir = os.path.join(args.cache_dir, folder_name)
    if not os.path.exists(proposed_cache_dir):
        os.makedirs(proposed_cache_dir)

    args.cache_dir = proposed_cache_dir

    # get whether running on cpu or gpu
    device = torch.device('cpu') if args.no_gpu else get_device()
    args.device = device
    logger.info('Using device {}'.format(args.device))

    # output args to logger
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument {}: {}".format(arg, value))

    # Set seed
    set_seed(args)

    if args.train:
        # load the examples and features, build the tokenizer if not already implemented, and the knowledge base
        dataset, knowledge_base = load_and_cache_training(args)

        # Load models or randomly initialize, everything after the randomization should be doable in a single function
        model = GraphBlock(args, knowledge_base)

        # throw model to device
        model.to(args.device)

        # intiailize optimizer
        optimizer = optim.Adam(params=model.parameters())

        # do training here
        train(args, dataset, model, optimizer)

    # do evaluation here for dev and test
    if args.evaluate_dev:
        evaluate(args, subset='dev', all_models=args.evaluate_all_models)

    if args.evaluate_test:
        evaluate(args, subset='test', all_models=args.evaluate_all_models)


if __name__ == '__main__':
    main()


# TODO explore adding more parameters to models
# TODO try resampling examples between epochs
# TODO graphs of error for training (mainly context but also overall)
# TODO summary of errors in ablation study (specifically force-matter-energy)
# TODO try taking parts out and compare performance, such as the resampling/ edge sampling/ edge weighting/ anything to do with GNN
# TODO find total number of parameters in network and individual models
