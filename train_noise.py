import os, glob
import spacy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler
import logging
import csv
from tqdm import trange, tqdm
import torch.optim as optim
import nltk
from nltk.corpus import stopwords

#logging
logger = logging.getLogger(__name__)

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_device()

stop_words =  set(stopwords.words('english'))

def get_words(cutoff):
    data_filename = '../ARC/essential_data/turkerSalientTermsWithOmnibus-v3.tsv'
    words_list = []
    labels_list = []

    with open(data_filename, 'r') as tsv_file:
        tsv_reader = csv.reader(tsv_file)

        for line in tsv_reader:
            line = ''.join(line)
            line = line.split('\t')

            question_text = line[0]
            num_annontators = int(line[1])
            annotations = line[2]

            answer_letters = ['(A)', '(B)', '(C)', '(D)']

            if not all([answer_letter in question_text for answer_letter in answer_letters]):
                assert any([answer_letter in question_text for answer_letter in answer_letters]), 'One of {} is not in the question text'.format(' '.join(answer_letters))
                answer_letters = [a_l for a_l in question_text if a_l in answer_letters]

            answer_inds = [question_text.index(answer_letter) for answer_letter in answer_letters] + [len(question_text)]

            answer_ind_tuples = [(answer_inds[i], answer_inds[j]) for i, j in zip(range(len(answer_inds)-1), range(1, len(answer_inds)))]

            answers = []
            answer_scores = []
            for t in answer_ind_tuples:
                answer_words = question_text[t[0]:t[1]].rstrip().split(' ')
                answer_words = [aw for aw in answer_words[1:] if aw not in stop_words]

                answers.append(answer_words)
                answer_scores.append([1.]*len(answer_words))

            annotations = annotations.split('|')
            words_and_scores = [(annotation[:-1].lower(), int(annotation[-1])) for annotation in annotations]
            assert all([score <= num_annontators for _, score in words_and_scores])

            words_and_percents = [(w, s/num_annontators) for w, s in words_and_scores]

            q_words, percents = map(list, zip(*words_and_percents))

            for answer, answer_score in zip(answers, answer_scores):

                words = ['[BOS]'] + q_words + answer + ['[EOS]']
                labels = [0.] + percents + answer_score + [0.]

                words_list.append(words)
                labels_list.append(labels)

            if cutoff is not None and len(words_list) >= cutoff:
                break

    logger.info('There are {} lines in the data'.format(len(words_list)))

    max_length = max([len(w) for w in words_list])

    pad_token = '[PAD]'
    all_input_masks = []
    all_words_list = []
    all_labels_list = []

    for words, labels in zip(words_list, labels_list):
        assert len(words) == len(labels)
        padding_length = max_length - len(words)
        input_mask = [1]*len(words) + [0]*padding_length

        all_input_masks.append(input_mask)
        all_words_list.append(words + [pad_token]*padding_length)
        all_labels_list.append(labels + [0.]*padding_length)

    assert all([len(input) == len(words) == len(labels) for input, words, labels in zip(all_input_masks, all_words_list, all_labels_list)])

    all_token_list = []
    word_to_idx = {}
    for sent in all_words_list:
        new_sent = []
        for word in sent:
            word = ''.join([c for c in word if c.isalpha()])
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)

            new_sent.append(word_to_idx[word])
        all_token_list.append(new_sent)

    return all_token_list, all_input_masks, all_labels_list, word_to_idx


def create_embedding_matrix(word_to_idx):
    nlp = spacy.load("en_core_web_md", disable=['ner', 'parser'])

    embedding_matrix = torch.empty((len(word_to_idx), 300))

    assert isinstance(word_to_idx, dict)
    for word, id in word_to_idx.items():
        if nlp.vocab.has_vector(word):
            embedding_matrix[id, :] = torch.tensor(nlp.vocab.get_vector(word))
        else:
            logger.info('The token {} does not have a vector. Replacing with noise.'.format(word))
            embedding_matrix[id, :] = torch.rand((300,))

    return embedding_matrix


class LSTM2MLP(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim):
        super(LSTM2MLP, self).__init__()

        self.embedding_matrix = embedding_matrix
        self.embedding_dim = embedding_matrix.shape[1]

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, 1),
                                    nn.Sigmoid())
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, input_ids, input_masks, labels):
        # each should be batch_size x max_length

        batch_size = input_ids.shape[0]
        max_length = input_ids.shape[1]

        # now these are max_length x batch_size
        # labels = labels.t()
        # input_masks = input_masks.t()

        inputs = torch.empty((batch_size, max_length, self.embedding_dim))
        for s_ind, sentence in enumerate(input_ids):
            for m_ind in range(max_length):
                inputs[s_ind, m_ind, :] = self.embedding_matrix[sentence[m_ind]]

        lstm_out, (last_hidden, last_cell) = self.lstm(inputs)
        lstm_out = torch.mean(lstm_out.view(batch_size, max_length, 2, self.hidden_dim), dim=2)

        # expecting lstm_out to be batch_size x max_length x hidden_dim
        out_scores = self.linear(lstm_out.view(-1, self.hidden_dim))

        # expecting out_scores to be batch_size*max_length x 1
        out_errors = self.loss(out_scores, labels.reshape((-1, 1)))
        out_errors = out_errors * input_masks.reshape((-1, 1))
        sum_errors = torch.sum(out_errors.view((batch_size, max_length)), dim=1)
        sum_masks = torch.sum(input_masks, dim=1)
        out_errors = torch.mean(sum_errors/sum_masks)

        return out_errors


def train(batch_size, epochs, hidden_dim, cutoff=None):

    all_token_list, all_input_masks, all_labels_list, word_to_idx = get_words(cutoff)

    embedding_matrix = create_embedding_matrix(word_to_idx)

    model = LSTM2MLP(embedding_matrix = embedding_matrix, hidden_dim=hidden_dim)
    # throw model to device
    model.to(device)

    # intiailize optimizer
    optimizer = optim.Adam(params=model.parameters())

    tokens_tensor = torch.tensor(all_token_list, dtype=torch.long)
    input_masks_tensor = torch.tensor(all_input_masks, dtype=torch.long)
    labels_tensor = torch.tensor(all_labels_list, dtype=torch.float)

    dataset = TensorDataset(tokens_tensor, input_masks_tensor, labels_tensor)

    train_sampler = RandomSampler(dataset, replacement=False)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    train_iterator = trange(int(epochs), desc="Epoch")
    # start training
    logger.info('Starting to train!')
    logger.info('There are {} examples.'.format(len(dataset)))

    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration, batch size {}".format(batch_size))
        for iteration, batch in enumerate(epoch_iterator):

            model.zero_grad()

            # get batch
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'input_masks': batch[1],
                      'labels': batch[2],
                      }

            # send through model
            model.train()
            error = model(**inputs)

            # backwards pass
            error.backward()
            optimizer.step()

            print('model params')
            for i in range(len(list(model.parameters()))):
                print(i)
                print(list(model.parameters())[i].grad)
                print(torch.max(list(model.parameters())[i].grad))
            print('hi')

            logger.info('The error is {}'.format(error))


if __name__ == '__main__':
    # Setup logging
    num_noise_logging_files = len(glob.glob('logging/loggingnoise_*'))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename='logging/loggingnoise_{}'.format(num_noise_logging_files))

    epochs = 10
    batch_size = 12
    hidden_dim = 100

    train(batch_size, epochs, hidden_dim, cutoff=50)
