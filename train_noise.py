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

    num_lost = 0

    with open(data_filename, 'r') as tsv_file:
        tsv_reader = csv.reader(tsv_file)

        for line in tsv_reader:
            line = ''.join(line)
            line = line.split('\t')

            question_text = line[0]
            num_annotators = int(line[1])
            annotations = line[2]

            answer_letters = ['(A)', '(B)', '(C)', '(D)', '(1)', '(2)', '(3)', '(4)']

            if not any([answer_letter in question_text for answer_letter in answer_letters]):
                logger.info('One of {} is not in the question text: {}. Skipping.'.format(', '.join(answer_letters), question_text))
                num_lost += 1
                continue

            answer_letters = [a_l for a_l in answer_letters if a_l in question_text]

            answer_inds = [question_text.index(answer_letter) for answer_letter in answer_letters] + [len(question_text)]

            answer_ind_tuples = [(answer_inds[i], answer_inds[j]) for i, j in zip(range(len(answer_inds)-1), range(1, len(answer_inds)))]

            answers = []
            answer_scores = []
            for t in answer_ind_tuples:
                answer_words = question_text[t[0]:t[1]].rstrip().split(' ')
                answer_words = [aw.lower() for aw in answer_words[1:] if aw not in stop_words]

                answers.append(answer_words)
                answer_scores.append([1.]*len(answer_words))

            annotations = annotations.split('|')
            words_and_scores = [(annotation[:-1].lower(), int(annotation[-1])) for annotation in annotations]
            assert all([score <= num_annotators for _, score in words_and_scores])

            words_and_percents = [(w, s/num_annotators) for w, s in words_and_scores]

            q_words, percents = map(list, zip(*words_and_percents))

            for answer, answer_score in zip(answers, answer_scores):

                words = ['[BOS]'] + q_words + answer + ['[EOS]']
                labels = [0.] + percents + answer_score + [0.]

                words_list.append(words)
                labels_list.append(labels)

            if cutoff is not None and len(words_list) >= cutoff:
                break

    logger.info('There are {} lines in the data'.format(len(words_list)))
    logger.info('Lost {} to no answers'.format(num_lost))

    padding_tokens = ['[EOS]', '[BOS]']
    token_list = []
    word_to_idx = {'[PAD]': 0}
    for ind1, (sent, labels) in enumerate(zip(words_list, labels_list)):
        new_sent = []
        new_labels = []
        for ind2, (word, label) in enumerate(zip(sent, labels)):
            word = ''.join([c for c in word if c.isalnum()]) if word not in padding_tokens else word

            if not nlp.vocab.has_vector(word) and word not in padding_tokens:
                spaced_tokens = [(word[:i], word[i:]) for i in range(1, len(word))]
                temp = [all((nlp.vocab.has_vector(st[0]), nlp.vocab.has_vector(st[1]))) for st in spaced_tokens]
                if any(temp):
                    if sum(temp) > 1:
                        logger.info('More than one acceptable pairing of words: {}'.format(' '.join(['({}, {})'.format(st[0], st[1]) for st, t in zip(spaced_tokens, temp) if t])))
                    best_words = spaced_tokens[temp.index(True)]
                    for best_word in best_words:
                        if best_word not in word_to_idx:
                            word_to_idx[best_word] = len(word_to_idx)

                        new_sent.append(word_to_idx[best_word])
                    new_labels.extend([label]*2)
                    continue

            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)

            new_sent.append(word_to_idx[word])
            new_labels.append(label)

        assert len(new_sent) == len(new_labels)

        token_list.append(new_sent)
        labels_list[ind1] = new_labels

    max_length = max([len(t) for t in token_list])

    all_input_masks = []
    all_tokens_list = []
    all_labels_list = []

    for tokens, labels in zip(token_list, labels_list):
        assert len(tokens) == len(labels), 'The length of tokens is {} and the length of labels is {}'.format(len(tokens), len(labels))
        padding_length = max_length - len(tokens)
        input_mask = [1]*len(tokens) + [0]*padding_length

        all_input_masks.append(input_mask)
        all_tokens_list.append(tokens + [0]*padding_length)
        all_labels_list.append(labels + [0.]*padding_length)

    assert all([len(input) == len(tokens) == len(labels) == max_length for input, tokens, labels in zip(all_input_masks, all_tokens_list, all_labels_list)])

    return all_tokens_list, all_input_masks, all_labels_list, word_to_idx


def create_embedding_matrix(word_to_idx):
    embedding_matrix = torch.empty((len(word_to_idx), 300))

    assert isinstance(word_to_idx, dict)
    for word, id in word_to_idx.items():
        if nlp.vocab.has_vector(word):
            embedding_matrix[id, :] = torch.tensor(nlp.vocab.get_vector(word))
        else:
            spaced_tokens = [(word[:i], word[i:]) for i in range(1, len(word)-1)]
            temp = [all((nlp.vocab.has_vector(st[0]), nlp.vocab.has_vector(st[1]))) for st in spaced_tokens]
            if any(temp):
                best_words = temp.index(True)

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


def save_model(model, embedding_matrix, hidden_dim):
    model_save_file = 'saved/train_noise/essential_terms_hidden{}.pt'.format(hidden_dim)

    with open(model_save_file, 'wb') as mf:
        torch.save(model.state_dict(), mf)

    embedding_matrix_save_file = 'saved/train_noise/embedding_matrix.py'

    with open(embedding_matrix_save_file, 'wb') as ef:
        torch.save(embedding_matrix, ef)


def load_model(hidden_dim):
    filenames = glob.glob('saved/train_noise/*hidden*.pt')

    model_filename = [f for f in filenames if f[(f.index('hidden')+len('hidden')):f.index('.pt')] == str(hidden_dim)]

    assert len(model_filename) == 1

    embedding_matrix_save_file = 'saved/train_noise/embedding_matrix.py'
    if os.path.exists(embedding_matrix_save_file):
        with open(embedding_matrix_save_file, 'rb') as ef:
            embedding_matrix = torch.load(ef)

    model = LSTM2MLP(embedding_matrix=embedding_matrix, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_filename[0]))
    model.eval()

    return model


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
            # Todo output predictions too for analysis
            error = model(**inputs)

            # backwards pass
            error.backward()
            optimizer.step()

            # print('model params')
            # for i in range(len(list(model.parameters()))):
            #     print(i)
            #     print(list(model.parameters())[i].grad)
            #     print(torch.max(list(model.parameters())[i].grad))
            # print('hi')

            logger.info('The error is {}'.format(error))

            break
        break

    save_model(model, embedding_matrix, hidden_dim)
    _ = load_model(hidden_dim)


if __name__ == '__main__':
    # Setup logging
    num_noise_logging_files = len(glob.glob('logging/loggingnoise_*'))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename='logging/loggingnoise_{}'.format(num_noise_logging_files))

    # output device
    logger.info('Using device: {}'.format(device))

    # parser/ embeddings
    nlp = spacy.load("en_core_web_md", disable=['ner', 'parser'])

    epochs = 10
    batch_size = 12
    hidden_dim = 100

    train(batch_size, epochs, hidden_dim)
