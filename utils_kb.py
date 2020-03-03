import dgl
import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader, TensorDataset
import math
import spacy

# logging
logger = logging.getLogger(__name__)


def compute_edge_values(input_ids, num_nodes, threshold):
    # initialize torch square 2D ones array with dimension [number of nodes, number of nodes]
    edge_values = torch.ones((num_nodes, num_nodes), dtype=torch.float)

    # make a list of all rows which have the indices first, then double for loop through that to get intersections
    row_input_ids = []
    for ell in range(num_nodes):
        ell_inds = (input_ids == ell).nonzero(as_tuple=False)
        unique_ell_rows = set(ell_inds[:, 0].squeeze(0).tolist())
        row_input_ids.append(unique_ell_rows)

    # for each lower triangle index pair in the matrix
        # find the rows which the indices exist
        # calculate the number of rows for each
        # take intersection and calculate number of elements
        # use these to caculate PMI
        # if the value is above the threshold add it to the matrix, otherwise replace with 0
        # make matrix symmetric
    for i in range(1, num_nodes):
        for j in range(i):
            num_i = len(row_input_ids[i])
            num_j = len(row_input_ids[j])

            num_i_j = len(row_input_ids[i].intersection(row_input_ids[j]))

            pmi = math.log(num_i_j/(num_i*num_j))

            if pmi >= threshold:
                edge_values[i, j] = pmi
                edge_values[j, i] = pmi
            else:
                edge_values[i, j] = 0.
                edge_values[j, i] = 0.

    # return matrix
    return edge_values


def get_word_embeddings(vocabulary):
    # load in spacy model for word embeddings
    nlp = spacy.load("en_core_web_md")
    tokens = nlp(' '.join(vocabulary))

    embeddings = torch.empty((len(vocabulary), 300))

    for token_index, token in enumerate(tokens):
        if token.has_vector:
            embeddings[token_index, :] = torch.tensor(token.vector)
        else:
            logger.info('The token {} does not have a vector. Replacing with noise.'.format(token))
            embeddings[token_index, :] = torch.rand((300,))

    return embeddings


def build_graph(args, dataset, vocabulary):
    # assuming vocabulary is a list of words in order of mytokenizer, so the 0th word is the 0 represented in the tokenizer etc.
    assert isinstance(dataset, TensorDataset)

    num_nodes = len(vocabulary)

    edge_values = compute_edge_values(dataset[0], num_nodes, args.pmi_threshold) # TODO this is wrong, need to prep the input_ids first

    g = dgl.DGLGraph()

    g.add_nodes(num_nodes)

    embeddings = get_word_embeddings(vocabulary)
    g.ndata['embedding'] = embeddings

    non_zero_edge_values = edge_values.nonzero(as_tuple=True)
    dest = non_zero_edge_values[:, 0]
    source = non_zero_edge_values[:, 1]

    g.add_edges(dest, source)

    g.edges[dest, source].data['value'] = edge_values[non_zero_edge_values]

    return g


def save_graph(args, G):
    raise NotImplementedError


def load_graph(args, dataset):
    # if the filename exists load it, otherwise build it
    raise NotImplementedError





