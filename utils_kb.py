import dgl
from dgl.data.utils import save_graphs, load_graphs
import torch
import logging
import math
import spacy
import os

# logging
logger = logging.getLogger(__name__)


def compute_edge_values(input_ids, num_nodes, threshold):
    # initialize torch square 2D ones array with dimension [number of nodes, number of nodes]
    edge_values = torch.ones((num_nodes, num_nodes), dtype=torch.float)

    # make a list of all rows which have the indices first, then double for loop through that to get intersections
    row_input_ids = [{}]
    for ell in range(1, num_nodes):
        ell_inds = (input_ids == ell).nonzero(as_tuple=False)
        unique_ell_rows = set(ell_inds[:, 0].tolist())
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
            freq_i = len(row_input_ids[i])/input_ids.shape[0]
            freq_j = len(row_input_ids[j])/input_ids.shape[0]

            freq_i_j = len(row_input_ids[i].intersection(row_input_ids[j]))/input_ids.shape[0]

            npmi = math.log(freq_i*freq_j)/math.log(freq_i_j) - 1 if freq_i_j > 0 else -1

            if npmi >= threshold:
                edge_values[i, j] = npmi
                edge_values[j, i] = npmi
            else:
                edge_values[i, j] = 0.
                edge_values[j, i] = 0.

    # return matrix
    return edge_values


def get_word_embeddings(vocabulary):
    # load in spacy model for word embeddings
    nlp = spacy.load("en_core_web_md", disable=['ner', 'parser'])
    # tokens = nlp(' '.join(vocabulary))

    embeddings = torch.empty((len(vocabulary), 300))

    for token_index, token in enumerate(vocabulary):
        if nlp.vocab.has_vector(token):
            embeddings[token_index, :] = torch.tensor(nlp.vocab.get_vector(token))
        else:
            logger.info('The token {} does not have a vector. Replacing with noise.'.format(token))
            embeddings[token_index, :] = torch.rand((300,))

    return embeddings


def build_graph(args, dataset, vocabulary):
    # assuming vocabulary is a list of words in order of mytokenizer, so the 0th word is the 0 represented in the tokenizer etc.

    num_nodes = len(vocabulary)

    input_ids = dataset.tensors[0]
    input_ids = input_ids.reshape((-1, input_ids.shape[-1]))
    random_indices = torch.randint(low=0, high=4, size=(dataset.tensors[0].shape[0],)) + torch.arange(start=0, end=input_ids.shape[0], step=4)
    randomed_inputed_ids = input_ids[random_indices, :]
    edge_values = compute_edge_values(randomed_inputed_ids, num_nodes, args.pmi_threshold)

    g = dgl.DGLGraph()

    g.add_nodes(num_nodes)

    embeddings = get_word_embeddings(vocabulary)
    g.ndata['embedding'] = embeddings

    non_zero_edge_values = edge_values.nonzero(as_tuple=False)
    dest = non_zero_edge_values[:, 0]
    source = non_zero_edge_values[:, 1]

    g.add_edges(dest, source)

    non_zero_edge_values = edge_values.nonzero(as_tuple=True)
    edges_to_add = edge_values[non_zero_edge_values]
    g.edges[dest, source].data['value'] = edges_to_add

    return g


def save_graph(args, G):
    cutoff_str = '' if args.cutoff is None else '_cutoff{}'.format(args.cutoff)
    graph_filename = os.path.join(args.cache_dir, 'graph{}.py'.format(cutoff_str))

    logger.info('Saving graph to {}'.format(graph_filename))

    save_graphs(graph_filename, G)

    return -1


def load_graph(args):
    cutoff_str = '' if args.cutoff is None else '_cutoff{}'.format(args.cutoff)
    graph_filename = os.path.join(args.cache_dir, 'graph{}.py'.format(cutoff_str))

    logger.info('Loading graph from {}'.format(graph_filename))

    glist, _ = load_graphs(graph_filename)
    G = glist[0]

    return G


