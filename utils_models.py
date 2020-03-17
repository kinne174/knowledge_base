import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import math
import dgl
import dgl.function as df
from scipy.sparse import csr_matrix, tril
from collections import Counter

# logging
logger = logging.getLogger(__name__)


class GraphBlock(nn.Module):
    def __init__(self, args, knowledge_base, good_counter=None, all_counter=None):
        super(GraphBlock, self).__init__()

        self.args = args # the input from user

        # Graph Neural Network
        self.knowledge_base = knowledge_base.to(args.device)

        # simple models
        self.linear1 = nn.Linear(in_features=args.word_embedding_dim, out_features=args.mlp_hidden_dim)
        self.lstm = nn.LSTM(args.mlp_hidden_dim, args.lstm_hidden_dim)
        self.linear2 = nn.Linear(args.lstm_hidden_dim, 1)

        self.loss_function = nn.BCEWithLogitsLoss()

        # dicts to help with posterior edge values used in evaluation
        self.good_edge_connections = Counter() if good_counter is None else good_counter
        self.all_edge_connections = Counter() if all_counter is None else all_counter

    @classmethod
    def from_pretrained(cls, args, knowledge_base, good_filename=None, all_filename=None):

        # if files exist for counters load them, the parameters are done outside
        if good_filename is None and all_filename is None:
            return cls(args, knowledge_base)

        with open(good_filename, 'rb') as gf:
            good_counter = torch.load(gf)
        with open(all_filename, 'rb') as af:
            all_counter = torch.load(af)

        return cls(args, knowledge_base, good_counter, all_counter)

    def forward(self, training, **inputs):
        '''
        :param training: bool if training or evaluating
        :param knowledge_base: DGL graph that needs to be subsetted based on the inputs
        :param **inputs: 'input_ids': batch size x 4 x args.max_length , 'label' : batch size * 4 - one hot vector
        :return: error of predicted versus real labels
        '''

        input_ids = inputs['input_ids']
        input_mask = inputs['input_mask']
        labels = inputs['labels']

        # for evaluation assume they come in one at a time, 1 dimensional

        batch_size = input_ids.shape[0]

        # if evaluating, don't care about this
        if labels is None:
            labels = torch.zeros((batch_size, 4), dtype=torch.float)
            labels = labels.to(self.args.device)

        all_answer_scores = torch.empty((batch_size, 4)).to(self.args.device)

        # keep track of these to update good and all edge connections
        all_edge_data = []
        all_ids_mapping = []

        # for each batch
        for b in range(batch_size):
            edge_data_l = []
            ids_mapping_l = []

            # for each input
            for i in range(4):
                # subset graph based on current input, grab nodes of inputs and first neighbor nodes and all corresponding edges, create a copy so can change edges/ nodes
                current_input_ids = input_ids[b, i, :torch.sum(input_mask[b, i, :])]
                subset_knowledge_base, ids_mapping = self.subset_graph(self.knowledge_base, current_input_ids)

                # if training cast edges to {0, 1} in a bernoulli fashion based on word2vec function, else return as is
                subset_knowledge_base = self.cast_edges(subset_knowledge_base, training, ids_mapping)

                # aggregate nodes in a mean/ summing way, can try to do this with dgl
                subset_knowledge_base.update_all(message_func=df.u_mul_e('embedding', 'value', 'embedding_'),
                                                 reduce_func=df.mean('embedding_', out='embedding_mean'))

                # MLP on aggregated nodes, not clear if output needs to be the same dimension or not
                h = subset_knowledge_base.ndata.pop('embedding_mean')
                transformed_nodes = self.linear1(h)

                # LSTM on transformed nodes in order of the input
                lstm_transformed_nodes = torch.cat([transformed_nodes[ids_mapping[int(cui)], :] for cui in current_input_ids]).reshape((current_input_ids.shape[0], 1, -1))
                lstm_out, (last_hidden, last_cell) = self.lstm(lstm_transformed_nodes)

                # sentence embedding through an MLP to get a scalar score of accuracy
                answer_score = self.linear2(last_hidden.squeeze())

                all_answer_scores[b, i] = answer_score

                # list of 2d tuples with tensors, tuple[0] is sender nodes and tuple[1] is receiver nodes
                edge_data_l.append(subset_knowledge_base.edges())
                # list of ids mapping
                ids_mapping_l.append(ids_mapping)

            all_edge_data.append(edge_data_l)
            all_ids_mapping.append(ids_mapping_l)

        # soft max over logits, may need to transform again if output from MLP is not "correct"
        softmaxed_scores = F.softmax(all_answer_scores, dim=1)

        # calculate error using Binary Cross Entropy loss
        error = self.loss_function(softmaxed_scores, labels)

        if training:
            # calculate prediction
            predictions = torch.argmax(softmaxed_scores, dim=1)

            # update good and all edge connections, if the model was correct add these edges to good connections
            correct = [labels[i, p].item() for i, p in zip(range(labels.shape[0]), predictions)]
            for c, batch_edges, batch_ids_map in zip(correct, all_edge_data, all_ids_mapping):
                pairings = []
                for edges, ids_map in zip(batch_edges, batch_ids_map):
                    # maps from subsetted node indices to original node indices
                    reverse_ids_map = {v: k for k, v in ids_map.items()}

                    # map from subset to original and add these connections to current pairings
                    pairings.extend([(reverse_ids_map[s], reverse_ids_map[r]) for s, r in zip(edges[0].tolist(), edges[1].tolist())])

                pairings_set = set(pairings)
                if c == 1:
                    # update good counter with edge connections
                    self.good_edge_connections.update(pairings_set)

                # update all counters
                self.all_edge_connections.update(pairings_set)

        # return error and individual predictions for each element in batch
        return error, softmaxed_scores

    def subset_graph(self, G, input_ids):
        # subset dgl graph G and return a copy
        # input_ids is a 1 dimensional tensor of length args.max_length
        assert isinstance(G, dgl.DGLGraph)

        unique_ids = torch.unique(input_ids).tolist()

        # use adjancency matrix to find all edge connections in original graph
        adj_matrix = G.adjacency_matrix_scipy()
        assert adj_matrix.format == 'csr'

        # indptr is a list of length num_rows + 1, for row i the column indices which data exists is located in indices[indptr[i]:indptr[i+1]]
        # great explanation here: https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr
        indptr = adj_matrix.indptr.tolist()
        indices = adj_matrix.indices.tolist()

        # keep track of all nodes and edges used
        all_nodes = []
        all_edges = []
        for ui in unique_ids:
            if indptr[ui+1] - indptr[ui] > 0:
                # find all nodes current ui is connected to
                col_indices = indices[indptr[ui]:indptr[ui+1]]

                # keep all relevant nodes
                all_nodes.extend(col_indices)

                # keep all edges by defining sender<->receiver nodes
                for ci in col_indices:
                    all_edges.append((ui, ci))

        # only keep unique nodes
        all_nodes = list(set(all_nodes))
        all_nodes.sort()

        dest, src = map(list, zip(*all_edges))

        # in subsetted graph nodes should be 0,1,...
        new_dest = [all_nodes.index(d) for d in dest]
        new_src = [all_nodes.index(s) for s in src]

        new_G = dgl.DGLGraph()

        # new graph node data still used the old embedding indices, but this is why I had to map previously
        new_G.add_nodes(len(all_nodes))
        new_G.ndata['embedding'] = G.ndata['embedding'][all_nodes]

        # add undirected edges
        new_G.add_edges(new_src, new_dest)
        new_G.add_edges(new_dest, new_src)

        # old edges to represent undirectional graph and grab edge data from original graph
        src_dest = src + dest
        dest_src = dest + src

        new_G.edata['value'] = G.edges[src_dest, dest_src].data['value']

        # maps from original G nodes indices to new node indices
        ids_mapping = {ui: all_nodes.index(ui) for ui in all_nodes}

        new_G = new_G.to(self.args.device)

        return new_G, ids_mapping

    def cast_edges(self, G, training, id_mapping):
        assert isinstance(G, dgl.DGLGraph)

        if not training:
            # use posterior information to make new edges, the more times a connection was used in a correct answer the more weight it gets
            edges = G.edges()

            # map from subset node indices to original
            reverse_id_mapping = {v: k for k, v in id_mapping.items()}

            # find edge values in original graph setting and use good and all connections dicts to calculate posterior value
            new_edge_values = []
            for s, r in zip(edges[0].tolist(), edges[1].tolist()):
                old_s = reverse_id_mapping[s]
                old_r = reverse_id_mapping[r]
                old_s_r = (old_s, old_r)

                if old_s == old_r:
                    new_edge_values.append(1)
                    continue

                num_good_connections = self.good_edge_connections[old_s_r]
                if num_good_connections == 0:
                    new_edge_values.append(0)
                    continue

                num_all_connections = self.all_edge_connections[old_s_r]

                assert 0 <= num_good_connections <= num_all_connections

                new_edge_values.append(num_good_connections/num_all_connections)

            new_edge_values = torch.tensor(new_edge_values).reshape((-1,))
            new_edge_values = new_edge_values.to(self.args.device)

            G.edata['value'] = new_edge_values

            return G

        else:  # if training
            # randomly decide whether an edge exists (1) or not (0) depending on depressing for high probability and lifting for low probability transformation function
            edge_values = G.edata['value'].tolist()

            def f(edge_val, parameter):
                return float(np.random.binomial(1, 1-math.sqrt(parameter/edge_val), None))  # make sure returning a float

            new_edge_values = torch.tensor([f(e, self.args.edge_parameter) for e in edge_values]).reshape((-1,))

            new_edge_values = new_edge_values.to(self.args.device)

            G.edata['value'] = new_edge_values

            return G
