import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import math
import dgl
import dgl.function as df
from scipy.sparse import csr_matrix, tril

# logging
logger = logging.getLogger(__name__)


class GraphBlock(nn.Module):
    def __init__(self, args, knowledge_base):
        super(GraphBlock, self).__init__()

        self.args = args # the input from user
        # self.training = training # a bool if the model is training or not

        self.knowledge_base = knowledge_base.to(args.device)

        self.linear1 = nn.Linear(in_features=args.word_embedding_dim, out_features=args.mlp_hidden_dim)
        self.lstm = nn.LSTM(args.mlp_hidden_dim, args.lstm_hidden_dim)
        self.linear2 = nn.Linear(args.lstm_hidden_dim, 1)

        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, training, **inputs):
        '''
        # TODO make sure all things that need to be thrown to device are
        :param knowledge_base: DGL graph that needs to be subsetted based on the inputs
        :param **inputs: 'input_ids': batch size x 4 x args.max_length , 'label' : batch size * 4 - one hot vector
        :return: error of predicted versus real labels
        '''

        input_ids = inputs['input_ids']
        input_mask = inputs['input_mask']
        labels = inputs['labels']

        # for evaluation assume they come in one at a time

        batch_size = input_ids.shape[0]

        # if evaluating, don't care about this
        if labels is None:
            labels = torch.zeros((batch_size, 4), dtype=torch.float)

        all_answer_scores = torch.empty((batch_size, 4)).to(self.args.device)

        # for each batch
        for b in range(batch_size):
            # for each input
            for i in range(4):
                # subset graph based on current input, grab nodes of inputs and first neighbor nodes and all corresponding edges, create a copy so can change edges/ nodes
                current_input_ids = input_ids[b, i, :torch.sum(input_mask[b, i, :])]
                subset_knowledge_base, ids_mapping = self.subset_graph(self.knowledge_base, current_input_ids)

                # if training cast edges to {0, 1} in a bernoulli fashion based on word2vec function, else return as is
                subset_knowledge_base = self.cast_edges(subset_knowledge_base, training)

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

        # soft max over logits, may need to transform again if output from MLP is not "correct"
        softmaxed_scores = F.softmax(all_answer_scores, dim=1)

        # calculate error using Binary Cross Entropy loss
        error = self.loss_function(softmaxed_scores, labels)

        # calculate prediction
        predictions = torch.argmax(softmaxed_scores, dim=1)

        # return error and individual predictions for each element in batch
        return error, predictions

    def subset_graph(self, G, input_ids):
        # subset dgl graph G and return a copy
        # input_ids is a 1 dimensional tensor of length args.max_length
        assert isinstance(G, dgl.DGLGraph)

        unique_ids = torch.unique(input_ids).tolist()

        adj_matrix = G.adjacency_matrix_scipy()
        assert adj_matrix.format == 'csr'

        indptr = adj_matrix.indptr.tolist()
        indices = adj_matrix.indices.tolist()

        all_nodes = []
        all_edges = []
        for ui in unique_ids:
            if indptr[ui+1] - indptr[ui] > 0:
                col_indices = indices[indptr[ui]:indptr[ui+1]]

                all_nodes.extend(col_indices)

                for ci in col_indices:
                    all_edges.append((ui, ci))

        all_nodes = list(set(all_nodes))
        all_nodes.sort()

        dest, src = map(list, zip(*all_edges))

        new_dest = [all_nodes.index(d) for d in dest]
        new_src = [all_nodes.index(s) for s in src]

        new_G = dgl.DGLGraph()

        new_G.add_nodes(len(all_nodes))
        new_G.ndata['embedding'] = G.ndata['embedding'][all_nodes]

        new_G.add_edges(new_src, new_dest)
        new_G.add_edges(new_dest, new_src)

        src_dest = src + dest
        dest_src = dest + src

        new_G.edata['value'] = G.edges[src_dest, dest_src].data['value']

        ids_mapping = {ui: all_nodes.index(ui) for ui in all_nodes}

        new_G = new_G.to(self.args.device)

        return new_G, ids_mapping

    def cast_edges(self, G, training):
        if not training:
            return G

        assert isinstance(G, dgl.DGLGraph)

        edge_values = G.edata['value'].tolist()

        def f(edge_val, parameter):
            return float(np.random.binomial(1, 1-math.sqrt(parameter/edge_val), None)) # make sure returning a float

        new_edge_values = torch.tensor([f(e, self.args.edge_parameter) for e in edge_values]).reshape((-1,))

        new_edge_values = new_edge_values.to(self.args.device)

        G.edata['value'] = new_edge_values

        return G





