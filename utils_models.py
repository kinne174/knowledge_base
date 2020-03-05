import torch
import torch.nn as nn
import logging
import numpy as np

# logging
logger = logging.getLogger(__name__)


class GraphBlock(nn.Module):
    def __init__(self, args, training,):
        super(GraphBlock, self).__init__()

        self.args = args # the input from user
        self.training = training # a bool if the model is training or not

    def forward(self, knowledge_base, **inputs):
        '''

        :param knowledge_base: DGL graph that needs to be subsetted based on the inputs
        :param **inputs: 'input_ids': batch size x 4 x args.max_length , 'label' : batch size * 4 - one hot vector
        :return: error of predicted versus real labels
        '''

        # for each batch
            # for each input
                # subset graph based on current input, grab nodes of inputs and first neighbor nodes and all corresponding edges, create a copy so can change edges/ nodes

                # if training cast edges to {0, 1} in a bernoulli fashion based on word2vec function, else return as is

                # aggregate nodes in a mean/ summing way, can try to do this with dgl

                # MLP on aggregated nodes, not clear if output needs to be the same dimension or not

                # LSTM on transformed nodes in order of the input

                # sentence embedding through an MLP to get a scalar score of accuracy

            # soft max over logits, may need to transform again if output from MLP is not "correct"

            # calculate error using Binary Cross Entropy loss

            # calculate prediction

        # return error and individual predictions for each element in batch