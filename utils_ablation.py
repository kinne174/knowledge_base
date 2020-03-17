import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
import logging
import os, glob

from utils_tokenizer import MyTokenizer

# logging
logger = logging.getLogger(__name__)

def ablation(args, subset, model, checkpoint, dataset):

    # for speeding up computation load in mapping dict for good edges, keys are senders nodes; values are all receiver nodes
    good_edge_connections = model.good_edge_connections
    sender_good = {}
    for kg in good_edge_connections.keys():
        id1, id2 = kg
        if id1 not in sender_good:
            sender_good[id1] = [id2]
        else:
            sender_good[id1].append(id2)

    all_edge_connections = model.all_edge_connections

    # create the ablation files in their own directory
    ablation_dir = os.path.join(args.output_dir, 'ablation_{}'.format(subset))

    if not os.path.exists(ablation_dir):
        os.makedirs(ablation_dir)

    ablation_filename = os.path.join(ablation_dir, 'checkpoint_{}.txt'.format(checkpoint))

    # load in tokenizer for the myind_to_word dict to help translate the ids back to words
    my_tokenizer = MyTokenizer.load_tokenizer(args, evaluating=True)
    myind_to_word = my_tokenizer.myind_to_word

    # load in lines one at a time
    train_sampler = SequentialSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=1)

    with open(ablation_filename, 'w') as af:

        # for each question in the subset
        for batch_ind, batch in enumerate(train_dataloader):
            model.eval()

            # get batch
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'input_mask': batch[1],
                      'labels': batch[2],
                      }

            _, softmaxed_scores = model(training=False, **inputs)

            inputs = {k: v.squeeze() for k, v in inputs.items()}

            # calculate prediction
            prediction = torch.argmax(softmaxed_scores, dim=1)

            input_ids = inputs['input_ids']
            input_masks = inputs['input_mask']

            # print out the question with the prediction, prediction score, and the correct label
            # get up until the index they are all the same
            change_index_list = [input_ids[1, i] == input_ids[2, i] == input_ids[3, i] == input_ids[0, i] for i in
                                 range(input_ids.shape[1])]
            if False not in change_index_list:
                continue

            change_index = change_index_list.index(False)

            # get first token which is padding to separate the answers
            pad_indices = [input_masks[i, :].tolist().index(0) for i in range(input_masks.shape[0])]

            if not all([pi > change_index for pi in pad_indices]):
                continue

            question_ids = input_ids[1, :change_index]
            answers_ids = [input_ids[i, change_index:pad_ind] for i, pad_ind in zip(range(input_ids.shape[0]), pad_indices)]

            question_text = ' '.join([myind_to_word[qi.item()] for qi in question_ids])

            # get all answer features to display
            answers_text = [' '.join([myind_to_word[ai.item()] for ai in answer_id]) for answer_id in answers_ids]
            answer_choice_text = ['A.', 'B.', 'C.', 'D.']
            correct_label = [' ' if lab == 0 else '*' for lab in inputs['labels']]
            predicted_label = [' ']*4
            predicted_label[prediction.item()] = '#'
            softmaxed_scores = [round(ss, 3) for ss in softmaxed_scores.squeeze().tolist()]

            assert len(correct_label) == len(predicted_label) == len(softmaxed_scores) == len(answer_choice_text) == len(answers_text)

            answer_features = list(map(tuple, zip(correct_label, predicted_label, softmaxed_scores, answer_choice_text, answers_text)))

            # print out the batch_ind then the question text then newline
            af.write('{}. {}\n'.format(batch_ind+1, question_text))
            # print out a * for correct answer, # for prediction, rounded softmaxed score, and then answer text for each of four options
            for (cl, pl, ss, act, at) in answer_features:
                af.write('{} {} {} {}{}\n'.format(cl, pl, ss, act, at))

            af.write('\n')

            ## print out the best connections for the subsetted graph
            # get all unique ids
            unique_ids = torch.unique(input_ids)

            # get all first neighbor connections within good and all
            relevant_best_connections = {}
            for ui in unique_ids:
                ui = ui.item()
                if ui in sender_good and sender_good[ui]:
                    for id2 in sender_good[ui]:
                        if id2 == ui:
                            continue
                        assert (ui, id2) in good_edge_connections
                        assert (ui, id2) in all_edge_connections

                        relevant_best_connections[(ui, id2)] = good_edge_connections[(ui, id2)]/float(all_edge_connections[(ui, id2)])

            # print out top n=10 in percentage
            best_connections = [(k, v) for k, v in relevant_best_connections.items()]
            best_connections.sort(key=lambda t: t[1], reverse=True)

            num_to_print = min(10, len(best_connections))
            for i in range(num_to_print):
                id1, id2 = best_connections[i][0]
                val = best_connections[i][1]
                word1 = myind_to_word[id1]
                word2 = myind_to_word[id2]

                af.write('The connection between "{}" and "{}" has value {}.\n'.format(word1, word2, round(val, 3)))

            af.write('\n\n')

    return -1


