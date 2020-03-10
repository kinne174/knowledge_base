import os
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import brown
import random
import json_lines
import tqdm
import logging

logger = logging.getLogger(__name__)

random_words = set(brown.words(categories='news'))


class ArcExample(object):

    def __init__(self, example_id, sentences, sentence_type, changed_words_indices, label):
        self.example_id = example_id
        self.sentence_features = [{
            'sentence': s,
            'changed_word_indices': cwi,
        } for s, cwi in zip(sentences, changed_words_indices)]
        self.label = label
        self.sentence_type = sentence_type


def domain_finder(args, question, contexts, answers):

    question_words = word_tokenize(question.lower())
    question_in_domain = any([dw in question_words for dw in args.domain_words])

    if not question_in_domain:
        for answer in answers:
            answer_words = word_tokenize(answer.lower())
            question_in_domain = any([dw in answer_words for dw in args.domain_words])

            if question_in_domain:
                break

    all_context_sentences_in_domain = []
    for context in contexts:
        context_sentences = sent_tokenize(context)

        for context_sentence in context_sentences:
            context_words = word_tokenize(context_sentence.lower())

            if any([dw in context_words for dw in args.domain_words]):
                all_context_sentences_in_domain.append(context_sentence)

    return question_in_domain, all_context_sentences_in_domain


def attention_loader(words):
    # TODO make this based on that one paper with essential learning
    # return 1 change index and then three noisy sentences
    center_index = random.randint(1, len(words)-2)
    changed_inds = [1 if (center_index - 1) <= i <= (center_index + 2) else 0 for i in range(len(words))]
    return changed_inds


def noisy_sentences(words, changed_indices):
    # TODO change this to sample something intelligent
    noise_sentences = []
    for _ in range(3):
        noise_words = [w if ci == 0 else random.sample(random_words, 1)[0].lower() for w, ci in zip(words, changed_indices)]
        noise_sentences.append(' '.join(noise_words))

    return noise_sentences


def examples_loader(args):
    # returns an object of type ArcExample similar to hugging face transformers

    # bad ids, each has at least one answer that does not contain any context
    # if another question answer task is used this will need to be fixed
    bad_ids = ['OBQA_9-737', 'OBQA_45', 'OBQA_750', 'OBQA_7-423', 'OBQA_619', 'OBQA_9-778', 'OBQA_10-201', 'OBQA_10-791', 'OBQA_10-1138', 'OBQA_12-717', 'OBQA_13-129', 'OBQA_13-468', 'OBQA_13-957', 'OBQA_14-10', 'OBQA_14-949', 'OBQA_14-1140', 'OBQA_14-1274']

    subsets = ['train', 'dev', 'test']

    all_examples = []
    logger_ind = 0

    if args.only_context:

        for subset in subsets:

            data_filename = os.path.join(args.data_dir, '{}.jsonl'.format(subset))
            with open(data_filename, 'r') as file:
                jsonl_reader = json_lines.reader(file)

                # this will show up when running on console
                for json_ind, line in tqdm.tqdm(enumerate(jsonl_reader), desc='Creating {} examples.'.format(subset), mininterval=1):

                    id = line['id']

                    if id in bad_ids:
                        continue

                    # if the number of options is not equal to 4 update the logger and skip it, all of the formatting works with
                    # 4 options, maybe can update in the future to put a dummy one there or set the probability to 0 that it is
                    # selected as correct later
                    if len(line['question']['choices']) != 4:
                        logger.info('Question id {} did not contain four options. Skipped it.'.format(id))
                        continue

                    label = line['answerKey']
                    if label not in '1234':
                        if label not in 'ABCD':
                            logger.info('Question id {} had an incorrect label of {}. Skipped it'.format(id, label))
                            continue
                    # label should be the position in the list that the correct answer is
                        else:
                            label = ord(label) - ord('A')
                    else:
                        label = int(label) - 1

                    # extract question text, answer texts and contexts
                    question_text = line['question']['stem']
                    contexts = [c['para'] for c in line['question']['choices']]
                    answer_texts = [c['text'] for c in line['question']['choices']]

                    question_in_domain, sentences_in_domain = domain_finder(args, question_text, contexts, answer_texts)

                    # if question is in the domain then create an ArcExample and add it to examples
                    if question_in_domain: # this should be a bool
                        question_answer_sentences = []
                        question_answer_indices = []
                        for answer in answer_texts:
                            question_answer_sentences.append(' '.join([question_text, answer]))
                            question_answer_indices.append([0]*len(' '.split(question_text)) + [1]*len(' '.split(answer)))

                        all_examples.append(ArcExample(example_id=id,
                                                       sentences=question_answer_sentences,
                                                       changed_words_indices=question_answer_indices,
                                                       sentence_type=1,
                                                       label=label))

                        if len(all_examples) >= logger_ind*100:
                            logger.info('Writing {} example.'.format(logger_ind*100))
                            logger_ind += 1

                    # if sentence_in_domain is not empty, create noise and add to examples
                    if sentences_in_domain: # this should be a list
                        for sentence_ind, sentence in enumerate(sentences_in_domain):
                            sentence_words = word_tokenize(sentence)

                            if len(sentence_words) <= 5:
                                continue

                            changed_words_indices = attention_loader(sentence_words)

                            assert len(changed_words_indices) == len(sentence_words), 'indices length and words length do not match'

                            noise_sentences = noisy_sentences(sentence_words, changed_words_indices)

                            sentence_label_tuples = [(sentence, 1)] + [(ns, 0) for ns in noise_sentences]

                            random.shuffle(sentence_label_tuples)

                            all_sentences = [t[0] for t in sentence_label_tuples]
                            label, = [i for i, t in enumerate(sentence_label_tuples) if t[1] == 1]

                            all_examples.append(ArcExample(example_id='{}-{}-{}'.format(subset, json_ind, sentence_ind),
                                                           sentences=all_sentences,
                                                           changed_words_indices=changed_words_indices,
                                                           sentence_type=0,
                                                           label=label))

                            if len(all_examples) >= logger_ind * 100:
                                logger.info('Writing {} example.'.format(logger_ind * 100))
                                logger_ind += 1

                    if args.cutoff is not None and len(all_examples) >= args.cutoff:
                        all_examples = all_examples[:args.cutoff]
                        break
    else:
        data_filename = '../ARC/ARC-V1-Feb2018-2/ARC_Corpus.txt'
        with open(data_filename, 'r') as file:
            jsonl_reader = json_lines.reader(file)

            # this will show up when running on console
            for json_ind, line in tqdm.tqdm(enumerate(jsonl_reader), desc='Creating corpus examples.',
                                            mininterval=1):

                sentence_words = word_tokenize(line.lower())

                if not any([dw in sentence_words for dw in args.domain_words]):
                    continue

                if len(sentence_words) <= 5:
                    continue

                changed_words_indices = attention_loader(sentence_words)

                assert len(changed_words_indices) == len(
                    sentence_words), 'indices length and words length do not match'

                noise_sentences = noisy_sentences(sentence_words, changed_words_indices)

                sentence_label_tuples = [(sentence_words, 1)] + [(ns, 0) for ns in noise_sentences]

                random.shuffle(sentence_label_tuples)

                all_sentences = [t[0] for t in sentence_label_tuples]
                label, = [i for i, t in enumerate(sentence_label_tuples) if t[1] == 1]

                all_examples.append(ArcExample(example_id='CORPUS_{}'.format(len(all_examples)),
                                               sentences=all_sentences,
                                               changed_words_indices=changed_words_indices,
                                               sentence_type=0,
                                               label=label))

                if len(all_examples) >= logger_ind * 100:
                    logger.info('Writing {} example.'.format(logger_ind * 100))
                    logger_ind += 1

                if args.cutoff is not None and len(all_examples) >= args.cutoff:
                    all_examples = all_examples[:args.cutoff]
                    break

    # make sure there is at least one example
    assert len(all_examples) > 1, 'No examples in the domain!'

    # informative
    logger.info('Number of examples in returned is {}'.format(len(all_examples)))

    return all_examples
