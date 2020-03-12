from utils_real_data import ArcExample
import tqdm
import logging

# logging
logger = logging.getLogger(__name__)


class ArcFeature(object):
    def __init__(self, example_id, input_features, sentence_type, label=None):
        # label is 0,1,2,3 depending on correct answer
        self.example_id = example_id
        self.input_features = [{
            'input_ids': input_id,
            'input_mask': input_mask,
        } for (input_id, input_mask) in input_features]
        self.label = label
        self.sentence_type = sentence_type


def features_loader(args, tokenizer, examples):
    # returns a list of objects of type ArcFeature similar to hugging face transformers
    break_flag = False
    all_features = []
    for ex_ind, ex in tqdm.tqdm(enumerate(examples), desc='Examples to Features'):
        if ex_ind % 1000 == 0:
            logger.info('Converting example number {} of {} to features.'.format(ex_ind, len(examples)))
        assert isinstance(ex, ArcExample)

        input_features = []
        for sentence in ex.sentences:

            try:
                inputs = tokenizer.encode(
                    sentence,
                    add_special_tokens=False,
                    max_length=args.max_length
                )
            except AssertionError as err_msg:
                logger.info('Assertion error at example id {}: {}'.format(ex_ind, err_msg))
                break_flag = True
                break

            if 'num_truncated_tokens' in inputs and inputs['num_truncated_tokens'] > 0:
                logger.info('Truncating context for question id {}'.format(ex.example_id))

            input_ids = inputs['input_ids']

            input_mask = [1]*len(input_ids)

            padding_length = args.max_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [0]*padding_length
                input_mask = input_mask + [0]*padding_length

            assert len(input_ids) == args.max_length
            assert len(input_mask) == args.max_length

            # the token_type_mask and attention_mask is the same so can just use token_type_mask twice
            input_features.append((input_ids, input_mask))

        if break_flag:
            break_flag = False
            continue

        if ex_ind == 0:
            logger.info('Example of features used. input_ids is the tokenized form of the sentences,\n input mask is 0 in positions there is padding and 1 otherwise\n'
                        'sentence type is 0 if from context and 1 if from a question,\n label is the index of the correct sentence')
            logger.info('Question ID: {}'.format(ex.example_id))
            logger.info('input_ids: {}'.format(' :: '.join([' '.join([str(ii) for ii in i_f[0]]) for i_f in input_features])))
            logger.info('input_mask: {}'.format(' :: '.join([' '.join([str(ii) for ii in i_f[1]]) for i_f in input_features])))
            logger.info('sentence type: {}'.format(ex.sentence_type))
            logger.info('label: {}'.format(ex.label))

        all_features.append(ArcFeature(example_id=ex.example_id,
                                       input_features=input_features,
                                       sentence_type=ex.sentence_type,
                                       label=ex.label))

    return all_features
