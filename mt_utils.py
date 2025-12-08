import numpy as np
from nltk.tokenize import word_tokenize


def tokenize(data):
    data['EN_tokens'] = data['EN'].apply(lambda x: word_tokenize(x.lower(),
                                                                 language='english'))
    data['ES_tokens'] = data['ES'].apply(lambda x: word_tokenize(x.lower(),
                                                                 language='spanish'))


def append_start_end_token(data):
    data['EN_tokens'] = data['EN_tokens'].apply(lambda x: np.concatenate((['<START>'], x, ['<END>'])))
    data['ES_tokens'] = data['ES_tokens'].apply(lambda x: np.concatenate((['<START>'], x, ['<END>'])))


def create_vocabulary(sentences):
    vocab = set()
    for sentence in sentences:
        vocab.update(sentence)
    vocab = list(vocab)
    w_to_i = {word: index for index, word in enumerate(vocab)}
    i_to_w = {index: word for index, word in enumerate(vocab)}

    return vocab, w_to_i, i_to_w


def map_to_index(data, w_to_i_en, w_to_i_es):
    data['EN_index'] = data['EN_tokens'].apply(lambda x: [w_to_i_en[word] for word in x])
    data['ES_index'] = data['ES_tokens'].apply(lambda x: [w_to_i_es[word] for word in x])
