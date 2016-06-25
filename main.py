from __future__ import print_function

import sys
import regex
import numpy as np

from functools import reduce
from load import Loader
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Merge, Dropout, RepeatVector
from keras.layers import recurrent
from keras.models import Sequential

np.random.seed(1337)


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in regex.split(r'(\W+)?', sent) if x.strip()]

def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)
        # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return pad_sequences(X, maxlen=story_maxlen), \
        pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)


def generate_pqas(data, is_test=False):
    for article in data:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if is_test:
                    yield (
                        tokenize(paragraph['context']),
                        tokenize(qa['question']),
                        qa['answers'][0]['text']
                    )
                else:
                    for answer in qa['answers']:
                        yield (
                            tokenize(paragraph['context']),
                            tokenize(qa['question']),
                            answer['text']
                        )

def keep_single_answer_pqas(data):
    def is_one_word(answer):
        return len(answer.split(' ')) == 1
    for pqa in data:
        if is_one_word(pqa[2]):
            yield pqa

def main(train_raw, test_raw):
    """Main."""
    train = list(keep_single_answer_pqas(generate_pqas(train_raw)))
    test = list(keep_single_answer_pqas(generate_pqas(test_raw, is_test=True)))
    vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer])
                          for story, q, answer in train + test)))
    vocab_size = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))

    X, Xq, Y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
    tX, tXq, tY = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

    print('vocab = {}'.format(vocab))
    print('X.shape = {}'.format(X.shape))
    print('Xq.shape = {}'.format(Xq.shape))
    print('Y.shape = {}'.format(Y.shape))
    print('story_maxlen, query_maxlen = {}, {}'.format(
        story_maxlen, query_maxlen))
    print('Build model...')

    RNN = recurrent.LSTM
    EMBED_HIDDEN_SIZE = 50
    SENT_HIDDEN_SIZE = 100
    QUERY_HIDDEN_SIZE = 100
    BATCH_SIZE = 32
    EPOCHS = 40
    sentrnn = Sequential()
    sentrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE,
                          input_length=story_maxlen))
    sentrnn.add(Dropout(0.3))

    qrnn = Sequential()
    qrnn.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE,
                       input_length=query_maxlen))
    qrnn.add(Dropout(0.3))
    qrnn.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
    qrnn.add(RepeatVector(story_maxlen))

    model = Sequential()
    model.add(Merge([sentrnn, qrnn], mode='sum'))
    model.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('Training')
    model.fit([X, Xq], Y, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=0.05)
    loss, acc = model.evaluate([tX, tXq], tY, batch_size=BATCH_SIZE)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

if __name__ == '__main__':
    if(len(sys.argv) <= 2):
        raise ValueError('Give location of dataset train and dev json file.')
    train_data = Loader(sys.argv[1]).get_data()
    dev_data = Loader(sys.argv[1]).get_data()
    main(train_data, dev_data)
