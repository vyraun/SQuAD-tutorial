from __future__ import print_function

import sys
import regex
import numpy as np

from functools import reduce
from load import Loader
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.engine.training import slice_X
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

def compile_model(vocab_size, story_maxlen, query_maxlen):
    RNN = recurrent.LSTM
    EMBED_HIDDEN_SIZE = 50
    SENT_HIDDEN_SIZE = 100
    QUERY_HIDDEN_SIZE = 100
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
    return model

def train_model(model, X_train, y_train, BATCH_SIZE=32, EPOCHS=1):
    model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        nb_epoch=EPOCHS,
        validation_split=0.05
    )

def test_model(model, X_test, Y_test, BATCH_SIZE=32):
    loss, acc = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


def validate_model(model, X_val, y_val, inv):
    class colors:
        ok = '\033[92m'
        fail = '\033[91m'
        close = '\033[0m'
    for i in range(10):
        ind = np.random.randint(0, len(X_val))
        rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowX, verbose=0)
        print(preds)

        q = inv(rowX[0])
        correct = inv(rowy[0])
        guess = inv(preds[0], calc_argmax=False)
        print('Q', q[::-1])
        print('T', rowy[0])
        print(colors.ok + '☑' + colors.close if correct == guess
              else colors.fail + '☒' + colors.close, guess)
        print('---')


def main(train_raw, test_raw):
    train = list(keep_single_answer_pqas(generate_pqas(train_raw)))
    test = list(keep_single_answer_pqas(generate_pqas(test_raw, is_test=True)))

    vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer])
                          for story, q, answer in train + test)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    word_idx_inv = dict((i + 1, c) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))

    vX, vXq, vY = vectorize_stories(
        train, word_idx, story_maxlen, query_maxlen)
    vtX, vtXq, vtY = vectorize_stories(
        test, word_idx, story_maxlen, query_maxlen)

    X, y = [vX, vXq], vY
    X_test, y_test = [vtX, vtXq], vtY

    print('vocab = {}'.format(vocab))
    print('X.shape = {}'.format(vX.shape))
    print('Xq.shape = {}'.format(vXq.shape))
    print('Y.shape = {}'.format(vY.shape))
    print('story_maxlen, query_maxlen = {}, {}'.format(
        story_maxlen, query_maxlen))
    print('Build model...')

    model = compile_model(len(vocab) + 1, story_maxlen, query_maxlen)

    print('Training...')

    for iteration in range(1, 200):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        train_model(model, X, y)
        #validate_model(model, X, y, word_idx_inv)
    test_model(model, X_test, y_test)


if __name__ == '__main__':
    if(len(sys.argv) <= 2):
        raise ValueError('Give location of dataset train and dev json file.')
    train_data = Loader(sys.argv[1]).get_data()
    dev_data = Loader(sys.argv[1]).get_data()
    main(train_data, dev_data)
