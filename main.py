from __future__ import print_function

import sys
import regex
import numpy as np
import pickle
from spacy.en import English
import itertools
from functools import reduce
from load import Loader
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.engine.training import slice_X
from keras.layers import Dense, Merge, Dropout, RepeatVector
from keras.layers import recurrent
from keras.models import Sequential

nlp = English()

np.random.seed(1337)

def transform_token(word):
    return word.orth_.lower()

def vectorize_stories(
    data, word_idx_X, word_idx_y, story_maxlen, query_maxlen
):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx_X[transform_token(w)] for w in story]
        xq = [word_idx_X[transform_token(w)] for w in query]
        y = np.zeros(len(word_idx_y) + 1)
        # let's not forget that index 0 is reserved
        y[word_idx_y[transform_token(answer[0])]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)

    return pad_sequences(X, maxlen=story_maxlen), \
        pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)


def generate_pqas(data, is_test=False):
    for index, article in enumerate(data):
        print(article['title'])
        for paragraph in article['paragraphs']:
            try:
                c = list(nlp(paragraph['context'], tag=False))
                for qa in paragraph['qas']:
                    q = list(nlp(qa['question'], tag=False))
                    for answer in qa['answers']:
                        a = list(nlp(answer['text'], tag=False))
                        if (len(a) == 1):
                            yield (c, q, a)
                            if is_test:
                                break
            except Exception:
                continue

def compile_model(vocab_X_size, vocab_y_size, story_maxlen, query_maxlen, embedding_weights):
    RNN = recurrent.LSTM
    EMBED_HIDDEN_SIZE = 300
    sentrnn = Sequential()
    sentrnn.add(Embedding(
        vocab_X_size,
        EMBED_HIDDEN_SIZE,
        input_length=story_maxlen,
        weights=[embedding_weights],
        mask_zero=True)
    )
    sentrnn.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=True))
    # sentrnn.add(Dropout(0.3))

    qrnn = Sequential()
    qrnn.add(Embedding(
        vocab_X_size,
        EMBED_HIDDEN_SIZE,
        input_length=query_maxlen,
        weights=[embedding_weights],
        mask_zero=True)
    )
    # qrnn.add(Dropout(0.3))
    qrnn.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=True))
    qrnn.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
    qrnn.add(RepeatVector(story_maxlen))

    model = Sequential()
    model.add(Merge([sentrnn, qrnn], mode='sum'))
    model.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
    # model.add(Dropout(0.3))
    model.add(Dense(vocab_y_size, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, BATCH_SIZE=16, EPOCHS=10):
    model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        nb_epoch=EPOCHS,
        validation_split=0.05
    )

def test_model(model, X_test, Y_test, BATCH_SIZE=16):
    loss, acc = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

def main(train_raw, test_raw):
    train = list(generate_pqas(train_raw))
    test = list(generate_pqas(test_raw, is_test=True))

    vocab_y = sorted(
        reduce(
            lambda x, y: x | y,
            (
                set([transform_token(word) for word in answer])
                for s, d, answer in train + test
            )
        )
    )

    word_idx_y = dict((c, i + 1) for i, c in enumerate(vocab_y))

    vocab_X = sorted(
        reduce(
            lambda x, y: x | y,
            (
                set(
                    [transform_token(word) for word in story] +
                    [transform_token(word) for word in q])
                for story, q, _ in train + test
            )
        )
    )

    word_idx_X = dict((c, i + 1) for i, c in enumerate(vocab_X))

    word_vectors = dict()
    for story, q, _ in train + test:
        for word in story + q:
            word_vectors[transform_token(word)] = word.vector

    story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))

    vX, vXq, vY = vectorize_stories(
        train, word_idx_X, word_idx_y, story_maxlen, query_maxlen)
    vtX, vtXq, vtY = vectorize_stories(
        test, word_idx_X, word_idx_y, story_maxlen, query_maxlen)

    X, y = [vX, vXq], vY
    X_test, y_test = [vtX, vtXq], vtY

    print('X.shape = {}'.format(vX.shape))
    print('Xq.shape = {}'.format(vXq.shape))
    print('Y.shape = {}'.format(vY.shape))
    print('story_maxlen, query_maxlen = {}, {}'.format(
        story_maxlen, query_maxlen))
    print('Build model...')

    vocab_dim = 300 # dimensionality of your word vectors
    vocab_X_length = len(vocab_X) + 1
    embedding_weights = np.zeros((vocab_X_length, vocab_dim))
    for word, index in word_idx_X.items():
        embedding_weights[index, :] = word_vectors[word]

    model = compile_model(
        vocab_X_length,
        len(vocab_y) + 1,
        story_maxlen,
        query_maxlen,
        embedding_weights
    )

    print('Training...')

    #for iteration in range(1, 200):
    train_model(model, X, y)
    #validate_model(model, X, y, word_idx_inv)
    test_model(model, X_test, y_test)


if __name__ == '__main__':
    if(len(sys.argv) <= 2):
        raise ValueError('Give location of dataset train and dev json file.')
    train_data = Loader(sys.argv[1]).get_data()
    dev_data = Loader(sys.argv[2]).get_data()
    main(train_data, dev_data)
