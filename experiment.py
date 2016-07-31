import simplejson as json
import string
import argparse
from tqdm import tqdm
spacy = None


def load_spacy():
    """Load spacy."""
    from spacy.en import English
    global spacy
    spacy = English()
    print('Spacy loaded.')


def tokenize_text(c):
    translator = str.maketrans({
        key: None for key in string.punctuation})
    c = c.translate(translator)
    if spacy is None:
        for word in c.replace('-', ' ').split():
            word = word.lower()
            yield word
    else:
        for word in list(spacy(c, tag=False)):
            word = word.orth_.lower()
            yield word


def get_context_vocab(data):
    """Get the set of words in the paragraphs of data."""
    set_of_words_in_data = set()
    for article in tqdm(data):
        for paragraph in article['paragraphs']:
            c = paragraph['context']
            c_tokens = tokenize_text(c)
            set_of_words_in_data |= set(c_tokens)
    return set_of_words_in_data


def compare_context_vocabs(train, dev):
    """Compare the vocabularies in the train and dev sets."""
    print("Comparing Context Vocabularies")
    train_words = get_context_vocab(train)
    print("# Unique Context Words in train:", len(train_words))
    dev_words = get_context_vocab(dev)
    print("# Unique Context Words in dev:", len(dev_words))
    difference_words = dev_words - train_words
    print("# Unique Context Words in dev not in train:", len(difference_words))


def compare_dev_answer_vocab_with_context(train, dev):
    print("Comparing Context / Answer Vocabularies")

    def get_answer_vocab(data):
        """Get the set of words in the answers of data."""
        set_of_words_in_data = set()
        for article in tqdm(data):
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    answer = qa['answers'][0]
                    for answer in qa['answers']:
                        a = answer['text']
                        a_tokens = tokenize_text(a)
                        tok_set = set(a_tokens)
                        set_of_words_in_data |= tok_set
        return set_of_words_in_data

    train_context_words = get_context_vocab(train)
    print("# Unique Context Words in train:", len(train_context_words))
    dev_answer_words = get_answer_vocab(dev)
    print("# Unique Answer Words in dev:", len(dev_answer_words))
    difference_words = dev_answer_words - train_context_words
    print("# Unique Answer Words in dev but not in context train:",
          len(difference_words))


def load_data_file(filepath):
    """Load the json file, and check the version."""
    with open(filepath) as data_file:
        parsed_file = json.load(data_file)
        if (parsed_file['version'] != '1.0'):
            raise ValueError('Dataset version unrecognized.')
        return parsed_file['data']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SQuAD model')
    parser.add_argument('-d', '--dev', help='dev file', required=True)
    parser.add_argument('-t', '--train', help='train file', required=True)
    parser.add_argument('--use-spacy', dest='spacy', action='store_true')
    parser.add_argument('--stats', dest='stats', action='store_true')
    args = parser.parse_args()
    train, dev = load_data_file(args.train), load_data_file(args.dev)
    if (args.spacy is True):
        load_spacy()
    if (args.stats is True):
        compare_context_vocabs(train, dev)
        compare_dev_answer_vocab_with_context(train, dev)
