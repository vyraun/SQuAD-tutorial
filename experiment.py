import simplejson as json
import string
import argparse
spacy = None


def load_spacy():
    """Load spacy."""
    from spacy.en import English
    global spacy
    spacy = English()


def compare_context_vocabs(train, dev):
    """Compare the vocabularies in the train and dev sets."""
    def get_context_vocab(data):
        """Get the set of words in the paragraphs of data."""
        set_of_words_in_data = set()

        def add_context_to_word_set(c):
            """Perform transformations before adding to word set."""
            translator = str.maketrans({
                key: None for key in string.punctuation})
            c = c.translate(translator)
            if spacy is None:
                for word in c.split():
                    word = word.lower()
                    set_of_words_in_data.add(word)
            else:
                for word in list(spacy(c, tag=False)):
                    word = word.orth_.lower()
                    set_of_words_in_data.add(word)

        for article in data:
            for paragraph in article['paragraphs']:
                c = paragraph['context']
                add_context_to_word_set(c)

        return set_of_words_in_data

    print("Comparing Context Vocabularies")
    train_words = get_context_vocab(train)
    print("# Unique Words in train:", len(train_words))
    dev_words = get_context_vocab(dev)
    print("# Unique Words in dev:", len(dev_words))
    difference_words = dev_words - train_words
    print("# Unique Words in dev not in train:", len(difference_words))


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
    args = parser.parse_args()
    if (args.spacy is True):
        load_spacy()
    train = load_data_file(args.train)
    dev = load_data_file(args.dev)
    compare_context_vocabs(train, dev)
