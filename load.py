"""Module to load the dataset."""
import simplejson as json
import sys


class Loader:
    """Load the dataset."""

    def __init__(self, filepath):
        """Init."""
        self.data = None
        self.version = None
        self._load(filepath)

    def _load(self, filepath):
        with open(filepath) as data_file:
            parsed_file = json.load(data_file)
            self.data = parsed_file['data']
            self.version = parsed_file['version']

    def get_data(self):
        """Get the loaded data."""
        return self.data

    def get_article_titles(self):
        """Generator for article titles."""
        if (self.version != '1.0'):
            raise ValueError('Dataset version unrecognized.')
        return map(lambda x: x['title'], self.data)

if __name__ == '__main__':
    if(len(sys.argv) <= 1):
        raise ValueError('Give location of dataset json file.')
    dl = Loader(sys.argv[1])
    print("Successfully loaded data.")
