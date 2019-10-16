import json
import jsonlines
import os
from collections import Counter

import torch


class Dictionary(object):
    """
    A look-up machine\\
    method:\\
        add_word(): adds the position of the first appearance of the word
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    """
    Corpus of data/penn/
    """
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

class Tokenizer(object):
    """
    Tokenize .txt files to .jsonl files
    """
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(self.path)
        self.tofiles = self.files.copy()
        for i in range(len(self.tofiles)):
            self.tofiles[i] = self.tofiles[i].replace('.txt', '.jsonl')
        # print(self.tofiles)

    def forward(self):
        for filename in self.tofiles:
            file = self.path + '/' + filename
            if os.path.exists(file):
                os.remove(file)
        for idx in range(len(self.files)):
            with open(self.path+'/'+self.files[idx], 'r') as fp:
                lines = fp.readlines()
                # i = 0
                for line in lines:
                    # i += 1
                    tokenlist = line.split(' ')[1:-1]
                    # tokenlist_1 = line.split(' ')[1:]
                    tokendict = {"tokens": tokenlist}
                    # tokendict_1 = {"tokens": tokenlist_1}
                    # print(tokendict)
                    # print(tokendict_1)
                    # exit()
                    # print(i, tokens)
                    with jsonlines.open(self.path+'/'+self.tofiles[idx], mode='a') as writer:
                        writer.write(tokendict)

if __name__ == '__main__':
    tokenizer = Tokenizer('data/penn')
    tokenizer.forward()
