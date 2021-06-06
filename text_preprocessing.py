import torch
import numpy as np
import spacy
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
from scipy.sparse import csgraph
import time
from collections import defaultdict
nlp = spacy.load("en_core_web_sm")

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

class GCN_Data:
    def __init__(self, list_name, min_count, sent_max_len):
        self.input_list_name = list_name
        self.sent_max_len = sent_max_len
        t1 = time.time()
        self.get_words(min_count)
        t2 = time.time()
        print('\nTIME', t2-t1)
        t1 = time.time()
        self.get_adj()
        t2 = time.time()
        print('\nTIME', t2-t1)
        print('Word Count: %d' % len(self.word2id))
        print('Sentence Length: %d' % (self.sentence_length))
        print('Sentence Count: %d' % (self.sentence_count))

    def get_words(self, min_count):
        self.input_list = self.input_list_name
        self.sentence_length = 0
        word_frequency = defaultdict(int)
        deps_list = ['SEP', 'compound', 'nsubj', 'aux', 'root', 'prep', 'det',
                        'amod', 'punct', 'pobj', 'nsubjpass', 'acl', 'nummod', 'dobj',
                            'auxpass', 'mark', 'advcl', 'nmod', 'cc', 'conj', 'appos', 'advmod',
                                'agent', 'npadvmod', 'relcl', 'pcomp', 'ccomp', 'poss', 'case', 'xcomp',
                                    'quantmod', 'neg', 'attr', 'csubj', 'meta', 'acomp', 'preconj', 'intj',
                                        'csubjpass', 'dep', 'expl', 'oprd', 'parataxis', 'prt', 'dative', 'predet']

        self.sents = []
        for line in self.input_list:
            for sent in sent_tokenize(line):
                self.sents.append(sent)

        for sent in self.sents:
            doc = nlp.make_doc(sent)
            self.sentence_length += len(doc)
            for w in doc:
                word_frequency[w.text.lower()] += 1

        self.deps_dict = {d:i for i, d in enumerate(deps_list)}
        self.word2id = dict()
        self.id2word = dict()
        wid = 0
        self.word_frequency = dict()
        for w, c in word_frequency.items():
            if c < min_count:
                self.sentence_length -= c
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        self.word_count = len(self.word2id)

    def get_adj(self):
        self.sentence_count = 0
        self.AdjM = []
        self.fM = []
        self.ohM = []
        # self.ohMn = []
        self.input_list = self.input_list_name
        for doc in nlp.pipe(self.sents, n_process = 1, disable=[]):
                ld = len(doc)
                if ld>self.sent_max_len:
                    # self.sent_max_len = ld
                    continue
                self.sentence_count +=1
                cAdjM = np.zeros((self.sent_max_len, self.sent_max_len))
                cfM = torch.zeros(self.sent_max_len)
                cohM = torch.zeros(self.sent_max_len)
                # cohMn = torch.zeros(self.sent_max_len)
                # neg_counter = 0
                for ti in range(ld):
                  token = doc[ti]
                  cohM[ti] = self.word2id[token.text.lower()]
                  cfM[ti] = self.deps_dict[token.dep_.lower()]
                  childs = [i for i in token.children]
                  for c in childs:
                      cAdjM[ti][c.i] = 1.
                      # sohMn.append(self.word2id[c.text.lower()])
                  # cohMn.append(sohMn)
                LapM = csgraph.laplacian(cAdjM, normed=True)
                sparse_LapM = to_sparse(torch.FloatTensor(LapM))
                self.AdjM.append(sparse_LapM)
                self.fM.append(cfM.unsqueeze(0))
                self.ohM.append(cohM.unsqueeze(0))
                # self.ohMn.append(cohMn)
