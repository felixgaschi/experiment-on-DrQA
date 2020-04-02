from bpemb import BPEmb
from .spacy_tokenizer import SpacyTokenizer
from .tokenizer import Tokenizer, Tokens


class BPETokenizer(Tokenizer):

    def __init__(self, **kwargs):
        lang = kwargs.get("lang", "en")
        vs = kwargs.get("limit", 200000)
        
        self.bpemb = BPEmb(lang=lang, vs=vs)
        self.tokenizer = SpacyTokenizer(model="en", annotators=["lemma", "pos", "ner"])
        self.annotators = self.tokenizer.annotators

    def tokenize(self, text):
        data_spacy = self.tokenizer.tokenize(text).data

        data = []
        start_ws = 0
        for i in range(len(data_spacy)):
            subwords = self.bpemb.encode(data_spacy[i][0])
            for j, sub in enumerate(subwords):
                tuple_idx = [-2, -2]
                if j == 0:
                    tuple_idx[0] = data_spacy[i][2][0]
                if j + 1 == len(subwords):
                    tuple_idx[1] = data_spacy[i][2][1]
                data.append((
                    sub,
                    data_spacy[i][1] if j == 0 else "",
                    tuple_idx,
                    data_spacy[i][3],
                    data_spacy[i][4],
                    data_spacy[i][5]
                ))
            
        return Tokens(data, self.annotators, opts={'non_ent': ''})

