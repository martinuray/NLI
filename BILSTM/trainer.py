
from BILSTM.model import FAIRModel
from BILSTM.data_manager import *
from parameter import *
from collections import Counter

def build_voca():
    def tokenize(string):
        string = re.sub(r'\(|\)', '', string)
        return string.split()

    run_size = 100
    voca = Counter()
    mnli_train = load_nli_data(path_dict["training_mnli"])
    for datum in mnli_train:
        s1_tokenize = tokenize(datum['sentence1_binary_parse'])
        s2_tokenize = tokenize(datum['sentence2_binary_parse'])
        for token in s1_tokenize + s2_tokenize:
            voca[token] += 1
    print(len(voca))

    word2idx = dict()
    idx = 1
    for word, count in voca.items():
        if count > 4 :
            word2idx[word] = idx
            idx += 1
    print(len(word2idx))
    return word2idx

def load_voca():
    ""


def get_model():
    indice = load_voca()
    model = FAIRModel(max_sequence=100, word_indice=indice, batch_size=10, num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)


if __name__ == "__main__":
    word2idx = build_voca()

    # TODO reformat corpus

    # TODO Build Model


    # TODO Start Train
