
from BILSTM.model import FAIRModel
from BILSTM.data_manager import *
from parameter import *
from collections import Counter
import pickle
from BILSTM.common import *

def tokenize(string):
    string = re.sub(r'\(|\)', '', string)
    return string.split()

def build_voca():
    # word is valid if count > 10 or exists in GLOVE
    run_size = 100
    voca = Counter()
    glove_voca_list = glove_voca()
    max_length = 0
    mnli_train = load_nli_data(path_dict["training_mnli"])
    for datum in mnli_train:
        s1_tokenize = tokenize(datum['sentence1_binary_parse'])
        s2_tokenize = tokenize(datum['sentence2_binary_parse'])
        for token in s1_tokenize + s2_tokenize:
            voca[token] += 1

        if len(s1_tokenize) > max_length:
            max_length = len(s1_tokenize)
        if len(s2_tokenize) > max_length:
            max_length = len(s2_tokenize)
    print(len(voca))
    print("Max length : {}".format(max_length))

    word2idx = dict()
    word2idx["<OOV>"] = 0
    word2idx["<PADDING>"] = 1
    idx = 2
    glove_found = 0
    for word, count in voca.items():
        if count > 10 or word in glove_voca_list:
            word2idx[word] = idx
            idx += 1
        if word in glove_voca_list:
            glove_found += 1
    print(len(word2idx))
    print("Glove found : {}".format(glove_found))
    return word2idx

def load_voca():
    return pickle.load(open("pickle\\word2idx", "rb"))


def get_model():
    indice = load_voca()
    model = FAIRModel(max_sequence=100, word_indice=indice, batch_size=10, num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)


def transform_corpus(max_sequence = 400):
    voca = load_voca()
    mnli_train = load_nli_data(path_dict["training_mnli"])
    def convert(tokens):
        OOV = 0
        l = []
        for t in tokens:
            if t in voca:
                l.append(voca[t])
            else:
                l.append(OOV)
            if len(l) == max_sequence:
                break
        while len(l) < max_sequence:
            l.append(1)
        return np.array(l), len(tokens)


    data = []
    for datum in mnli_train:
        s1_tokenize = tokenize(datum['sentence1_binary_parse'])
        s2_tokenize = tokenize(datum['sentence2_binary_parse'])

        s1, s1_len = convert(s1_tokenize)
        s2, s2_len = convert(s2_tokenize)
        label = datum["label"]
        y = label
        data.append({
            'p':s1,
            'p_len':s1_len,
            'h':s2,
            'h_len':s2_len,
            'y':y})

    pickle.dump(data, open("pickle\\np_corpus", "wb"))
    return data


def train():
    voca = load_voca()
    model = FAIRModel(max_sequence=400, word_indice=voca, batch_size=10, num_classes=3, vocab_size=1000,
                      embedding_size=300, lstm_dim=1024)
    data = pickle.load(open("pickle\\np_corpus", "rb"))
    epochs = 10
    model.train(epochs, data)



if __name__ == "__main__":
    action = "transform train"
    if "build_voca" in action:
        word2idx = build_voca()
        pickle.dump(word2idx, open("pickle\\word2idx", "wb"))

    # reformat corpus
    if "transform" in action:
        transform_corpus()

    if "train" in action:
        train()
