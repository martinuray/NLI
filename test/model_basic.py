from BILSTM.model import FAIRModel


indice = {
    'OOV':0,
    'the':1,
    'thing':2
}
model = FAIRModel(max_sequence=100, word_indice=indice, batch_size=10, num_classes=3, vocab_size=1000,
                  embedding_size=300, lstm_dim=1024)

