from torch import nn, optim


# TODO implement LSTM module
# TODO initialization?

class LSTMClassifier(nn.Module):

    def __init__(self, hidden_dim, vocab_size, num_layers, n_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        # Embed the tokens with a learnable embedding matrix
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        # Given a pooling of the hidden states, we can use a linear layer to predict the output
        # TODO project ALL the hidden states, but use only the last one for classification, and all for auto-regressive task
        self.classifier_head = nn.Linear(hidden_dim, n_classes)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self,
                sentence,  # Bsz, SeqLen
                ):
        out = self.word_embeddings(sentence)  # Bsz, SeqLen, EmbDim
        out, _ = self.lstm(out)  # Bsz, SeqLen, HiddenDim

        # For LM (next-token prediction)
        logits_vocab = self.lm_head(out)  # Bsz, SeqLen, VocabSize

        # For classification:
        last_out = out[:, -1, :]  # Bsz, HiddenDim;  performs last hidden-state pooling
        logits_class = self.classifier_head(last_out)  # Bsz, NumClasses

        return logits_class, logits_vocab

