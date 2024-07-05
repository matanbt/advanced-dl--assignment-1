import torch
from torch import nn, optim

class GenericClassifier(nn.Module):

    def __init__(self,

                 # Dataset-related params:
                 vocab_size,  # number of tokens in the vocabulary
                 n_classes,  # number of classes to predict
                 block_size,  # max sequence length (for positional embeddings)

                 # Model (arch.) Hyperparameters:
                 encoder_module: nn.Module,  # the encoder module
                 hidden_dim, num_layers,  # Encoder model's HPs
                 ):
        super(GenericClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        # TODO init method?

        # Embed the tokens and their positions with a learnable embedding matrix
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.positional_embeddings = nn.Embedding(block_size, hidden_dim)

        # The encoder model further processes the hidden states (and outputs new hidden states)
        self.encoder = encoder_module(hidden_dim, num_layers, batch_first=True)

        # Linear heads for tasks; for classification (e.g., ListOps) and for LM (next-token prediction)
        # These will be applied after some pooling of the hidden states
        self.classifier_head = nn.Linear(hidden_dim, n_classes)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self,
                sentence,  # Bsz, SeqLen
                ):
        B, L = sentence.size()

        # Embed tokens:
        tok_emb = self.word_embeddings(sentence)  # B, L, H

        # Embed position (following GPT2 method for positional embeddings):
        pos = torch.arange(0, L, dtype=torch.long, device=sentence.device)
        pos_emb = self.positional_embeddings(pos)  # L, H

        # Init hidden representation:
        out = tok_emb + pos_emb  # B, L, H

        out = self.encoder(out)  # Bsz, SeqLen, HiddenDim

        # For LM (next-token prediction)
        logits_vocab = self.lm_head(out)  # Bsz, SeqLen, VocabSize

        # For classification:
        last_out = out[:, -1, :]  # Bsz, HiddenDim;  performs last hidden-state pooling  # TODO could've also been max- or mean-pooling!
        logits_class = self.classifier_head(last_out)  # Bsz, NumClasses

        return logits_class, logits_vocab

