from lstm import LSTMClassifier
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data_processing import ListOpsDataset, TORCH_IGNORE_INDEX, OpenWebTextDataset
from torch.nn import functional as F

# TODO control n_samples and n_epoch to discard test setting


def train(model, train_dataloader, test_dataloader, num_epochs=10, task='classification'):
    # TODO add val-set support! (through the training)
    # TODO live plot it
    """

    :param model: the model to train. We assume two outputs - logits for vocabulary and logits for classification.
    :param train_dataloader: data to train on.
    :param test_dataloader: data to test on.
    :param num_epochs:
    :return:
    """
    # this criterion works for both classification and auto-regressive tasks (with the targets defined in the ListOpsDataset)
    criterion_lm = lambda logits, targets: F.cross_entropy(logits.view(-1, logits.size(-1)),
                                                           targets.view(-1),
                                                           ignore_index=TORCH_IGNORE_INDEX)
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(num_epochs):
        avg_loss, tot = 0, 0
        for X, y in train_dataloader:
            optimizer.zero_grad()
            logits_class, logits_vocab = model(X)
            if task == 'classification':
                loss = criterion_cls(logits_class, y)
            elif task == 'auto_regressive':
                loss = criterion_lm(logits_vocab, y)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            tot += 1
        # TODO validation loss on each step
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss / tot}')

    # Test the model for classification
    # TODO model.eval
    if task == 'classification':
        with torch.no_grad():
            correct, total = 0, 0
            for X, y in test_dataloader:
                outputs = model(X)
                predicted = outputs[0].argmax(dim=-1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            print(f'Accuracy: {100 * correct / total}%')
    # TODO support auto-regressive task validation