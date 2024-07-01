from lstm import LSTMClassifier
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data_processing import ListOpsDataset, TORCH_IGNORE_INDEX, OpenWebTextDataset
from torch.nn import functional as F


# Load the dataset
# task = 'auto_regressive'
# text_dataset = OpenWebTextDataset()

def setting_1__directly_on_listops(batch_size=256):
    task = 'classification'  # we train directly on listops

    train_dataset = ListOpsDataset(split='train', n_samples=1500, task=task)  # TODO modify n_samples
    tokenizer, vocab_size = train_dataset.tokenizer, train_dataset.tokenizer.vocab_size
    test_dataset = ListOpsDataset(split='test', tokenizer=tokenizer, n_samples=10, task=task)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # TODO decouple model
    # Define model parameters
    hidden_dim = 28
    num_layers = 3
    n_classes = 10  # listops has 10 classes

    # Instantiate the model
    model = LSTMClassifier(hidden_dim, vocab_size, num_layers, n_classes=10)

    train(model, train_dataloader, test_dataloader, num_epochs=5, task=task)


def train(model, train_dataloader, test_dataloader, num_epochs=10, task='classification'):
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
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss / tot}')

    # Test the model for classification
    # TODO model.eval
    with torch.no_grad():
        correct, total = 0, 0
        for X, y in test_dataloader:
            outputs = model(X)
            predicted = outputs[0].argmax(dim=-1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        print(f'Accuracy: {100 * correct / total}%')


setting_1__directly_on_listops()
