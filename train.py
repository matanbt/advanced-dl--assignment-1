from torch import nn, optim
import torch
from data_processing import TORCH_IGNORE_INDEX
from torch.nn import functional as F

from utils import print_num_params
from livelossplot import PlotLosses
import time


def train(model: nn.Module, train_dataloader, test_dataloader, num_epochs=10,
          task='classification', device='cuda'):
    """

    :param model: the model to train. We assume two outputs - logits for vocabulary and logits for classification.
    :param train_dataloader: data to train on.
    :param test_dataloader: data to test on.
    :param num_epochs:
    :return:
    """

    print_num_params(model.encoder)
    plotlosses = PlotLosses()
    start_time = time.time()

    # Define criterion for each task
    criterion_lm = lambda logits, targets: F.cross_entropy(logits.view(-1, logits.size(-1)),
                                                           targets.view(-1),
                                                           ignore_index=TORCH_IGNORE_INDEX)
    criterion_cls = nn.CrossEntropyLoss()

    # Init optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)  [TODO OPTIONAL]

    # Train the model
    model.train()
    for epoch in range(num_epochs):
        avg_loss, tot = 0, 0
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits_class, logits_vocab = model(X)
            if task == 'classification':
                loss = criterion_cls(logits_class, y)
            elif task == 'auto_regressive':
                loss = criterion_lm(logits_vocab, y)
            loss.backward()
            optimizer.step()

            # aggregate metrics
            avg_loss += loss.item()
            tot += 1
            plotlosses.update({'train_loss': loss.item()})
            plotlosses.send()
            # TODO validation loss on each step

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss / tot}, time elapsed: {(time.time() - start_time) / 60:.2f}mins')

    # Test the model for classification
    model.eval()
    if task == 'classification':
        with torch.no_grad():
            correct, total = 0, 0
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                predicted = outputs[0].argmax(dim=-1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            print(f'Accuracy: {100 * correct / total}%')
    # TODO support auto-regressive task validation

