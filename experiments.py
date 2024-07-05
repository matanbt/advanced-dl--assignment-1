from torch.utils.data import DataLoader
from data_processing import ListOpsDataset, TORCH_IGNORE_INDEX, OpenWebTextDataset
from train import train
from generic_classifier import GenericClassifier

BLOCK_SIZE = 1024

def setting_1__directly_on_listops(model_cls, model_kwargs, batch_size=256):
    train_dataset = ListOpsDataset(split='train', task='classification')  # for debug: `n_samples=1500`
    tokenizer = train_dataset.tokenizer
    n_classes = train_dataset.n_classes
    vocab_size = tokenizer.vocab_size

    test_dataset = ListOpsDataset(split='test', tokenizer=tokenizer, task='classification')  # for debug: `n_samples=10`
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model
    model = model_cls(vocab_size=vocab_size, n_classes=n_classes, block_size=BLOCK_SIZE, **model_kwargs)

    train(model, train_dataloader, test_dataloader, num_epochs=5, task='classification')


def setting_2__clm_pretrain_text_then_listops(model_cls, model_kwargs, batch_size=256):
    # 1. Load data and init model
    # 1.A. Text:
    text_dataset = OpenWebTextDataset()  # for debug: `n_samples=1500`
    tokenizer = text_dataset.tokenizer
    vocab_size = tokenizer.vocab_size
    # 1.B. ListOps:
    listops_train_dataset = ListOpsDataset(split='train', tokenizer=tokenizer, task='classification')
    n_classes = listops_train_dataset.n_classes
    listops_test_dataset = ListOpsDataset(split='test', tokenizer=tokenizer, task='classification')

    # 1.C. Init model
    model = model_cls(vocab_size=vocab_size, n_classes=n_classes, block_size=BLOCK_SIZE, **model_kwargs)

    # 2. Causal-LM pretrain on other dataset (e.g. OpenWebText)
    print("Performing auto-regressive/causal LM training on OpenWebText dataset.")
    text_train_dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=True)
    text_test_dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=True)  # TODO make it an held-out test-set

    train(model, text_train_dataloader, text_test_dataloader, num_epochs=5, task='auto_regressive')

    # 3. Train on ListOps
    print("Fine-tuning on ListOps dataset.")
    train_dataloader = DataLoader(listops_train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(listops_test_dataset, batch_size=batch_size, shuffle=True)

    train(model, train_dataloader, test_dataloader, num_epochs=5, task='classification')


def setting_3__clm_pretrain_listops_then_listops(model_cls, model_kwargs, batch_size=256):
    # 1. Load data and init model
    listops_train_dataset = ListOpsDataset(split='train', task='auto_regressive')
    tokenizer = listops_train_dataset.tokenizer
    vocab_size = tokenizer.vocab_size
    n_classes = listops_train_dataset.n_classes
    listops_test_dataset = ListOpsDataset(split='test', tokenizer=tokenizer, task='auto_regressive')

    # 1.C. Init model
    model = model_cls(vocab_size=vocab_size, n_classes=n_classes, block_size=BLOCK_SIZE, **model_kwargs)

    # 2. Causal-LM pretrain on other dataset (e.g. OpenWebText)
    print("Performing auto-regressive/causal-LM training on ListOps dataset.")
    train_dataloader = DataLoader(listops_train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(listops_test_dataset, batch_size=batch_size, shuffle=True)
    train(model, train_dataloader, test_dataloader, num_epochs=5, task='auto_regressive')

    # 3. Train on ListOps
    print("Fine-tuning on ListOps dataset.")
    listops_train_dataset.task = 'classification'
    listops_test_dataset.task = 'classification'
    train_dataloader = DataLoader(listops_train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(listops_test_dataset, batch_size=batch_size, shuffle=True)
    train(model, train_dataloader, test_dataloader, num_epochs=5, task='classification')


if __name__ == '__main__':
    from models.transformer import TransformerEncoder
    from models.s4 import S4Encoder
    from models.lstm import LSTMEncoder, LSTMTorchEncoder
    setting_1__directly_on_listops(GenericClassifier,
                                   {'hidden_dim': 64, 'num_layers': 5,
                                    'encoder_module': S4Encoder})