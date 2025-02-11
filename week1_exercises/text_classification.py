import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
from torchtext import data, datasets, vocab
import tqdm
import itertools

from transformer import TransformerClassifier, to_device

NUM_CLS = 2
VOCAB_SIZE = 50_000
SAMPLED_RATIO = 0.2
MAX_SEQ_LEN = 512

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def prepare_data_iter(sampled_ratio=0.2, batch_size=16):
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)
    tdata, _ = datasets.IMDB.splits(TEXT, LABEL)
    # Reduce dataset size
    reduced_tdata, _ = tdata.split(split_ratio=sampled_ratio)
    # Create train and test splits
    train, test = reduced_tdata.split(split_ratio=0.8)
    print('training: ', len(train), 'test: ', len(test))
    TEXT.build_vocab(train, max_size=VOCAB_SIZE - 2)
    LABEL.build_vocab(train)
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), 
        batch_size=batch_size, 
        device=to_device()
    )
    return train_iter, test_iter

def main(embed_dim=256, num_heads=16, num_layers=12, num_epochs=20,
         pos_enc='learnable', pool='max', dropout=0.0, fc_dim=None,
         batch_size=16, lr=1e-4, warmup_steps=625, 
         weight_decay=1e-4, gradient_clipping=1):
    """
    Train the TransformerClassifier with the given hyperparameters.
    Returns the best validation accuracy achieved over the training epochs.
    """
    loss_function = nn.CrossEntropyLoss()

    train_iter, test_iter = prepare_data_iter(sampled_ratio=SAMPLED_RATIO, 
                                              batch_size=batch_size)

    model = TransformerClassifier(embed_dim=embed_dim, 
                                  num_heads=num_heads, 
                                  num_layers=num_layers,
                                  pos_enc=pos_enc,
                                  pool=pool,  
                                  dropout=dropout,
                                  fc_dim=fc_dim,
                                  max_seq_len=MAX_SEQ_LEN, 
                                  num_tokens=VOCAB_SIZE, 
                                  num_classes=NUM_CLS)
    
    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))

    best_acc = 0.0

    # Training loop
    for e in range(num_epochs):
        print(f'\nEpoch {e}')
        model.train()
        for batch in tqdm.tqdm(train_iter, desc="Training"):
            opt.zero_grad()
            input_seq = batch.text[0]
            # Optionally, limit sequence length
            if input_seq.size(1) > MAX_SEQ_LEN:
                input_seq = input_seq[:, :MAX_SEQ_LEN]
            # Adjust labels to be zero-indexed
            label = batch.label - 1
            out = model(input_seq)
            loss = loss_function(out, label)
            loss.backward()
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            sch.step()

        # Validation loop after each epoch
        with torch.no_grad():
            model.eval()
            tot, cor = 0.0, 0.0
            for batch in test_iter:
                input_seq = batch.text[0]
                if input_seq.size(1) > MAX_SEQ_LEN:
                    input_seq = input_seq[:, :MAX_SEQ_LEN]
                label = batch.label - 1
                predictions = model(input_seq).argmax(dim=1)
                tot += float(input_seq.size(0))
                cor += float((label == predictions).sum().item())
            acc = cor / tot
            best_acc = max(best_acc, acc)
            print(f'-- Validation accuracy: {acc:.3f}')

    print(f'Best validation accuracy achieved: {best_acc:.3f}')
    return best_acc

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model will run on {device}")

    # Define hyperparameter grid for grid search
    grid = {
        'embed_dim': [128, 256],
        'num_heads': [8, 16],
        'num_layers': [6, 12],
        'dropout': [0.0, 0.1],
        'lr': [1e-4, 5e-5],
        'pool': ['cls', 'max']
    }

    # List to store grid search results
    results = []

    # Iterate over all combinations in the grid
    for embed_dim, num_heads, num_layers, dropout, lr, pool in itertools.product(
            grid['embed_dim'], grid['num_heads'], grid['num_layers'], 
            grid['dropout'], grid['lr'], grid['pool']):
        
        # Skip invalid combinations where embed_dim is not divisible by num_heads
        if embed_dim % num_heads != 0:
            continue
        
        print("\n===============================================")
        print(f"Running experiment with: embed_dim={embed_dim}, "
              f"num_heads={num_heads}, num_layers={num_layers}, "
              f"dropout={dropout}, lr={lr}, pool={pool}, "
              f"pos_enc=learnable")
        print("===============================================")
        
        # Set seed for reproducibility for each experiment
        set_seed(seed=1)
        
        # Run training and get the best validation accuracy
        best_acc = main(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            num_layers=num_layers, 
            dropout=dropout, 
            lr=lr,
            pool=pool,
            pos_enc='learnable'
        )
        
        # Record the result
        results.append({
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dropout': dropout,
            'lr': lr,
            'pool': pool,
            'pos_enc': 'learnable',
            'best_val_accuracy': best_acc
        })
    
    # Sort results by accuracy
    results.sort(key=lambda x: x['best_val_accuracy'], reverse=True)
    
    # Print all grid search results
    print("\nGrid Search Results (sorted by accuracy):")
    print("\nTop 5 configurations:")
    for i, res in enumerate(results[:5], 1):
        print(f"\n{i}. Configuration:")
        print(f"   Accuracy: {res['best_val_accuracy']:.3f}")
        print(f"   Parameters:")
        for k, v in res.items():
            if k != 'best_val_accuracy':
                print(f"      {k}: {v}")
    
    # Print summary statistics
    accuracies = [r['best_val_accuracy'] for r in results]
    print("\nSummary Statistics:")
    print(f"Mean accuracy: {np.mean(accuracies):.3f}")
    print(f"Std accuracy: {np.std(accuracies):.3f}")
    print(f"Min accuracy: {np.min(accuracies):.3f}")
    print(f"Max accuracy: {np.max(accuracies):.3f}")
