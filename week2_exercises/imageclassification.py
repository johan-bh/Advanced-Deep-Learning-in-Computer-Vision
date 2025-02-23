import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from vit import ViT
from vit_analysis import plot_hyperparameter_impact, visualize_attention
import optuna
import pandas as pd

# Constants
CIFAR_MEAN = (0.5, 0.5, 0.5)
CIFAR_STD = (0.5, 0.5, 0.5)

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def select_two_classes_from_cifar10(dataset, classes):
    idx = (np.array(dataset.targets) == classes[0]) | (np.array(dataset.targets) == classes[1])
    dataset.targets = np.array(dataset.targets)[idx]
    dataset.targets[dataset.targets == classes[0]] = 0
    dataset.targets[dataset.targets == classes[1]] = 1
    dataset.targets = dataset.targets.tolist()
    dataset.data = dataset.data[idx]
    return dataset

def prepare_dataloaders(batch_size, classes=[3, 7]):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    try:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        os.makedirs('./data', exist_ok=True)
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    trainset = select_two_classes_from_cifar10(trainset, classes=classes)
    testset = select_two_classes_from_cifar10(testset, classes=classes)

    train_size, test_size = 5000, 1000
    trainset, _ = random_split(trainset, [train_size, len(trainset) - train_size])
    testset, _ = random_split(testset, [test_size, len(testset) - test_size])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader, trainset, testset

def train_and_evaluate(image_size=(32, 32), patch_size=(4, 4), channels=3, 
                       embed_dim=128, num_heads=4, num_layers=4, num_classes=2,
                       pos_enc='learnable', pool='cls', dropout=0.3, fc_dim=None, 
                       num_epochs=20, batch_size=16, lr=1e-4, warmup_steps=625,
                       weight_decay=1e-3, gradient_clipping=1):
    """Train and evaluate the model, returning the best validation accuracy and model."""
    loss_function = nn.CrossEntropyLoss()
    trainloader, testloader, _, _ = prepare_dataloaders(batch_size=batch_size)

    model = ViT(image_size=image_size, patch_size=patch_size, channels=channels,
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim,
                num_classes=num_classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))

    best_val_loss = float('inf')
    best_accuracy = 0.0
    best_model_state = None

    for e in range(num_epochs):
        model.train()
        train_loss = 0
        for image, label in tqdm(trainloader, desc=f'Epoch {e+1}/{num_epochs}', leave=False):
            image, label = image.to(device), label.to(device)
            opt.zero_grad()
            out = model(image)
            loss = loss_function(out, label)
            loss.backward()
            train_loss += loss.item()
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            sch.step()

        train_loss /= len(trainloader)

        val_loss = 0
        with torch.no_grad():
            model.eval()
            tot, cor = 0.0, 0.0
            for image, label in testloader:
                image, label = image.to(device), label.to(device)
                out = model(image)
                loss = loss_function(out, label)
                val_loss += loss.item()
                out = out.argmax(dim=1)
                tot += float(image.size(0))
                cor += float((label == out).sum().item())
            acc = cor / tot
            val_loss /= len(testloader)
            print(f'Epoch {e+1}: train_loss={train_loss:.3f}, val_acc={acc:.3f}, val_loss={val_loss:.3f}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_accuracy = acc
                best_model_state = model.state_dict()

    torch.save(best_model_state, 'best_model.pth')
    model.load_state_dict(best_model_state)
    return best_accuracy, model, testloader

def objective(trial):
    """Optuna objective function to optimize hyperparameters."""
    # Define hyperparameter search space
    embed_dim = trial.suggest_categorical('embed_dim', [128, 256])
    num_heads = trial.suggest_categorical('num_heads', [4, 8])
    num_layers = trial.suggest_categorical('num_layers', [4, 6])
    dropout = trial.suggest_categorical('dropout', [0.1, 0.3])
    pool = trial.suggest_categorical('pool', ['cls', 'mean'])
    lr = trial.suggest_categorical('lr', [1e-4, 3e-4])
    num_epochs = trial.suggest_int('num_epochs', 5, 20)  # Searchable epochs

    # Ensure embed_dim is divisible by num_heads
    if embed_dim % num_heads != 0:
        raise optuna.TrialPruned()

    set_seed(1)
    accuracy, model, testloader = train_and_evaluate(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        pool=pool,
        lr=lr,
        num_epochs=num_epochs
    )

    # Store additional trial info
    trial.set_user_attr('model', model)
    trial.set_user_attr('testloader', testloader)
    return accuracy

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model will run on {device}")

    # Create Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)  # Adjust number of trials as needed

    # Collect results
    results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            result = {
                'trial_number': trial.number,
                'embed_dim': trial.params['embed_dim'],
                'num_heads': trial.params['num_heads'],
                'num_layers': trial.params['num_layers'],
                'dropout': trial.params['dropout'],
                'pool': trial.params['pool'],
                'lr': trial.params['lr'],
                'num_epochs': trial.params['num_epochs'],
                'accuracy': trial.value
            }
            results.append(result)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('hyperparameter_results.csv', index=False)
    print("\nResults saved to 'hyperparameter_results.csv'")

    # Analyze results
    print("\nAnalyzing results...")
    plot_hyperparameter_impact(results)

    # Get best trial
    best_trial = study.best_trial
    best_accuracy = best_trial.value
    best_model = best_trial.user_attrs['model']
    best_testloader = best_trial.user_attrs['testloader']
    print(f"\nBest accuracy: {best_accuracy:.3f}")
    print("Best Configuration:")
    for k, v in best_trial.params.items():
        print(f"{k}: {v}")

    # Visualize attention maps for best model
    sample_batch = next(iter(best_testloader))
    sample_images, _ = sample_batch
    sample_image = sample_images[0]
    visualize_attention(best_model, sample_image, save_path='attention_maps.png')

    # Summary statistics
    accuracies = [r['accuracy'] for r in results]
    print("\nSummary Statistics:")
    print(f"Mean accuracy: {np.mean(accuracies):.3f}")
    print(f"Std accuracy: {np.std(accuracies):.3f}")
    print(f"Min accuracy: {np.min(accuracies):.3f}")
    print(f"Max accuracy: {np.max(accuracies):.3f}")