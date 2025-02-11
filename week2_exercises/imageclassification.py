import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from vit import ViT
from vit_analysis import plot_hyperparameter_impact, visualize_attention



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
    dataset.targets[dataset.targets==classes[0]] = 0
    dataset.targets[dataset.targets==classes[1]] = 1
    dataset.targets= dataset.targets.tolist()  
    dataset.data = dataset.data[idx]
    return dataset

def prepare_dataloaders(batch_size, classes=[3, 7]):
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    # Test transform - only normalize
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    try:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=test_transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating data directory...")
        os.makedirs('./data', exist_ok=True)
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=test_transform)

    # Select two classes
    trainset = select_two_classes_from_cifar10(trainset, classes=classes)
    testset = select_two_classes_from_cifar10(testset, classes=classes)

    # Reduce dataset size
    train_size = 5000
    test_size = 1000
    trainset, _ = random_split(trainset, [train_size, len(trainset) - train_size])
    testset, _ = random_split(testset, [test_size, len(testset) - test_size])

    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=batch_size,
                           shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, testloader, trainset, testset


def main(image_size=(32,32), patch_size=(4,4), channels=3, 
         embed_dim=128, num_heads=4, num_layers=4, num_classes=2,
         pos_enc='learnable', pool='cls', dropout=0.3, fc_dim=None, 
         num_epochs=20, batch_size=16, lr=1e-4, warmup_steps=625,
         weight_decay=1e-3, gradient_clipping=1
    ):
    """Returns the best validation accuracy and model state"""
    
    loss_function = nn.CrossEntropyLoss()
    trainloader, testloader, _, _ = prepare_dataloaders(batch_size=batch_size)

    model = ViT(image_size=image_size, patch_size=patch_size, channels=channels, 
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim, 
                num_classes=num_classes)
    
    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))

    best_val_loss = 1e10
    best_accuracy = 0.0
    best_model_state = None

    # training loop
    for e in range(num_epochs):
        print(f'\n epoch {e}')
        model.train()
        train_loss = 0
        progress_bar = tqdm(trainloader, desc=f'Epoch {e+1}/{num_epochs}')
        for image, label in progress_bar:
            if torch.cuda.is_available():
                image, label = image.to('cuda'), label.to('cuda')
            opt.zero_grad()
            out = model(image)
            loss = loss_function(out, label)
            loss.backward()
            train_loss += loss.item()
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            sch.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.3f}'})

        train_loss/=len(trainloader)

        val_loss = 0
        with torch.no_grad():
            model.eval()
            tot, cor= 0.0, 0.0
            for image, label in testloader:
                if torch.cuda.is_available():
                    image, label = image.to('cuda'), label.to('cuda')
                out = model(image)
                loss = loss_function(out, label)
                val_loss += loss.item()
                out = out.argmax(dim=1)
                tot += float(image.size(0))
                cor += float((label == out).sum().item())
            acc = cor / tot
            val_loss /= len(testloader)
            print(f'-- train loss {train_loss:.3f} -- validation accuracy {acc:.3f} -- validation loss: {val_loss:.3f}')
            if val_loss <= best_val_loss:
                torch.save(model.state_dict(), 'model.pth')
                best_val_loss = val_loss
                best_accuracy = acc
                best_model_state = model.state_dict()

    # Load best model state
    model.load_state_dict(torch.load('model.pth'))
    return best_accuracy, model, testloader

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    
    # Expanded hyperparameter grid
    grid = {
        'embed_dim': [128, 256],
        'num_heads': [4, 8],
        'num_layers': [4, 6],
        'dropout': [0.1, 0.3],
        'pool': ['cls', 'mean'],
        'lr': [1e-4, 3e-4]
    }
    
    # Store results
    results = []
    best_overall_acc = 0.0
    best_overall_model = None
    best_test_iter = None

    # Modify main function call to use more epochs but not too many
    def quick_main(**kwargs):
        return main(**kwargs, num_epochs=5)  # Increased to 5 epochs
    
    total_combinations = len(list(itertools.product(
        grid['embed_dim'], grid['num_heads'], grid['num_layers'],
        grid['dropout'], grid['pool'], grid['lr']
    )))
    print(f"\nTotal configurations to test: {total_combinations}")
    
    # Grid search with progress tracking
    for idx, (embed_dim, num_heads, num_layers, dropout, pool, lr) in enumerate(itertools.product(
            grid['embed_dim'], grid['num_heads'], grid['num_layers'],
            grid['dropout'], grid['pool'], grid['lr']
    ), 1):
        
        # Skip invalid combinations
        if embed_dim % num_heads != 0:
            continue
            
        print(f"\nConfiguration {idx}/{total_combinations}:")
        print(f"embed_dim={embed_dim}, num_heads={num_heads}, "
              f"num_layers={num_layers}, dropout={dropout}, pool={pool}, lr={lr}")
        
        set_seed(seed=1)
        
        # Train model with this configuration
        acc, model, test_iter = quick_main(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            pool=pool,
            lr=lr
        )
        
        # Store results
        result = {
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dropout': dropout,
            'pool': pool,
            'lr': lr,
            'accuracy': acc
        }
        results.append(result)
        
        # Track best model
        if acc > best_overall_acc:
            best_overall_acc = acc
            best_overall_model = model
            best_test_iter = test_iter
            print(f"\nNew best accuracy: {acc:.3f}")

    # Analyze results
    print("\nAnalyzing results...")
    
    # Plot hyperparameter impact
    plot_hyperparameter_impact(results)
    
    # Get a sample batch for attention visualization
    sample_batch = next(iter(best_test_iter))
    sample_images, _ = sample_batch
    sample_image = sample_images[0]
    
    # Visualize attention maps for best model
    visualize_attention(best_overall_model, sample_image, save_path='attention_maps.png')
    
    # Print summary statistics
    accuracies = [r['accuracy'] for r in results]
    print("\nSummary Statistics:")
    print(f"Mean accuracy: {np.mean(accuracies):.3f}")
    print(f"Std accuracy: {np.std(accuracies):.3f}")
    print(f"Min accuracy: {np.min(accuracies):.3f}")
    print(f"Max accuracy: {np.max(accuracies):.3f}")
    
    # Print best configuration
    best_result = max(results, key=lambda x: x['accuracy'])
    print("\nBest Configuration:")
    for k, v in best_result.items():
        print(f"{k}: {v}")
