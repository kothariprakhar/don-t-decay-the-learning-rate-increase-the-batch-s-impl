import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import math

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
class Config:
    # Dataset settings
    DATASET = 'CIFAR10'
    NUM_CLASSES = 10
    
    # Training settings
    EPOCHS = 40
    INITIAL_LR = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    
    # The Core Paper Strategy Parameters
    INITIAL_BATCH_SIZE = 128
    # Instead of decaying LR by 10 at these epochs, we multiply BS by 10
    MILESTONES = [20, 30] 
    BS_MULTIPLIER = 5.0  # Paper suggests matching decay factor. Using 5x here to prevent OOM on typical Colab/Consumer GPUs for the demo. (Ideal is 10x)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. ResNet Adaptation for CIFAR-10
# ==========================================
# Standard ResNet18 is designed for ImageNet (224x224). 
# For CIFAR (32x32), we must modify the first layers to prevent excessive downsampling.
def get_cifar_resnet():
    model = torchvision.models.resnet18(weights=None)
    # Replace first 7x7 conv with 3x3 conv, stride 1, padding 1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove the first maxpool
    model.maxpool = nn.Identity()
    # Change fc layer for 10 classes
    model.fc = nn.Linear(model.fc.in_features, Config.NUM_CLASSES)
    return model.to(Config.DEVICE)

# ==========================================
# 3. The Innovated Batch Size Scheduler
# ==========================================
class BatchSizeScheduler:
    """
    Mimics the API of a Learning Rate Scheduler, but operates on the DataLoader.
    According to Smith et al., decaying LR is equivalent to increasing Batch Size.
    """
    def __init__(self, dataset, initial_bs, milestones, multiplier, num_workers=2):
        self.dataset = dataset
        self.current_bs = initial_bs
        self.milestones = set(milestones)
        self.multiplier = multiplier
        self.num_workers = num_workers
        self.loader = self._make_loader(self.current_bs)
        
    def _make_loader(self, bs):
        return DataLoader(
            self.dataset, 
            batch_size=int(bs), 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def step(self, epoch):
        """Checks if batch size needs updating at the start of an epoch."""
        if epoch in self.milestones:
            old_bs = self.current_bs
            self.current_bs = int(self.current_bs * self.multiplier)
            print(f"\n[Scheduler] Milestone reached at Epoch {epoch}.")
            print(f"[Scheduler] Increasing Batch Size: {old_bs} -> {self.current_bs}")
            print(f"[Scheduler] Learning Rate stays constant (Simulating Decay).")
            # Re-instantiate loader with new batch size
            self.loader = self._make_loader(self.current_bs)
        return self.loader

    def get_loader(self):
        return self.loader

# ==========================================
# 4. Data Preparation
# ==========================================
def prepare_data():
    print("Preparing CIFAR-10 Dataset...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download usually required once
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    return trainset, testloader

# ==========================================
# 5. Training Loop
# ==========================================
def train():
    trainset, testloader = prepare_data()
    
    # Initialize Model
    model = get_cifar_resnet()
    
    # Optimizer: Standard SGD. Note we do NOT use an LR Scheduler here.
    optimizer = optim.SGD(model.parameters(), lr=Config.INITIAL_LR, 
                          momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Custom Batch Size Scheduler
    bs_scheduler = BatchSizeScheduler(
        trainset, 
        Config.INITIAL_BATCH_SIZE, 
        Config.MILESTONES, 
        Config.BS_MULTIPLIER
    )

    print(f"\nStarting training on {Config.DEVICE} for {Config.EPOCHS} epochs.")
    print(f"Strategy: Keep LR constant ({Config.INITIAL_LR}), Increase Batch Size at {Config.MILESTONES}.")
    
    total_start_time = time.time()

    for epoch in range(Config.EPOCHS):
        epoch_start = time.time()
        
        # Update Loader based on Schedule
        trainloader = bs_scheduler.step(epoch)
        current_bs = trainloader.batch_size
        
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training Steps
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        avg_loss = running_loss / len(trainloader)
        
        # Validation Steps
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * test_correct / test_total
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | BS: {current_bs} | "
              f"Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {test_acc:.2f}% | Time: {epoch_time:.2f}s")

    total_time = time.time() - total_start_time
    print(f"\nTraining Complete. Total Time: {total_time/60:.2f} minutes.")
    print(f"Final Validation Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    # install requirements if needed: 
    # pip install torch torchvision
    train()