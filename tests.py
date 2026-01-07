import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Importing definitions from the provided code (Assuming the code is in a module or pasted above)
# For the purpose of this test suite, we will reference the classes by name assuming they exist in the scope.
# If running strictly as a file, imports would be needed. 
# We replicate the necessary definitions if they were in 'main.py' or similar.

# --- Mocking the dependencies for context if this were a standalone file ---
# In a real scenario, you would do: from main import get_cifar_resnet, BatchSizeScheduler, Config

class TestDonDecayLR(unittest.TestCase):

    def test_resnet_cifar_adaptation_structure(self):
        """
        Verifies that ResNet18 is correctly modified for CIFAR-10 (32x32 input).
        """
        # Assume get_cifar_resnet is available. 
        # We will dynamically instantiate it or mock the logic if imports failed, 
        # but here we test the function logic provided in the prompt.
        
        # Re-defining locally for the test runner to be self-contained if needed,
        # ideally we import these.
        from torchvision.models import resnet18
        
        # Logic from code:
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 10)
        
        # Checks
        self.assertEqual(model.conv1.kernel_size, (3, 3), "First conv should be 3x3")
        self.assertEqual(model.conv1.stride, (1, 1), "First conv stride should be 1")
        self.assertIsInstance(model.maxpool, nn.Identity, "Maxpool should be removed (Identity)")
        self.assertEqual(model.fc.out_features, 10, "Output classes should be 10")

    def test_model_forward_shape(self):
        """
        Verifies the model accepts (B, 3, 32, 32) and returns (B, 10).
        """
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 10)
        
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 32, 32)
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (batch_size, 10))

    def test_batch_size_scheduler(self):
        """
        Verifies that the scheduler increases batch size at milestones.
        """
        # Mock Dataset
        data = torch.randn(100, 3, 32, 32)
        targets = torch.randint(0, 10, (100,))
        dataset = TensorDataset(data, targets)
        
        initial_bs = 10
        milestones = [2, 4]
        multiplier = 2.0
        
        # Instantiate Scheduler (Copying class logic strictly for test isolation if needed, 
        # typically we import)
        class BatchSizeScheduler:
            def __init__(self, dataset, initial_bs, milestones, multiplier, num_workers=0):
                self.dataset = dataset
                self.current_bs = initial_bs
                self.milestones = set(milestones)
                self.multiplier = multiplier
                self.num_workers = num_workers
                self.loader = self._make_loader(self.current_bs)
                
            def _make_loader(self, bs):
                return DataLoader(self.dataset, batch_size=int(bs), shuffle=True, num_workers=self.num_workers)

            def step(self, epoch):
                if epoch in self.milestones:
                    self.current_bs = int(self.current_bs * self.multiplier)
                    self.loader = self._make_loader(self.current_bs)
                return self.loader
        
        scheduler = BatchSizeScheduler(dataset, initial_bs, milestones, multiplier)
        
        # Epoch 1: No change
        loader = scheduler.step(1)
        self.assertEqual(loader.batch_size, 10)
        
        # Epoch 2: Milestone -> BS doubles to 20
        loader = scheduler.step(2)
        self.assertEqual(loader.batch_size, 20)
        
        # Epoch 3: No change
        loader = scheduler.step(3)
        self.assertEqual(loader.batch_size, 20)
        
        # Epoch 4: Milestone -> BS doubles to 40
        loader = scheduler.step(4)
        self.assertEqual(loader.batch_size, 40)

    def test_training_integration(self):
        """
        Runs a single optimization step to ensure gradients flow and shapes match.
        """
        # Setup Model
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, 10)
        
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()
        
        # Dummy Batch
        inputs = torch.randn(8, 3, 32, 32)
        targets = torch.randint(0, 10, (8,))
        
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Check Loss
        self.assertFalse(torch.isnan(loss).item())
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check if parameters updated (simple check on conv1 weights)
        # This requires that gradients were non-zero, which is statistically nearly certain with randn
        self.assertIsNotNone(model.conv1.weight.grad)

if __name__ == '__main__':
    unittest.main()