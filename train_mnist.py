from stack import Stack, FFN

import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
import lightning as L

class MNISTClassifier(L.LightningModule):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        # self.model = Stack(784, 1024, 10, dropout=0.1, activation=F.silu)
        self.model = FFN(784, 1024, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        return self.model(x)
    
    def shared_step(self, batch, mode='train'):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log(f'{mode}_loss', loss, prog_bar=True)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean()
        self.log(f'{mode}_acc', acc, prog_bar=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch)
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.001) # default weight decay is 0

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    model = MNISTClassifier()
    trainer = L.Trainer(
        max_epochs=20,        
    )
    trainer.fit(model, trainloader, testloader)
    print("Model training finished.")