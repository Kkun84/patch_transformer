import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from model.model import TransformerModel
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm


batch_size = 64
lr = 0.001
max_epoch = 30


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
writer = SummaryWriter()
log_dir = Path(writer.get_logdir())

for dir_name in ['src', 'model']:
    shutil.copytree(Path(dir_name), log_dir / dir_name)

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

model = TransformerModel().to(device)
summary(model, (2, *model.input_shape))

optimizer = optim.Adam(model.parameters(), lr=lr)

n_iter = 0
for epoch in range(max_epoch):
    model.train()
    for i, (images, labels) in tqdm(
        enumerate(train_dataloader),
        desc=f'Train {epoch}/{max_epoch-1}',
        total=len(train_dataloader),
    ):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = (outputs.max(1)[1] == labels).float().mean().item()

        writer.add_scalar('metrics/train_loss', loss.item(), n_iter)
        writer.add_scalar('metrics/train_accuracy', accuracy, n_iter)

        n_iter += 1

    model.eval()

    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in tqdm(
            test_dataloader,
            desc=f'Test {epoch}/{max_epoch-1}',
            total=len(test_dataloader),
        ):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss += F.cross_entropy(outputs, labels, reduction='sum').item()
            correct += (outputs.max(1)[1] == labels).sum().item()
            total += len(labels)

        loss /= total
        accuracy = correct / total

        writer.add_scalar('metrics/test_loss', loss, n_iter)
        writer.add_scalar('metrics/test_accuracy', accuracy, n_iter)

    torch.save(model.state_dict(), log_dir / f'epoch{epoch:05}.pt')

print('Done.')
