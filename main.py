import random
import torch
import torchvision
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import (
    LinearNeuralNet, QuadraticNeuralNet,
    LinearDendriticNet, QuadraticDendriticNet
)

network = LinearDendriticNet
D = 28 * 28
K = 10
features = (D, 2048, 2048, K)
num_tasks = 10
num_epochs = 3
batch_size_train = 256
batch_size_test = 512
learning_rate = 5e-4


def prepare_dataset():
    train_loader, test_loader, prototype = {}, {}, Tensor()

    permute_idx = list(range(D))
    for task in range(num_tasks):
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        # dataset before permute
        train_set = torchvision.datasets.MNIST(
            root='', train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.MNIST(
            root='', train=False, download=True, transform=transform
        )

        # permute the dataset according to permute index
        train_set.data = train_set.data.reshape(-1, D)[:, permute_idx]
        test_set.data = test_set.data.reshape(-1, D)[:, permute_idx]

        # Compute the context vector of current task
        prototype = torch.cat((prototype, torch.mean(
            Tensor.float(train_set.data), dim=0, keepdim=True
        )), dim=0)

        # Data loader
        train_loader[task] = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=batch_size_train, shuffle=True
        )
        test_loader[task] = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=batch_size_test, shuffle=False
        )

        # shuffle the permutation index
        random.shuffle(permute_idx)
        # permute_idx = torch.randperm(D).tolist()

    return train_loader, test_loader, prototype


def test_all_task(model, test_loader, task_index):
    with torch.no_grad():
        for task in range(task_index):
            correct = 0
            total = 0
            for images, labels in test_loader[task]:
                images = images.reshape(-1, 28*28)
                outputs = model(images, test=True)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Task [{}/{}], Accuracy: {} %'
                  .format(task+1, task_index, 100 * correct / total))


def main():
    train_loader, test_loader, prototype = prepare_dataset()

    model = network(features, prototype=prototype)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # train
    for task in range(num_tasks):
        num_steps = len(train_loader[task])
        for epoch in range(num_epochs):
            for step, (images, labels) in enumerate(train_loader[task]):
                # Forward pass
                predict = model(images.reshape(-1, D))
                loss = criterion(predict, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step + 1) % int(0.2 * num_steps) != 0: continue
                print(
                    'Task [{}/{}], '.format(
                        task + 1, num_tasks
                    ) + 'Epoch [{}/{}], Step [{}/{}], '.format(
                        epoch + 1, num_epochs, step + 1, num_steps
                    ) + 'Loss: {:.4f}, '.format(
                        loss.item()
                    )
                )
                # print(model.num_prototype)

        test_all_task(model, test_loader, task+1)


if __name__ == "__main__":
    main()
