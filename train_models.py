import torch
from torchvision import models
import os

# In-Distribution Datasets
import torchvision.datasets as datasets
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt

# use cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import transforms
from torchvision import transforms
# random image using numpy
import numpy as np
import random

num_classes = 10
epochs = 15
batch_size = 64
learning_rate = 0.001

# Create transformations for each dataset

cifar_train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
])

cifar_test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# resnet requires 3 channels
mnist_train_transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.RandomCrop(28, 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.1307],
    #                                  std=[0.3081]),
])

mnist_test_transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.1307],
    #                      std=[0.3081])
])

svhn_train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
])

svhn_test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the in-distribution datasets in train-test-validation splits
cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_train_transform)
cifar10_test, cifar10_validation = torch.utils.data.random_split(
    datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_test_transform), [5000, 5000])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_train_transform)
mnist_test, mnist_validation = torch.utils.data.random_split(
    datasets.MNIST(root='./data', train=False, download=True, transform=mnist_test_transform), [5000, 5000])

svhn_train = datasets.SVHN(root='./data', split='train', download=True, transform=svhn_train_transform)
svhn_test, svhn_validation = torch.utils.data.random_split(
    datasets.SVHN(root='./data', split='test', download=True, transform=svhn_test_transform), [13016, 13016])

# define trainloaders
cifar10_trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
mnist_trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
svhn_trainloader = torch.utils.data.DataLoader(svhn_train, batch_size=batch_size, shuffle=True)

# define testloaders
cifar10_testloader = torch.utils.data.DataLoader(cifar10_test, batch_size=1000, shuffle=False)
mnist_testloader = torch.utils.data.DataLoader(mnist_test, batch_size=1000, shuffle=False)
svhn_testloader = torch.utils.data.DataLoader(svhn_test, batch_size=1000, shuffle=False)

# define validationloaders
cifar10_validationloader = torch.utils.data.DataLoader(cifar10_validation, batch_size=1000, shuffle=False)
mnist_validationloader = torch.utils.data.DataLoader(mnist_validation, batch_size=1000, shuffle=False)
svhn_validationloader = torch.utils.data.DataLoader(svhn_validation, batch_size=1000, shuffle=False)


def load_model(model_name, num_classes):
    if model_name == 'resnet18':
        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True

    else:
        model = models.vgg16()
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
        model.classifier[-1].weight.requires_grad = True
        model.classifier[-1].bias.requires_grad = True


    # # unfreeze all layers
    # for param in model.parameters():
    #     param.requires_grad = True

    return model.to(device)


cifar10_model = load_model('resnet18', 10)
mnist_model = load_model('vgg16', 10)
svhn_model = load_model('vgg16', 10)


# Define training function
def train_model(model, train_loader, validation_loader, epochs=10, lr=0.001):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_accuracy = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        # Evaluate on validation data using sklearn
        y_true = []
        y_pred = []

        # Evaluation on validation data and save the best model
        model.eval()
        for i, (inputs, labels) in enumerate(validation_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())
        current_accuracy = metrics.accuracy_score(y_true, y_pred)
        # Save the model if it is better than the previous one
        if current_accuracy > best_accuracy:
            best_accuracy = metrics.accuracy_score(y_true, y_pred)
            torch.save(model, "best_model.pt")
            print("Saved model with accuracy: ", best_accuracy)

        print("Validation Accuracy: ", metrics.accuracy_score(y_true, y_pred))
        print("Validation recall: ", metrics.recall_score(y_true, y_pred, average='macro'))
        print("Validation precision: ", metrics.precision_score(y_true, y_pred, average='macro'))

    print('Finished Training', model)
    return model


# Train the models if not already trained
# if not os.path.exists("cifar10_model_resnet.pt"):
#     cifar10_model = train_model(cifar10_model, cifar10_trainloader, cifar10_validationloader, epochs=epochs)
#     torch.save(cifar10_model, "cifar10_model_resnet.pt")
if not os.path.exists("mnist_model_resnet.pt"):
    mnist_model = train_model(mnist_model, mnist_trainloader, mnist_validationloader, epochs=epochs)
    torch.save(mnist_model, "mnist_model_resnet.pt")
if not os.path.exists("svhn_model_resnet.pt"):
    svhn_model = train_model(svhn_model, svhn_trainloader, svhn_validationloader, epochs=epochs)
    torch.save(svhn_model, "svhn_model_resnet.pt")
if not os.path.exists("cifar10_model_vgg.pt"):
    cifar10_model = train_model(cifar10_model, cifar10_trainloader, cifar10_validationloader, epochs=epochs)
    torch.save(cifar10_model, "cifar10_model_vgg.pt")
if not os.path.exists("mnist_model_vgg.pt"):
    mnist_model = train_model(mnist_model, mnist_trainloader, mnist_validationloader, epochs=epochs)
    torch.save(mnist_model, "mnist_model_vgg.pt")
if not os.path.exists("svhn_model_vgg.pt"):
    svhn_model = train_model(svhn_model, svhn_trainloader, svhn_validationloader, epochs=epochs)
    torch.save(svhn_model, "svhn_model_vgg.pt")
