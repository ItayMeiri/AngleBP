import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.colors import LogNorm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.utils.data import TensorDataset
from torchvision import datasets, models, transforms
# import functional from torch
import torch.nn.functional as F
import faiss
import matplotlib.pyplot as plt

import copy
import torch
from torch.autograd import Variable
from torch.nn import functional as F



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    if use_pretrained:
        weights = "IMAGENET1K_V1"
    else:
        weights = None
    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg":
        """ VGG16
        """
        model_ft = models.vgg11_bn(weights=weights)
        # model_ft = models.vgg11_bn(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)


    else:
        print("Invalid model name, exiting...")
        exit()

    # Build a wrapper around the model
    return model_ft


def get_datasets(name, data_transformation=None):
    if name == 'cifar10':
        dataset = datasets.CIFAR10(root='data/', train=True, transform=data_transformation['train'], download=True)
        dataset_test = datasets.CIFAR10(root='data/', train=False, transform=data_transformation['val'], download=True)
    elif name == 'svhn':
        dataset = datasets.SVHN(root='data/', split='train', transform=data_transformation['train'], download=True)
        dataset_test = datasets.SVHN(root='data/', split='test', transform=data_transformation['val'], download=True)
    elif name == "mnist":
        dataset = datasets.MNIST(root='data/', train=True, transform=data_transformation['train'], download=True)
        dataset_test = datasets.MNIST(root='data/', train=False, transform=data_transformation['val'], download=True)
    elif name == 'kmnist':
        dataset = datasets.KMNIST(root='data/', train=True, transform=data_transformation['train'], download=True)
        dataset_test = datasets.KMNIST(root='data/', train=False, transform=data_transformation['val'], download=True)
    elif name == 'fashionmnist':
        dataset = datasets.FashionMNIST(root='data/', train=True, transform=data_transformation['train'], download=True)
        dataset_test = datasets.FashionMNIST(root='data/', train=False, transform=data_transformation['val'], download=True)
    elif name == 'cifar100':
        dataset = datasets.CIFAR100(root='data/', train=True, transform=data_transformation['train'], download=True)
        dataset_test = datasets.CIFAR100(root='data/', train=False, transform=data_transformation['val'], download=True)
    elif name == "lsun":
        dataset = datasets.LSUN(root='data/', classes=['bedroom_train'], transform=data_transformation['train'], download=True)
        dataset_test = datasets.LSUN(root='data/', classes=['bedroom_val'], transform=data_transformation['val'])
    else:
        raise ValueError("Dataset not supported")
    return dataset, dataset_test

def get_dataloaders(name, batch_size=32, data_transformation=None):
    dataset, dataset_test = get_datasets(name, data_transformation)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return {"train": train_loader, "val": test_loader}


def get_transformations(name):
    if name == 'mnist' or name== 'fashionmnist' or name == 'kmnist':
        train_transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(32),
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.1307],
            #                      std=[0.3081])
        ])
    elif name == 'cifar10' or name == 'cifar100' or name == 'svhn':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif name == "lsun":
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError("Dataset not supported")
    return {'train': train_transform, 'val': test_transform}


# returns accuracy, recall, precision, confusion matrix
def test_model(model, dataloaders, save_name="cifar10_restnet", device='cpu'):
    # load model weights
    save_name = save_name + ".pth"
    print("Loading model weights from {}".format(save_name))
    model.load_state_dict(torch.load(save_name))
    model.eval()

    testing_dataloader = dataloaders['val']
    ground_truth = []
    predictions = []
    for i, (inputs, labels) in enumerate(testing_dataloader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        ground_truth.extend(labels.cpu().numpy())

    # accuracy, recall, precision, confusion matrix
    return accuracy_score(ground_truth, predictions), \
           recall_score(ground_truth, predictions, average='macro'), \
           precision_score(ground_truth, predictions, average='macro'), \
           confusion_matrix(ground_truth, predictions)


def get_classifier(model_name, model):
    if model_name == "resnet":
        return model.fc

    elif model_name == "vgg":
        return model.classifier[6]

    elif model_name == "densenet":
        return model.classifier

    else:
        print("Invalid model name, exiting...")
        exit()


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, save_name="cifar10_resnet", device='cpu',
                scheduler=None):
    # define early stopping pytorch

    val_acc_history = []
    counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_size = len(dataloaders['train'].dataset)
    val_size = len(dataloaders['val'].dataset)

    for phase in ['train', 'val']:
        dataloaders[phase] = [(inputs.to(device), labels.to(device)) for inputs, labels in dataloaders[phase]]

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                counter += 1
            else:
                model.eval()

            running_loss = 0.0
            running_accuracy = 0

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                predictions = torch.argmax(outputs, dim=1)
                running_loss += loss.item() * inputs.size(0)
                running_accuracy += torch.sum(predictions == labels.data)

                # print every 100 mini-batches
                if i % 100 == 99:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

            if phase == 'train':
                epoch_loss = running_loss / train_size
                epoch_acc = running_accuracy / train_size
            else:
                epoch_loss = running_loss / val_size
                epoch_acc = running_accuracy / val_size

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print("cooling counter: {}".format(counter))

            if phase == 'val':
                if scheduler is not None:
                    scheduler.step(epoch_loss)
                val_acc_history.append(epoch_acc)
                if epoch_acc > best_acc:
                    counter = 0
                    print("Saving model with accuracy: {}".format(epoch_acc))
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), save_name + ".pth")

            if counter == 12:
                print("Early stopping")
                model.load_state_dict(best_model_wts)
                return model, val_acc_history

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# # returns a batch of perturbed images
# def odin_singles(model, inputs, eps=0.04, T=1000, device='cpu'):
#     perturbed_inputs = []
#     for i, input in enumerate(inputs):
#         input = input.to(device)
#         input.requires_grad = True
#         output = model(input)
#         loss = F.cross_entropy(output, torch.tensor([i]).to(device))
#         loss.backward()
#         input = input + eps * torch.sign(input.grad)
#         input = input.detach()
#         input.requires_grad = False
#         perturbed_inputs.append(input)
#     return torch.stack(perturbed_inputs)


def odin_prepreoccessing(model, x, criterion, temperature,eps=0.04 ,y=None, norm_std=None):
    # does not work in inference mode, this sometimes collides with pytorch-lightning
    if torch.is_inference_mode_enabled():
        print("ODIN not compatible with inference mode. Will be deactivated.")

    # we make this assignment here, because adding the default to the constructor messes with sphinx
    if criterion is None:
        criterion = F.nll_loss

    with torch.inference_mode(False):
        if torch.is_inference(x):
            x = x.clone()

        with torch.enable_grad():
            x = Variable(x, requires_grad=True)
            logits = model(x) / temperature
            if y is None:
                y = logits.max(dim=1).indices
            loss = criterion(logits, y)
            loss.backward()

            gradient = torch.sign(x.grad.data)

            # if norm_std:
            #     for i, std in enumerate(norm_std):
            #         gradient.index_copy_(
            #             1,
            #             torch.LongTensor([i]).to(gradient.device),
            #             gradient.index_select(1, torch.LongTensor([i]).to(gradient.device)) / std,
            #         )

            x_hat = x - eps * gradient

    return x_hat



# Maha - Mahalanobis distance
# https://arxiv.org/pdf/1703.06857.pdf

def maha(inputs, model, num_classes=10):
    """
    :param x: input image
    :param model: trained model
    :param num_classes: number of classes
    :return: mahalanobis distance
    """


    for x in inputs:
            with torch.no_grad():
                out = model(x)
                loss = torch.nn.CrossEntropyLoss()(out, torch.tensor([num_classes]))
                loss.backward()
                x_grad = x.grad.data
                x_grad = x_grad.sign()
                x_adversarial = x + 0.001 * x_grad
                out = model(x_adversarial)
                logit = out
    return logit


# Shadow backpropagation - Calculates the first backpropagation step without updating the weights
# with the model prediction as the ground truth

def get_predictions(model, data_loader):
    _, ground_truths = torch.max(model(data_loader.dataset.tensors[0]), axis=1)
    return ground_truths


# Data loader should already have "ground_truths"
def shadow_backprop(data_loader, model, loss, layer):
    # Freeze the model except for the last layer
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last layer
    layer.weight.requires_grad = True
    layer.bias.requires_grad = True

    # Calculate the gradients

    gradients = torch.zeros(len(data_loader.dataset))
    for i, data in enumerate(data_loader):
        input, label = data
        model.zero_grad()
        output = model(input)
        loss = loss(output, label)
        # calculate the backpropagation step of only the last layer
        loss.backward(retain_graph=True)
        # get the gradient
        gradients[i] = layer.weight.grad + layer.bias.grad
    return gradients


# Creates a bank of vectors for the nearest neighbor search
# Will be used for backpropagation vectors on one hand, and vector embeddings of the model on the other
def create_bank(vectors, labels):
    dim = vectors.shape[1]  # dimension of vectors
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    index.add_with_ids(vectors, labels)
    return index


def get_nearest_neighbors(index, vectors, k=1):
    distances, labels = index.search(vectors, k)
    nearest_neighbor_ids = labels[0]
    return distances, nearest_neighbor_ids


def display_images(images, reshape=False):
    images = np.array(images)
    if reshape:
        images = images.reshape(-1, images.shape[2], images.shape[3])
    plt.figure(figsize=(images.shape[0] * 2, 2))
    for i, image in enumerate(images):
        plt.subplot(1, images.shape[0], i + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.show()


def plot_images_with_heatmaps2(in_images, out_images, in_heatmaps, out_heatmaps, alpha=0.5):
        fig, ax = plt.subplots(nrows=6, ncols=len(in_images), figsize=(16, 6))
        fig.subplots_adjust(hspace=0.3, wspace=0.1)

        for i, (image, heatmap) in enumerate(zip(in_images, in_heatmaps)):
            minimum, maximum = np.min(heatmap) + 1e-10, np.max(heatmap) + 1e-10
            ax[0][i].set_title('ID', fontsize=16)


            ax[0][i].imshow(image)
            ax[0][i].axis('off')

            ax[1][i].imshow(heatmap, cmap='inferno', alpha=alpha, norm=LogNorm(vmin=minimum, vmax=maximum))
            ax[1][i].axis('off')

            ax[2][i].imshow(image)
            ax[2][i].imshow(heatmap, cmap='inferno', alpha=alpha, norm=LogNorm(vmin=minimum, vmax=maximum))
            ax[2][i].axis('off')

        for i, (image, heatmap) in enumerate(zip(out_images, out_heatmaps)):
            minimum, maximum = np.min(heatmap) + 1e-10, np.max(heatmap) + 1e-10
            # Set title for all images
            ax[3][i].set_title('OOD', fontsize=16)

            ax[3][i].imshow(image)
            ax[3][i].axis('off')

            ax[4][i].imshow(heatmap, cmap='inferno', alpha=alpha, norm=LogNorm(vmin=minimum, vmax=maximum))
            ax[4][i].axis('off')

            ax[5][i].imshow(image)
            ax[5][i].imshow(heatmap, cmap='inferno', alpha=alpha, norm=LogNorm(vmin=minimum, vmax=maximum))
            ax[5][i].axis('off')

        plt.show()


def plot_images_with_heatmaps(in_images, out_images, in_heatmaps, out_heatmaps,alpha=0.5, figsize=(15, 10)):


    fig, ax = plt.subplots(nrows=2, ncols=len(in_images), figsize=(16, 6))
    # fig, ax = plt.subplots(2, 3, figsize=figsize)
    fig.subplots_adjust(hspace=0.3, wspace=0.1)
    for i, (image, heatmap) in enumerate(zip(in_images, in_heatmaps)):
        minimum, maximum = np.min(heatmap) + 1e-10, np.max(heatmap) + 1e-10
        ax[0][i].imshow(image)
        ax[0][i].imshow(heatmap, cmap='inferno', alpha=alpha, norm=LogNorm(vmin=minimum, vmax=maximum))
        ax[0][i].axis('off')
        ax[0][i].set_title("In Distribution")

    for i, (image, heatmap) in enumerate(zip(out_images, out_heatmaps)):
        minimum, maximum = np.min(heatmap) + 1e-10, np.max(heatmap) + 1e-10

        ax[1][i].imshow(image)
        ax[1][i].imshow(heatmap, cmap='inferno', alpha=alpha, norm=LogNorm(vmin=minimum, vmax=maximum))
        ax[1][i].axis('off')
        ax[1][i].set_title("Out of Distribution")

    plt.show()


