# Testing the models
import os
from pprint import pprint

import utils
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1000

use_pretrained = True
feature_extract = False

# for each pair of dataset and model, find the best performing model
best_model_dict = {}
for dataset in ['mnist', 'cifar10', 'svhn']:
    all_models_accuracy = []
    transformations = utils.get_transformations(dataset)
    for model_backbone in ['vgg', 'resnet', 'densenet']:
        best_model = None
        best_model_acc = [0]
        dataloaders = utils.get_dataloaders(dataset, batch_size=batch_size,
                                            data_transformation=transformations)

        for feature_extract in [True, False]:
            for use_pretrained in[True, False]:
                if best_model is None:
                    best_model = f"{model_backbone}_{dataset}_p{use_pretrained}_f{feature_extract}"
                else:
                    current_model = f"{model_backbone}_{dataset}_p{use_pretrained}_f{feature_extract}"
                    if current_model + ".pth" not in os.listdir():
                        print(f"Skipping {current_model} because it does not exist")
                        continue

                    # Compare the current model to the best model
                    model = utils.initialize_model(model_backbone, num_classes=10, use_pretrained=use_pretrained,
                                                   feature_extract=feature_extract)
                    model = model.to(device)

                    # print("testing", current_model)
                    current_model_acc = utils.test_model(model=model, dataloaders=dataloaders, save_name=current_model, device=device)
                    all_models_accuracy.append((current_model, current_model_acc[0]))
                    # print(current_model_acc[0])
                    # print("testing", best_model)
                    # best_model_acc = utils.test_model(model=model, dataloaders=dataloaders, save_name=best_model, device=device)
                    # print(best_model_acc[0])

                    if current_model_acc[0] > best_model_acc[0]:
                        best_model = current_model
                        best_model_acc = current_model_acc
                        # print("New best model:", best_model, "with", current_model_acc[0], "accuracy")

                print("*" * 50)
        # Sort all_models_accuracy by accuracy

    print("Dataset:", dataset)
    pprint(sorted(all_models_accuracy, key=lambda x: x[1], reverse=True))
    print("*" * 50)

        # print("Best model for", dataset, "is", best_model, "with", best_model_acc[0], "accuracy")
        # best_model_dict[f"{model_backbone}_{dataset}"] = best_model_acc
