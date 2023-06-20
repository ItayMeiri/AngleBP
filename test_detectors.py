import utils
import torch
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score
import torch.nn.functional as F
from OOD.mad import MAD
from OOD.odin import ODIN
from OOD.backpropagation import GradBP
from OOD.approxgrad import ApproxGrad
import matplotlib.pyplot as plt
from itertools import product
import torchmetrics
from torchmetrics.classification import BinaryAUROC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from sklearn.decomposition import PCA

np.set_printoptions(suppress=True)

from sklearn import metrics


def plot_auroc(targets, scores, ax, name=""):
    fpr, tpr, thresholds = metrics.roc_curve(targets, scores)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name)
    display.plot(ax=ax)

    return roc_auc


def plotter(x, color=None, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        if color:
            plt.plot(x, color=color, marker='o', linestyle='None', markersize=1, alpha=0.5)
        plt.plot(x, marker='o', linestyle='None', markersize=1, alpha=0.5)
    else:
        if color:
            ax.plot(x, color=color, marker='o', linestyle='None', markersize=1, alpha=0.5)
        ax.plot(x, marker='o', linestyle='None', markersize=1, alpha=0.5)


def calculate_auroc(targets, scores, return_fpr_tpr=False):
    fpr, tpr, thresholds = metrics.roc_curve(targets, scores)
    roc_auc = metrics.auc(fpr, tpr)
    if return_fpr_tpr:
        return roc_auc, fpr, tpr, thresholds
    return roc_auc
# def calculate_auroc(predictions, labels):
#     # Predictions = logits
#     # target = labels
#
#     b_auroc = BinaryAUROC(thresholds=None)
#     auroc = b_auroc(torch.tensor(predictions), torch.tensor(labels))
#     return auroc

def print_statistics(model_path, method, labels, predictions, meta_data=None):
    # if meta_data:
    #     print("**********", model_path, method, meta_data)
    # else:
    #     print("**********", model_path, method)

    # print metrics excluding OOD - that is where labels
    acc = accuracy_score(labels[predictions != 11], predictions[predictions != 11])
    # TDR = np.sum(labels[labels == 11] == predictions[labels == 11]) / np.sum(labels == 11)
    # Number of times the model predicted OOD when it was in-distribution
    TDR = np.sum(predictions[labels == 11] == 11) / np.sum(labels == 11)
    FDR = np.sum(predictions[labels != 11] == 11) / np.sum(labels != 11)
    # FDR = np.sum(labels[labels != 11] != predictions[labels != 11]) / np.sum(labels != 11)

    # format accuracy, TDR, FDR as percentages to be copy-pasted into google sheets
    # acc = "{:.2%}".format(acc)
    # TDR = "{:.2%}".format(TDR)
    # FDR = "{:.2%}".format(FDR)

    # print(acc)
    # print(TDR)
    # print(FDR)
    # print("")
    #
    # print("Accuracy:", acc, ",TDR:", TDR, ",FDR:", FDR, ",TRR:", np.sum(predictions == 11) / len(predictions))
    # print("")
    #
    # # print accuracy, TDR, FDR in a table that can be copy-pasted into google sheets
    # print("Accuracy\tTDR\tFDR")
    # print(acc, "\t", TDR, "\t", FDR)
    # print("")
    # Turn ground truth labels into binary labels(11=1, everything else=0)
    labels[labels != 11] = 0
    labels[labels == 11] = 1

    predictions[predictions != 11] = 0
    predictions[predictions == 11] = 1

    auroc= 1

    return acc, TDR, FDR, auroc


def odin_testing(combined_dataloader, detector, model_path, method, T, eps):
    size = len(combined_dataloader)
    labels = np.zeros(size)
    predictions = np.zeros(size)
    confidences = np.zeros(size)

    for i, (x, y) in enumerate(combined_dataloader):
        x, y = x.to(device), y.to(device)
        confidence, prediction = detector.predict(x, return_prediction=True)
        labels[i] = y.item()
        predictions[i] = prediction.item()
        confidences[i] = confidence.item()

        if i % 10000 == 1:
            print(f"{i}/{len(combined_dataloader)}")

    return labels, predictions, confidences


methods = [
           "gradneighbors",]

model_data = [


    # ("svhn", "cifar100", "densenet", "91.60%", True, False),





    #
    # ("mnist", "kmnist", "vgg", "74.55%", False, True),
    # # ("mnist", "kmnist", "densenet", "68.63%", False, True),
    ("cifar10", "cifar100", "vgg", "86.47%", True, False),
    ("cifar10", "cifar100", "resnet", "81.25%", True, False),
    ("cifar10", "cifar100", "densenet", "85.77%", True, False),
    ("svhn", "cifar100", "vgg", "93.33%", True, False),
    ("svhn", "cifar100", "resnet", "91.32", True, False),
    ("svhn", "cifar100", "densenet", "93.17%", True, False),
    ("mnist", "kmnist", "resnet", "98.89%", False, False),
    ("mnist", "kmnist", "vgg", "98.97%", True, False),
    ("mnist", "kmnist", "densenet", "98.92%", True, False),
    # ("cifar10", "cifar100", "resnet", "74.86", False, False),
    # ("cifar10", "cifar100", "densenet", "77.99%", False, False),

]

def performance_func(y_preds, y_binary, data, weights):
    mags = data.copy()
    for i, weight in enumerate(weights):
        mags[y_preds == i] = mags[y_preds == i] * weight

    return calculate_auroc(y_binary, mags)

def binary_search(y_preds,y_binary, data, weights, performance_func, index):
    left = 0.0
    right = 1.0

    while right - left > 1e-6:
        mid = (left + right) / 2

        new_weights = weights.copy()
        new_weights[index] = mid

        if performance_func(y_preds,y_binary, data,new_weights) > performance_func(y_preds, y_binary, data, weights):
            left = mid
            weights[index] = left
        else:
            right = mid
            weights[index] = right


    return weights



# def binary_search(y_preds,y_binary, data, weights, performance_func, index):
#     left = 0.0
#     right = 1.0
#
#     while right - left > 1e-6:
#         mid = (left + right) / 2
#
#         new_weights = weights.copy()
#         new_weights[index] = mid
#
#         if performance_func(y_preds,y_binary, data,new_weights) > performance_func(y_preds, y_binary, data, weights):
#             left = mid
#
#         else:
#             right = mid



# def weighted_sum(angles, magnitudes, weight):
#     combined_scores = weight * angles +  magnitudes
#     return combined_scores
#
# def weighted_product(angles, magnitudes, weight):
#     combined_scores = weight * angles * magnitudes
#     return combined_scores
#
# def weighted_combined(angles, magnitudes, m_weight, a_weight):
#     combined_scores = m_weight * magnitudes + a_weight * angles
#     return combined_scores

def weighted_sum(features, weights):
    return np.sum(features * weights, axis=1)

def weighted_product(features, weights):
    return np.prod(features * weights, axis=1)

def weighted_sum_product(features, weights):
    try:
        return np.sum(features * weights, axis=1) * np.prod(features * weights, axis=1)
    except:
        return np.sum(features * weights, axis=1) * np.prod(features * weights, axis=1, keepdims=True)

# plot two bars
data = [13.026078939437866, 12.534195184707642]
# The labels are "dots + weight application" and "norm", the data is in microseconds(us)
labels = ["dots + weight application", "norm"]
plt.bar(labels, data)

def get_auroc(features, weights, y_binary, func, reverse=False, return_fpr_tpr=False):

    scores = func(features, weights)
    if reverse:
        scores = -scores
    auroc = calculate_auroc(y_binary, scores, return_fpr_tpr=return_fpr_tpr)
    return auroc


for in_dataset, out_dataset, model_name, model_type, use_pretrained, feature_extract in model_data:
    model_path = f"{model_name}_{in_dataset}_p{use_pretrained}_f{feature_extract}.pth"

    print("Testing:", model_name, in_dataset, out_dataset, model_type, model_path)

    in_transformation = utils.get_transformations(in_dataset)
    out_transformation = utils.get_transformations(out_dataset)
    # load model
    model = utils.initialize_model(model_name, num_classes=10, use_pretrained=use_pretrained,
                                   feature_extract=feature_extract)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    _, in_dist_test = utils.get_datasets(in_dataset, in_transformation, download=False)

    # Test accuracy on in_dist_test

    _, out_dist_test = utils.get_datasets(out_dataset, out_transformation, download=False)

    if len(out_dist_test) > 10000:
        # pick 10000 random samples
        out_dist_test = torch.utils.data.Subset(out_dist_test,
                                                np.random.choice(len(out_dist_test), 10000, replace=False))
    if len(in_dist_test) > 10000:
        # pick 10000 random samples
        in_dist_test = torch.utils.data.Subset(in_dist_test, np.random.choice(len(in_dist_test), 10000, replace=False))

    # change all test labels to 11
    # out_dist_test.targets = [11] * len(out_dist_test.targets)

    if out_dataset == "svhn":
        out_dist_test.dataset.labels = [11] * len(out_dist_test.dataset.labels)
    else:
        out_dist_test.targets = [11] * len(out_dist_test.targets)
    # Split in_dist_test into in_dist_test and in_dist_val
    # in_dist_test, in_dist_val = torch.utils.data.random_split(in_dist_test, [5000, 5000])

    # out_dist_test, _ = torch.utils.data.random_split(out_dist_test, [5000, 5000])

    # combine the two datasets
    combined_dataset = torch.utils.data.ConcatDataset([in_dist_test, out_dist_test])
    layer = utils.get_classifier(model_name, model)
    for method in methods:
        # print("Starting method:", method)
        if method == "gradneighbors":
            batch_size = 64
            combined_dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

            detector = ApproxGrad(model=model, criterion=F.cross_entropy, layer=layer)


            test_gradients = None
            y_preds = np.array([])
            y_true = np.array([])
            inputs_processed = 0
            import time
            for i, (x, y) in enumerate(combined_dataloader):
                x, y = x.to(device), y.detach().cpu().numpy()
                gradients, pred = detector.predict(x, return_predictions=True)

                # change to tensor and device
                if test_gradients is None:
                    test_gradients = gradients
                else:
                    test_gradients = np.concatenate((test_gradients, gradients))

                y_preds = np.concatenate((y_preds, pred))
                y_true = np.concatenate((y_true, y))
                # inputs_processed += batch_size

                # print number of batches per second
                # if time.time() - start > 15:
                #     inputs_per_second = inputs_processed / (time.time() - start)
                #     print("Inputs processed per second: {:.2f}".format(inputs_per_second))
                #     start = time.time()
                #     inputs_processed = 0

                # if time.time() - start > 15:
                #     print("Progress per second:", batch_size / (time.time() - start))
                #     start = time.time()
                # if i % 20 == 3:
                #     print(i)
            y = y_true.copy()
            y_binary = np.where(y == 11, 1, 0)

            flat_gradients = test_gradients.reshape((test_gradients.shape[0], -1))
            magnitudes = np.linalg.norm(flat_gradients, axis=1)
            unit_vector = np.ones(flat_gradients.shape[1])
            signed_gradients = np.sign(flat_gradients)
            # generate random vectors between -1 and 1
            dots = np.arctan(signed_gradients@unit_vector)
            cos_sim = torch.cosine_similarity(torch.tensor(signed_gradients), torch.tensor(unit_vector), dim=1).detach().cpu().numpy()
            cos_sim[np.isnan(cos_sim) | np.isinf(cos_sim)] = 1

            funcs = [weighted_product, weighted_sum_product]

            # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            feature1 = np.column_stack([magnitudes, cos_sim])
            feature2 = np.column_stack([magnitudes, dots])
            feature3 = np.column_stack([magnitudes, cos_sim, dots])
            weight_range = np.arange(-2, 2, 0.1)

            best_auroc, best_weights, best_func, best_feature = 0, None, None, None
            mag_auroc = calculate_auroc(y_binary, magnitudes)


            feature_name = ""
            start = time.time()
            for weights in product(weight_range, repeat=2):
                for func in funcs:
                    auroc = get_auroc(features=feature1, weights=weights, y_binary=y_binary, func=func)
                    if auroc > best_auroc:
                        best_auroc = auroc
                        best_weights = weights
                        best_func = func
                        best_feature = feature1
                        feature_name = "magnitudes, cos_sim"

                    auroc = get_auroc(features=feature2, weights=weights, y_binary=y_binary, func=func)
                    if auroc > best_auroc:
                        best_auroc = auroc
                        best_weights = weights
                        best_func = func
                        best_feature = feature2
                        feature_name = "magnitudes, dots"
                # if time.time() - start > 30:
                    start = time.time()
                    # Print progress
                    # print("current weights:", weights)
            for weights in product(weight_range, repeat=3):
                for func in funcs:
                    auroc = get_auroc(features=feature3, weights=weights, y_binary=y_binary, func=func)
                    if auroc > best_auroc:
                        best_auroc = auroc
                        best_weights = weights
                        best_func = func
                        best_feature = feature3
                        feature_name = "magnitudes, cos_sim, dots"
                # if time.time() - start > 90:
                #     start = time.time()
                    # Print progress
                    # print("current weights:", weights)
            if mag_auroc > best_auroc:
                print("Dots/Cos failed to beat magnitudes")
            print("Best weights, func, feature:", best_weights, best_func, feature_name)
            _, comb_fpr, comb_tpr, thresholds = get_auroc(features=best_feature, weights=best_weights, y_binary=y_binary, func=best_func, return_fpr_tpr=True)
            # Get best FPR when TPR=0.95
            # find closest number to 0.95
            closest = np.argmin(np.abs(comb_tpr - 0.95))
            best_fpr = comb_fpr[np.where(comb_tpr == comb_tpr[closest])][0]
            print("AngleBP FPR", best_fpr, "AUROC", best_auroc, "TPR", comb_tpr[closest])

            # Get best FPR when TPR=0.95
            mag_fpr, mag_tpr, _ = metrics.roc_curve(y_binary, magnitudes)
            closest = np.argmin(np.abs(mag_tpr - 0.95))
            best_mag_fpr = mag_fpr[np.where(mag_tpr == mag_tpr[closest])][0]
            roc_auc = metrics.auc(mag_fpr, mag_tpr)
            print("GradBP FPR", best_mag_fpr, "AUROC", roc_auc, "TPR", mag_tpr[closest])

            # largest_gap = 0
            #
            # m_fpr, m_tpr = 0, 0
            # c_fpr, c_tpr = 0, 0


            # for i in range(len(comb_fpr)):
            #     if comb_fpr[i] <= 0.3 and comb_tpr[i] > mag_tpr[i] and mag_fpr[i] >= comb_fpr[i] and comb_tpr[i] > 0.5:
            #         gap = comb_tpr[i] - mag_tpr[i]
            #         if gap > largest_gap:
            #             largest_gap = gap
            #             m_fpr, m_tpr = mag_fpr[i], mag_tpr[i]
            #             c_fpr, c_tpr = comb_fpr[i], comb_tpr[i]
            #
            #
            # print("Largest gap:", largest_gap)
            # print("Mag FPR:", m_fpr, "Mag TPR:", m_tpr)
            # print("Comb FPR:", c_fpr, "Comb TPR:", c_tpr)

        # print("ENDING METHOD", method)




