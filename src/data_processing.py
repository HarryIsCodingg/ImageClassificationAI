import os.path
import pickle
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchvision import models
from torchvision.models import ResNet18_Weights
from gaussian_naive_bayes import GaussianNaiveBayes
from decision_tree import DecisionTree
from mlp.mlp import MLP
from cnn.cnn import CNN
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from mlp.mlp_experiment1 import MLP_EXPERIMENT1
from mlp.mlp_experiment2 import MLP_EXPERIMENT2
from mlp.mlp_experiment3 import MLP_EXPERIMENT3
from mlp.mlp_experiment4 import MLP_EXPERIMENT4
from cnn.cnn_increased_depth import CNN_INCREASED_DEPTH
from cnn.cnn_kernel_2X2 import CNN_KERNEL_2X2
from cnn.cnn_kernel_5X5 import CNN_KERNEL_5X5
from cnn.cnn_kernel_7X7 import CNN_KERNEL_7X7
from cnn.cnn_reduced_depth import CNN_REDUCED_DEPTH


def load_cifar10():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    training_images = torchvision.datasets.CIFAR10('./data', train=True, transform=transform, download=True)
    validation_images = torchvision.datasets.CIFAR10('./data', train=False, transform=transform, download=True)
    return training_images, validation_images


#   Getting subset of data
def get_subset_of_data(dataset, num_per_class):
    class_counter = {i: 0 for i in range(10)}
    indices = []
    for index, (image, label) in tqdm(enumerate(dataset)):
        if class_counter[label] < num_per_class:
            indices.append(index)
            class_counter[label] += 1
        if all(count == num_per_class for count in class_counter.values()):
            break
    return indices


#   Extracting features for basic algorithms
def extract_features(data_loader, feature_extractor, device):
    features = []
    all_labels = []
    feature_extractor = feature_extractor.to(device)
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            extracted_feature = feature_extractor(images)
            extracted_feature = extracted_feature.view(extracted_feature.size(0), -1)
            features.append(extracted_feature)
            all_labels.extend(labels)
    return torch.cat(features).cpu().numpy(), np.array(all_labels)


#   Loading CIFAR 10 dataset and extracting training data and validation data
def load_data():

    print("Loading data...")
    training_images, validation_images = load_cifar10()
    training_indices = get_subset_of_data(training_images, 500)
    validation_indices = get_subset_of_data(validation_images, 100)

    print("Getting subset of training and validation data...")
    training_subset = Subset(training_images, training_indices)
    validation_subset = Subset(validation_images, validation_indices)

    training_data_loader = DataLoader(training_subset, batch_size=64, shuffle=True)
    validation_data_loader = DataLoader(validation_subset, batch_size=64, shuffle=False)

    return training_data_loader, validation_data_loader


# Extract Features using ResNet18 and remove last layer
def extract_pca_features(training_data_loader, validation_data_loader, device):
    feature_extractor = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    feature_extractor = feature_extractor.to(device)
    feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
    feature_extractor.eval()

    training_features, training_labels = extract_features(training_data_loader, feature_extractor, device)
    validation_features, validation_labels = extract_features(validation_data_loader, feature_extractor, device)

    pca = PCA(50)
    train_features_pca = pca.fit_transform(training_features)
    test_features_pca = pca.transform(validation_features)

    return train_features_pca, training_labels, test_features_pca, validation_labels


#   Save models in .pkl file
def save_model(model, file_name):
    with open(file_name, 'wb') as saved_model_file:
        pickle.dump(model, saved_model_file)


#   Load models from .pkl file
def load_model(file_name):
    with open(file_name, 'rb') as load_model_file:
        return pickle.load(load_model_file)


# Common evaluate function produces confusion matrix, report and calculates accuracy
def evaluate(predictions, model, y_true):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    accuracy = accuracy_score(y_true, predictions)
    conf_matrix = confusion_matrix(y_true, predictions)
    report = classification_report(y_true, predictions)

    print(f"Accuracy ({model}):", accuracy)

    print(f"\n\nClassification Report ({model}):\n", report)

    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
    display.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show(block=False)

    return accuracy


#   Training all Gaussian Naive Bayes
def train_gaussian_naive_bayes(train_features_pca, training_labels, test_features_pca, validation_labels, path="train_again_models/gaussian"):

    manual_bayes = GaussianNaiveBayes()
    if not os.path.exists(path):
        os.makedirs(path)
    print("Training the manual gaussian")
    manual_bayes.fit(train_features_pca, training_labels)
    print("Saving manual gaussian trained model\n")
    save_model(manual_bayes, f"{path}/manual.pkl")

    print("Manual Gaussian Naive Bayes Results:")
    evaluate(manual_bayes.predict(test_features_pca), "Manual Gaussian Naive Bayes", validation_labels)

    # For sklearn gaussian
    print("\nScikit-Learn Gaussian Naive Bayes Results:")
    sklearn_gnb = GaussianNB()
    sklearn_gnb.fit(train_features_pca, training_labels)

    print("Saving trained sklearn model\n")
    save_model(manual_bayes, f"{path}/sklearn.pkl")

    evaluate(sklearn_gnb.predict(test_features_pca), "Scikit-Learn", validation_labels)


#   Loading all gaussian models from default directory
def load_trained_gaussian(test_features_pca, validation_labels, path="./trained_models/gaussian"):

    print("Loading trained manual gaussian model")
    trained_model = load_model(f"{path}/manual.pkl")

    print("Manual gaussian model results:")
    evaluate(trained_model.predict(test_features_pca), "Manual Gaussian", validation_labels)

    print("Loading trained sklearn gaussian model")
    trained_model = load_model(f"{path}/sklearn.pkl")

    print("Sklearn gaussian model results:")
    evaluate(trained_model.predict(test_features_pca), "Sklearn Gaussian", validation_labels)


def train_sklearn_decision_tree(train_features_pca, training_labels, test_features_pca, validation_labels):
    print("\nScikit-Learn decision tree Results:")


#   Training Decision Tree models with varying depths
def train_all_decision_trees(train_features_pca, training_labels, test_features_pca, validation_labels, path="train_again_models/decision_tree"):
    depths = [5, 10, 20, 30, 50]
    accuracies = []

    if not os.path.exists(path):
        os.makedirs(path)

    for depth in depths:
        # Train Decision Tree
        decision_tree = DecisionTree(maximum_depth=depth)
        decision_tree.fit(train_features_pca, training_labels)

        # Save the trained model
        model_filename = f"manual_depth_{depth}.pkl"
        with open(os.path.join(path, model_filename), 'wb') as model_file:
            pickle.dump(decision_tree, model_file)

        # Evaluate the model
        accuracy = evaluate(decision_tree.predict(test_features_pca), f"Manual decision tree (Depth {depth})", validation_labels)
        accuracies.append(accuracy)
        print(f"Accuracy for depth {depth}: {accuracy:.4f}")

    print("\nScikit-Learn decision tree Results:")
    sklearn_decision_tree = DecisionTreeClassifier()
    sklearn_decision_tree.fit(train_features_pca, training_labels)
    save_model(sklearn_decision_tree, f"{path}/sklearn.pkl")
    evaluate(sklearn_decision_tree.predict(test_features_pca), "Scikit-Learn", validation_labels)

    # Accuracy vs depth graph
    plt.figure(figsize=(10, 6))
    plt.plot(depths, accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('Tree Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs Depth')
    plt.grid(True)
    plt.show()


#   Loading all trained decision trees from default folder
def load_all_trained_decision_trees(test_features_pca, validation_labels, path="trained_models/decision_trees"):

    sklearn_decision_tree = load_model(f'{path}/sklearn.pkl')
    print("loading sklearn decision tree")
    evaluate(sklearn_decision_tree.predict(test_features_pca), "Sklearn Decision Tree", validation_labels)

    default_decision_tree = load_model(f'{path}/manual_depth_50.pkl')
    print("loading default decision tree with depth 50")
    evaluate(default_decision_tree.predict(test_features_pca), "Default decision tree with depth 50", validation_labels)

    print("loading decision tree with depth 5")
    default_decision_tree = load_model(f'{path}/manual_depth_5.pkl')
    evaluate(default_decision_tree.predict(test_features_pca), "Decision tree with depth 5", validation_labels)

    print("loading decision tree with depth 10")
    default_decision_tree = load_model(f'{path}/manual_depth_10.pkl')
    evaluate(default_decision_tree.predict(test_features_pca), "Decision tree with depth 10", validation_labels)

    print("loading decision tree with depth 20")
    default_decision_tree = load_model(f'{path}/manual_depth_20.pkl')
    evaluate(default_decision_tree.predict(test_features_pca), "Decision tree with depth 20", validation_labels)

    print("loading decision tree with depth 30")
    default_decision_tree = load_model(f'{path}/manual_depth_30.pkl')
    evaluate(default_decision_tree.predict(test_features_pca), "Decision tree with depth 30", validation_labels)


# Evaluate trained mlp model
def evaluate_mlp_model(self, model_name, testing_features, testing_labels):

    # Feature and labels for the correct device
    testing_features = torch.tensor(testing_features, dtype=torch.float32).to(next(self.parameters()).device)
    testing_labels = torch.tensor(testing_labels, dtype=torch.long).to(next(self.parameters()).device)

    self.eval()
    with torch.no_grad():
        outputs = self.forward(testing_features)
        _, predictions = torch.max(outputs, 1)

    predictions = predictions.cpu().numpy()
    testing_labels = testing_labels.cpu().numpy()
    return evaluate(predictions, model_name, testing_labels)


# Implementation for evaluation of mlp
def train_evaluate_mlp(mlp_model, model_name, train_features_pca, training_labels, test_features_pca, validation_labels, path):

    # Converting features, labels to Tensors
    training_features_tensor = torch.tensor(train_features_pca, dtype=torch.float32)
    training_labels_tensor = torch.tensor(training_labels, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(training_features_tensor, training_labels_tensor)
    training_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    mlp_model.train_model(10, 0.01, training_data_loader)
    accuracy = evaluate_mlp_model(mlp_model, model_name, testing_features=test_features_pca, testing_labels=validation_labels)

    torch.save(mlp_model.state_dict(), path)

    return accuracy


# Training of default mlp models and variance also
def train_all_mlp_models(device, train_features_pca, training_labels, test_features_pca, validation_labels, path="train_again_models/mlp"):
    accuracies = []
    model_names = []

    if not os.path.exists(path):
        os.makedirs(path)

    print("\nTraining default mlp model:\n")
    default_mlp = MLP(device)
    accuracy = train_evaluate_mlp(default_mlp, "Default MLP", train_features_pca, training_labels, test_features_pca, validation_labels,
                                  f"{path}/default_mlp.pth")
    accuracies.append(accuracy)
    model_names.append("Default MLP")

    print("\nTraining experiment 1 mlp model:\n")
    mlp_experiment1 = MLP_EXPERIMENT1(device)
    accuracy = train_evaluate_mlp(mlp_experiment1, "MLP Experiment 1", train_features_pca, training_labels, test_features_pca, validation_labels,
                                  f"{path}/mlp_experiment1.pth")
    accuracies.append(accuracy)
    model_names.append("MLP Experiment 1")

    print("\nTraining experiment 2 mlp model:\n")
    mlp_experiment2 = MLP_EXPERIMENT2(device)
    accuracy = train_evaluate_mlp(mlp_experiment2, "MLP Experiment 2", train_features_pca, training_labels, test_features_pca, validation_labels,
                                  f"{path}/mlp_experiment2.pth")
    accuracies.append(accuracy)
    model_names.append("MLP Experiment 2")

    print("\nTraining experiment 3 mlp model:\n")
    mlp_experiment3 = MLP_EXPERIMENT3(device)
    accuracy = train_evaluate_mlp(mlp_experiment3, "MLP Experiment 3", train_features_pca, training_labels, test_features_pca, validation_labels,
                                  f"{path}/mlp_experiment3.pth")
    accuracies.append(accuracy)
    model_names.append("MLP Experiment 3")

    print("\nTraining experiment 4 mlp model:\n")
    mlp_experiment4 = MLP_EXPERIMENT4(device)
    accuracy = train_evaluate_mlp(mlp_experiment4, "MLP Experiment 4", train_features_pca, training_labels, test_features_pca, validation_labels,
                                  f"{path}/mlp_experiment4.pth")
    accuracies.append(accuracy)
    model_names.append("MLP Experiment 4")

    # Accuracies of all models
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies, color='skyblue')
    plt.xlabel('Model Name')
    plt.ylabel('Accuracy')
    plt.title('Comparison of MLP Model Accuracies')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def load_all_trained_mlp_models(device, test_features_pca, validation_labels, path="trained_models/mlp"):
    print("\nLoading and evaluating Default MLP:\n")
    default_mlp = MLP(device)
    default_mlp.load_state_dict(torch.load(f"{path}/default_mlp.pth", map_location=device, weights_only=True))
    default_mlp.eval()
    evaluate_mlp_model(default_mlp, "Default MLP", test_features_pca, validation_labels)

    print("\nLoading and evaluating MLP Experiment 1:\n")
    mlp_experiment1 = MLP_EXPERIMENT1(device)
    mlp_experiment1.load_state_dict(torch.load(f"{path}/mlp_experiment1.pth", map_location=device, weights_only=True))
    mlp_experiment1.eval()
    evaluate_mlp_model(mlp_experiment1, "MLP Experiment 1", test_features_pca, validation_labels)

    print("\nLoading and evaluating MLP Experiment 2:\n")
    mlp_experiment2 = MLP_EXPERIMENT2(device)
    mlp_experiment2.load_state_dict(torch.load(f"{path}/mlp_experiment2.pth", map_location=device, weights_only=True))
    mlp_experiment2.eval()
    evaluate_mlp_model(mlp_experiment2, "MLP Experiment 2", test_features_pca, validation_labels)

    print("\nLoading and evaluating MLP Experiment 3:\n")
    mlp_experiment3 = MLP_EXPERIMENT3(device)
    mlp_experiment3.load_state_dict(torch.load(f"{path}/mlp_experiment3.pth", map_location=device, weights_only=True))
    mlp_experiment3.eval()
    evaluate_mlp_model(mlp_experiment3, "MLP Experiment 3", test_features_pca, validation_labels)

    print("\nLoading and evaluating MLP Experiment 4:\n")
    mlp_experiment4 = MLP_EXPERIMENT4(device)
    mlp_experiment4.load_state_dict(torch.load(f"{path}/mlp_experiment4.pth", map_location=device, weights_only=True))
    mlp_experiment4.eval()
    evaluate_mlp_model(mlp_experiment4, "MLP Experiment 4", test_features_pca, validation_labels)


#   Evaluation for validation
def evaluate_cnn_model(cnn_model, model_name, validation_data_loader, device):
    cnn_model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch_features, batch_labels in validation_data_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = cnn_model(batch_features)
            _, predicted_labels = torch.max(outputs, 1)

            predictions.append(predicted_labels.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    return evaluate(predictions, model_name, labels)


#   Helper method to train cnn model
def train_evaluate_cnn(device, model_name, training_data_loader, validation_data_loader, epochs, learning_rate, path, cnn_model):
    cnn = cnn_model(device)

    cnn.train_model(epochs, learning_rate, training_data_loader)

    cnn.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch_features, batch_labels in validation_data_loader:

            # Move features and labels to selected device
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = cnn(batch_features)
            _, predicted_labels = torch.max(outputs, 1)

            predictions.append(predicted_labels.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    evaluate(predictions, model_name, labels)
    torch.save(cnn.state_dict(), path)


#   Training of all cnn models
def train_all_cnn_models(device, training_data_loader, validation_data_loader):

    base_cnn_directory = "train_again_models/cnn"

    # Train default CNN
    print("\nTraining default CNN model:")
    train_evaluate_cnn(device, "Default CNN", training_data_loader, validation_data_loader, epochs=10, learning_rate=0.01,
                       path=os.path.join(base_cnn_directory, "cnn.pth"), cnn_model=CNN)

    # Train CNN with increased depth
    print("\nTraining CNN with increased depth:")
    train_evaluate_cnn(device, "CNN Increased Depth", training_data_loader, validation_data_loader, epochs=10, learning_rate=0.01,
                       path=os.path.join(base_cnn_directory, "cnn_increased_depth.pth"), cnn_model=CNN_INCREASED_DEPTH)

    # Train CNN with reduced depth
    print("\nTraining CNN with reduced depth:")
    train_evaluate_cnn(device, "CNN reduced depth", training_data_loader, validation_data_loader, epochs=10, learning_rate=0.01,
                       path=os.path.join(base_cnn_directory, "cnn_reduced_depth.pth"), cnn_model=CNN_REDUCED_DEPTH)

    # Train CNN with 2x2 kernel
    print("\nTraining CNN with 2x2 kernel:")
    train_evaluate_cnn(device, "CNN 2X2 Kernel", training_data_loader, validation_data_loader, epochs=10, learning_rate=0.01,
                       path=os.path.join(base_cnn_directory, "cnn_kernel_2X2.pth"), cnn_model=CNN_KERNEL_2X2)

    # Train CNN with 5x5 kernel
    print("\nTraining CNN with 5x5 kernel:")
    train_evaluate_cnn(device, "CNN 5X5 Kernel", training_data_loader, validation_data_loader, epochs=10, learning_rate=0.01,
                       path=os.path.join(base_cnn_directory, "cnn_kernel_5X5.pth"), cnn_model=CNN_KERNEL_5X5)

    # Train CNN with 7x7 kernel
    print("\nTraining CNN with 7x7 kernel:")
    train_evaluate_cnn(device, "CNN 7X7 Kernel", training_data_loader, validation_data_loader, epochs=10, learning_rate=0.01,
                       path=os.path.join(base_cnn_directory, "cnn_kernel_7X7.pth"), cnn_model=CNN_KERNEL_7X7)

#   Load all trained cnn models
def load_all_trained_cnn_models(device, validation_data_loader, path="trained_models/cnn"):
    accuracies = []
    model_names = []

    print("\nLoading and evaluating Default CNN:\n")
    default_cnn = CNN(device)
    default_cnn.load_state_dict(torch.load(f"{path}/cnn.pth", map_location=device, weights_only=True))
    default_cnn.eval()
    accuracy = evaluate_cnn_model(default_cnn, "Default CNN", validation_data_loader, device)
    accuracies.append(accuracy)
    model_names.append("Default CNN")

    print("\nLoading and evaluating CNN Reduced Depth:\n")
    cnn_reduced_depth = CNN_REDUCED_DEPTH(device)
    cnn_reduced_depth.load_state_dict(torch.load(f"{path}/cnn_reduced_depth.pth", map_location=device, weights_only=True))
    cnn_reduced_depth.eval()
    accuracy = evaluate_cnn_model(cnn_reduced_depth, "CNN Reduced Depth", validation_data_loader, device)
    accuracies.append(accuracy)
    model_names.append("CNN Reduced Depth")

    print("\nLoading and evaluating CNN Increased Depth:\n")
    cnn_increased_depth = CNN_INCREASED_DEPTH(device)
    cnn_increased_depth.load_state_dict(torch.load(f"{path}/cnn_increased_depth.pth", map_location=device, weights_only=True))
    cnn_increased_depth.eval()
    accuracy = evaluate_cnn_model(cnn_increased_depth, "CNN Increased Depth", validation_data_loader, device)
    accuracies.append(accuracy)
    model_names.append("CNN Increased Depth")

    print("\nLoading and evaluating CNN Kernel 5X5:\n")
    cnn_kernel_5X5 = CNN_KERNEL_5X5(device)
    cnn_kernel_5X5.load_state_dict(torch.load(f"{path}/cnn_kernel_5X5.pth", map_location=device, weights_only=True))
    cnn_kernel_5X5.eval()
    accuracy = evaluate_cnn_model(cnn_kernel_5X5, "CNN Kernel 5X5", validation_data_loader, device)
    accuracies.append(accuracy)
    model_names.append("CNN Kernel 5X5")

    print("\nLoading and evaluating CNN Kernel 7X7:\n")
    cnn_kernel_7X7 = CNN_KERNEL_7X7(device)
    cnn_kernel_7X7.load_state_dict(torch.load(f"{path}/cnn_kernel_7X7.pth", map_location=device, weights_only=True))
    cnn_kernel_7X7.eval()
    accuracy = evaluate_cnn_model(cnn_kernel_7X7, "CNN Kernel 7X7", validation_data_loader, device)
    accuracies.append(accuracy)
    model_names.append("CNN Kernel 7X7")

    print("\nLoading and evaluating CNN Kernel 2X2:\n")
    cnn_kernel_2X2 = CNN_KERNEL_2X2(device)
    cnn_kernel_2X2.load_state_dict(torch.load(f"{path}/cnn_kernel_2X2.pth", map_location=device, weights_only=True))
    cnn_kernel_2X2.eval()
    accuracy = evaluate_cnn_model(cnn_kernel_2X2, "CNN Kernel 2X2", validation_data_loader, device)
    accuracies.append(accuracy)
    model_names.append("CNN Kernel 2X2")

    # Accuracies of all models
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies, color='skyblue')
    plt.xlabel('Model Name')
    plt.ylabel('Accuracy')
    plt.title('Comparison of CNN Model Accuracies')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show(block=False)


def load_models(model_choice, device, test_features_pca, validation_labels, validation_data_loader):
    if model_choice == "1":
        load_trained_gaussian(test_features_pca, validation_labels)
    if model_choice == "2":
        load_all_trained_decision_trees(test_features_pca, validation_labels)
    if model_choice == "3":
        load_all_trained_mlp_models(device, test_features_pca, validation_labels)
    if model_choice == "4":
        load_all_trained_cnn_models(device, validation_data_loader)


def train_models(model_choice, device, train_features_pca, training_labels, test_features_pca, validation_labels, training_data_loader, validation_data_loader):
    if model_choice == "1":
        train_gaussian_naive_bayes(train_features_pca, training_labels, test_features_pca, validation_labels)
    if model_choice == "2":
        train_all_decision_trees(train_features_pca, training_labels, test_features_pca, validation_labels)
    if model_choice == "3":
        train_all_mlp_models(device, train_features_pca, training_labels, test_features_pca, validation_labels)
    if model_choice == "4":
        train_all_cnn_models(device, training_data_loader, validation_data_loader)


def main():

    seed = 50
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_data_loader, validation_data_loader = load_data()
    train_features_pca, training_labels, test_features_pca, validation_labels = extract_pca_features(training_data_loader, validation_data_loader, device)
    
    # Input 1 to load the models and 2 to train the models
    while True:
        option = input("Enter 1 to load a model or 2 to train a model or 3 to exit: ").strip()
        if option == "1":
            model_choice = input("Enter 1 for Bayes, 2 for Decision Tree, 3 for MLP, 4 for CNN: ").strip()
            load_models(model_choice, device=device, test_features_pca=test_features_pca, validation_labels=validation_labels, validation_data_loader=validation_data_loader)
        elif option == "2":
            model_choice = input("Enter 1 for Bayes, 2 for Decision Tree, 3 for MLP, 4 for CNN: ").strip()
            train_models(model_choice, device, train_features_pca, training_labels, test_features_pca, validation_labels, training_data_loader, validation_data_loader)
        elif option == "3":
            exit(0)
        else:
            print("Invalid option. Please enter 1 or 2.")

if __name__ == "__main__":
    main()