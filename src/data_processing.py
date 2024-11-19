import os.path
import pickle

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import models
from torchvision.models import ResNet18_Weights
from gaussian_naive_bayes import GaussianNaiveBayes
from decision_tree import DecisionTree
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Step 1: Load CIFAR-10 Data
def load_cifar10():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    training_images = torchvision.datasets.CIFAR10('./data', train=True, transform=transform, download=True)
    validation_images = torchvision.datasets.CIFAR10('./data', train=False, transform=transform, download=True)
    return training_images, validation_images


def get_subset_of_data(dataset, num_per_class):
    class_counter = {i: 0 for i in range(10)}
    indices = []
    for index, (image, label) in enumerate(dataset):
        if class_counter[label] < num_per_class:
            indices.append(index)
            class_counter[label] += 1
        if all(count == num_per_class for count in class_counter.values()):
            break
    return indices

def extract_features(data_loader):
    features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            extracted_feature = feature_extractor(images)
            extracted_feature = extracted_feature.view(extracted_feature.size(0), -1)
            features.append(extracted_feature)
            all_labels.extend(labels)
    return torch.cat(features).numpy(), np.array(all_labels)


training_images, validation_images = load_cifar10()
training_indices = get_subset_of_data(training_images, 500)
validation_indices = get_subset_of_data(validation_images, 100)

training_subset = Subset(training_images, training_indices)
validation_subset = Subset(validation_images, validation_indices)

training_data_loader = DataLoader(training_subset, batch_size=64, shuffle=True)
validation_data_loader = DataLoader(validation_subset, batch_size=64, shuffle=False)

feature_extractor = models.resnet18(weights=ResNet18_Weights.DEFAULT)
feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
feature_extractor.eval()

training_features, training_labels = extract_features(training_data_loader)
validation_features, validation_labels = extract_features(validation_data_loader)

pca = PCA(50)
train_features_pca = pca.fit_transform(training_features)
test_features_pca = pca.transform(validation_features)

def save_model(model, file_name):
    with open(file_name, 'wb') as saved_model_file:
        pickle.dump(model, saved_model_file)

def load_model(file_name):
    with open(file_name, 'rb') as load_model_file:
        return pickle.load(load_model_file)


def evaluate(predictions, model, y_true):
    accuracy = accuracy_score(y_true, predictions)
    conf_matrix = confusion_matrix(y_true, predictions)
    report = classification_report(y_true, predictions)

    print(f"Accuracy ({model}):", accuracy)
    print(f"Confusion Matrix ({model}):\n", conf_matrix)
    print(f"Classification Report ({model}):\n", report)



# Manual Gaussian Naive Bayes
print("Manual Gaussian Naive Bayes Results:")
manual_bayes = GaussianNaiveBayes()
manual_bayes.fit(train_features_pca, training_labels)
evaluate(manual_bayes.predict(test_features_pca), "Manual Bayesian", validation_labels)

# Scikit-Learn Gaussian Naive Bayes
print("\nScikit-Learn Gaussian Naive Bayes Results:")
sklearn_gnb = GaussianNB()
sklearn_gnb.fit(train_features_pca, training_labels)
evaluate(sklearn_gnb.predict(test_features_pca), "Scikit-Learn", validation_labels)

# Manual Decision Tree
decision_tree = DecisionTree()
decision_tree_trained_model_path = "./models/decision_tree.pkl"
if os.path.exists(decision_tree_trained_model_path):
    print("Loading saved model of decision tree\n")
    decision_tree = load_model(decision_tree_trained_model_path)
else:
    print("Training and saving the model of decision tree\n")
    decision_tree.fit(train_features_pca, training_labels)
    save_model(decision_tree, decision_tree_trained_model_path)
print("Manual Decision Tree Results:")
evaluate(decision_tree.predict(test_features_pca), "Manual decision tree", validation_labels)

# Scikit-Learn Decision Tree
print("\nScikit-Learn decision tree Results:")
sklearn_decision_tree = DecisionTreeClassifier()
sklearn_decision_tree.fit(train_features_pca, training_labels)
evaluate(sklearn_decision_tree.predict(test_features_pca), "Scikit-Learn", validation_labels)