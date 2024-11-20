import os.path
import pickle

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
from mlp import MLP
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


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
    for index, (image, label) in tqdm(enumerate(dataset)):
        if class_counter[label] < num_per_class:
            indices.append(index)
            class_counter[label] += 1
        if all(count == num_per_class for count in class_counter.values()):
            break
    return indices


def extract_features(data_loader, feature_extractor):
    features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            extracted_feature = feature_extractor(images)
            extracted_feature = extracted_feature.view(extracted_feature.size(0), -1)
            features.append(extracted_feature)
            all_labels.extend(labels)
    return torch.cat(features).numpy(), np.array(all_labels)


def load_data():
    training_images, validation_images = load_cifar10()
    training_indices = get_subset_of_data(training_images, 500)
    validation_indices = get_subset_of_data(validation_images, 100)

    training_subset = Subset(training_images, training_indices)
    validation_subset = Subset(validation_images, validation_indices)

    training_data_loader = DataLoader(training_subset, batch_size=64, shuffle=True)
    validation_data_loader = DataLoader(validation_subset, batch_size=64, shuffle=False)

    return training_data_loader, validation_data_loader


# Extract Features using ResNet18
def extract_pca_features(training_data_loader, validation_data_loader):
    feature_extractor = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
    feature_extractor.eval()

    training_features, training_labels = extract_features(training_data_loader, feature_extractor)
    validation_features, validation_labels = extract_features(validation_data_loader, feature_extractor)

    pca = PCA(50)
    train_features_pca = pca.fit_transform(training_features)
    test_features_pca = pca.transform(validation_features)

    return train_features_pca, training_labels, test_features_pca, validation_labels


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


def evaluate_mlp_model(self, testing_features, testing_labels):

    # Feature and labels for the correct device
    testing_features = torch.tensor(testing_features, dtype=torch.float32).to(next(self.parameters()).device)
    testing_labels = torch.tensor(testing_labels, dtype=torch.long).to(next(self.parameters()).device)

    self.eval()
    with torch.no_grad():
        outputs = self.forward(testing_features)
        _, predictions = torch.max(outputs, 1)

    predictions = predictions.cpu().numpy()
    testing_labels = testing_labels.cpu().numpy()
    evaluate(predictions, "MLP", testing_labels)


def manual_gaussian_naive_bayes(train_features_pca, training_labels, test_features_pca, validation_labels):

    # Loading from models folder if already saved, training and saving otherwise
    manual_bayes_trained_model_path = "./models/gaussian.pkl"
    manual_bayes = GaussianNaiveBayes()
    if os.path.exists(manual_bayes_trained_model_path):
        print("Loading saved model of Gaussian Naive Bayes\n")
        manual_bayes = load_model(manual_bayes_trained_model_path)
    else:
        print("Training and saving the model of Gaussian Naive Bayes\n")
        manual_bayes.fit(train_features_pca, training_labels)
        save_model(manual_bayes, manual_bayes_trained_model_path)
    print("Manual Gaussian Naive Bayes Results:")
    evaluate(manual_bayes.predict(test_features_pca), "Manual Gaussian Naive Bayes", validation_labels)


def scikit_gaussian_naive_bayes(train_features_pca, training_labels, test_features_pca, validation_labels):
    print("\nScikit-Learn Gaussian Naive Bayes Results:")
    sklearn_gnb = GaussianNB()
    sklearn_gnb.fit(train_features_pca, training_labels)
    evaluate(sklearn_gnb.predict(test_features_pca), "Scikit-Learn", validation_labels)


def manual_decision_tree(train_features_pca, training_labels, test_features_pca, validation_labels):

    # Loading from models folder if already saved, training and saving otherwise
    decision_tree_trained_model_path = "./models/decision_tree.pkl"
    decision_tree = DecisionTree()
    if os.path.exists(decision_tree_trained_model_path):
        print("Loading saved model of decision tree\n")
        decision_tree = load_model(decision_tree_trained_model_path)
    else:
        print("Training and saving the model of decision tree\n")
        decision_tree.fit(train_features_pca, training_labels)
        save_model(decision_tree, decision_tree_trained_model_path)
    print("Manual Decision Tree Results:")
    evaluate(decision_tree.predict(test_features_pca), "Manual decision tree", validation_labels)


def scikit_decision_tree(train_features_pca, training_labels, test_features_pca, validation_labels):
    print("\nScikit-Learn decision tree Results:")
    sklearn_decision_tree = DecisionTreeClassifier()
    sklearn_decision_tree.fit(train_features_pca, training_labels)
    evaluate(sklearn_decision_tree.predict(test_features_pca), "Scikit-Learn", validation_labels)


def train_evaluate_mlp(device, train_features_pca, training_labels, test_features_pca, validation_labels):

    # Converting features, labels to Tensors
    training_features_tensor = torch.tensor(train_features_pca, dtype=torch.float32)
    training_labels_tensor = torch.tensor(training_labels, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(training_features_tensor, training_labels_tensor)
    training_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    mlp = MLP(device)
    mlp.train_model(10, 0.01, training_data_loader)
    evaluate_mlp_model(mlp, testing_features=test_features_pca, testing_labels=validation_labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_data_loader, validation_data_loader = load_data()
    train_features_pca, training_labels, test_features_pca, validation_labels = extract_pca_features(training_data_loader, validation_data_loader)

    manual_gaussian_naive_bayes(train_features_pca, training_labels, test_features_pca, validation_labels)
    scikit_gaussian_naive_bayes(train_features_pca, training_labels, test_features_pca, validation_labels)
    manual_decision_tree(train_features_pca, training_labels, test_features_pca, validation_labels)
    scikit_decision_tree(train_features_pca, training_labels, test_features_pca, validation_labels)
    train_evaluate_mlp(device, train_features_pca, training_labels, test_features_pca, validation_labels)


if __name__ == "__main__":
    main()