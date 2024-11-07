import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import models
from sklearn.decomposition import PCA

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

training_images = torchvision.datasets.CIFAR10('./data', train=True, transform=transform, download=True)
validation_images = torchvision.datasets.CIFAR10('./data', train=False, transform=transform, download=True)


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

training_indices = get_subset_of_data(training_images, 500)
validation_indices = get_subset_of_data(validation_images, 100)

training_subset = Subset(training_images, training_indices)
validation_subset = Subset(validation_images, validation_indices)

training_data_loader = DataLoader(training_subset, batch_size=64, shuffle=True)
validation_data_loader = DataLoader(validation_subset, batch_size=64, shuffle=False)

feature_extractor = models.resnet18(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
feature_extractor.eval()

def extract_features(data_loader):
    features = []
    labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            extracted_feature = feature_extractor(images)
            extracted_feature = extracted_feature.view(extracted_feature.size(0), -1)
            features.append(extracted_feature)
            labels.extend(labels)
    return torch.cat(features).numpy(), np.array(labels)


training_features, training_labels = extract_features(training_data_loader)
validation_features, validation_labels = extract_features(validation_data_loader)

pca = PCA(50)
train_features_pca = pca.fit_transform(training_features)
test_features_pca = pca.transform(validation_features)

