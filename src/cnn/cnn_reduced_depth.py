from torch import nn, optim, torch


class CNN_REDUCED_DEPTH(nn.Module):

    def __init__(self, device):
        super(CNN_REDUCED_DEPTH, self).__init__()

        # Decreased the depth to 4 convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        calculated_input = torch.zeros(1, 3, 224, 224)
        feature_output = self.features(calculated_input)
        self.flattened_size = feature_output.view(-1).size(0)

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )
        self.to(device)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_model(self, epochs, learning_rate, train_loader):
        self.train()

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), learning_rate, 0.9)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(next(self.parameters()).device)
                batch_labels = batch_labels.to(next(self.parameters()).device)

                optimizer.zero_grad()
                outputs = self.forward(batch_features)
                loss = loss_function(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print("Training Epoch is %d of %d and Current Loss is %.4f" % (epoch + 1, epochs, epoch_loss / len(train_loader)))
