from torch import nn, optim


class MLP_EXPERIMENT2(nn.Module):

    def __init__(self, device):
        super(MLP_EXPERIMENT2, self).__init__()

        # removed middle layer
        self.layers = nn.Sequential(
            nn.Linear(50, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        self.to(device)

    def forward(self, x):
        return self.layers(x)

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
