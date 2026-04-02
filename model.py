import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import os

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = nn.Linear(16 * 14 * 14, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16*14*14)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x
    def save(self, file_name='./model/model.pth'):
        if not os.path.exists('./model'):
            os.makedirs('./model')
        torch.save(self.state_dict(), file_name)


class Trainer:
    def __init__(self,model,lr):
        self.model = model
        self. lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def loadDate(train=True):
        if train:
            imagenet_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        else:
            imagenet_data = torchvision.datasets.MNIST(root='./data/test', train=False, transform=transforms.ToTensor(), download=True)
        dataLoader = DataLoader(imagenet_data, batch_size=64, shuffle=True)
        return dataLoader

    def train(self, image, correctAnswer):
        image = image.view(-1, 1, 28, 28)
        prediction = self.model(image)
        loss = self.criterion(prediction, correctAnswer)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def neuralLoad(self, path):
        saves = torch.load(path)
        self.model.load_state_dict(saves)
        self.model.eval()

    def numberWindow(self, number):
        number = torch.tensor(number, dtype=torch.float).unsqueeze(0)
        number = number.view(-1, 1, 28,28)
        self.neuralLoad("model/model.pth")
        prediction = self.model(number)
        index = prediction.argmax().item()
        print(index)



if __name__ == "__main__":
    model = NeuralNet(784, 128, 10)
    trainer = Trainer(model, lr=0.001)
    data = trainer.loadDate(train=True)
    for i in range(10):
        for image, answer in data:
            err = trainer.train(image, correctAnswer=answer)
            print(err)
    model.save('./model/model.pth')
