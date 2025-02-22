from CNN import *
from FCNN import *
from utilsCNN import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def main(epochs=50):

    train_images = datasets.CIFAR100(root='data', train=True, download=True, transform=transforms.ToTensor())
    test_images = datasets.CIFAR100(root='data', train=False, download=True, transform=transforms.ToTensor())
    img_dim = train_images[0][0].shape

    train_loader = DataLoader(train_images, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_images, batch_size=16, shuffle=False)

    model = FCNN(in_dim=img_dim[0]*img_dim[1]*img_dim[2], out_dim=100)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
            optimizer.zero_grad()
            out = model(x)
            l = loss(out, y)
            l.backward()
            optimizer.step()

        print(f"Epoch: {epoch} Loss: {l.item()}")

        model.eval()
        correct = 0
        total = 0
        for x, y in test_loader:
            x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
            predicted = model.predict(x)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        print(f"Epoch: {epoch} Accuracy: {correct/total}")

if __name__ == '__main__':
    main()