import time
from riga_dataset import RIGADataset, CropFundus, Rescale, ToTensor
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from unet import UNet


RIGA_TRAIN_BASE_PATH = r'./RIGA/train/'
RIGA_TRAIN_CSV_FILE = r'./RIGA/train/images.csv'

RIGA_TEST_BASE_PATH = r'./RIGA/test/'
RIGA_TEST_CSV_FILE = r'./RIGA/test/images.csv'

TRAIN_MODEL_PATH = './RIGA_model.pth'

# Parameters
EPOCHS = 1
BATCH_SIZE = 4
TRAIN = False
N_CLASSES = 2
LEARNING_RATE = 0.001
MOMENTUM = 0.9


def print_step_images(input, label, output, predicted=None):

    fig = plt.figure()
    plt.tight_layout()
    columns = 4 if predicted is None else 5

    cols = ['Raw', 'Mask', 'Class 1', 'Class 2'] if predicted is None else\
        ['Raw', 'Mask', 'Class 1', 'Class 2', 'Predicted']
    plt.suptitle(cols)

    for bs in range(BATCH_SIZE):
        a = input[bs].numpy().transpose(1, 2, 0)
        b = label[bs].numpy()
        c1 = output[bs][0].detach().numpy()
        c2 = output[bs][1].detach().numpy()

        fig.add_subplot(BATCH_SIZE, columns,(bs * columns) + 1)
        plt.imshow(a)

        fig.add_subplot(BATCH_SIZE, columns, (bs * columns) + 2)
        plt.imshow(b, cmap='gray')

        fig.add_subplot(BATCH_SIZE, columns, (bs * columns) + 3)
        plt.imshow(c1, cmap='gray')

        fig.add_subplot(BATCH_SIZE, columns, (bs * columns) + 4)
        plt.imshow(c2, cmap='gray')

        if predicted is not None:
            d = predicted[bs].detach().numpy()
            fig.add_subplot(BATCH_SIZE, columns, (bs * columns) + 5)
            plt.imshow(d, cmap='gray')

    plt.show()


def main():
    transform = transforms.Compose([CropFundus(450, 50),
                                    Rescale(64),
                                    ToTensor(),
                                    ])
    # Load train set
    train_set = RIGADataset(csv_file=RIGA_TRAIN_CSV_FILE, root_dir=RIGA_TRAIN_BASE_PATH, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)

    test_set = RIGADataset(csv_file=RIGA_TEST_CSV_FILE, root_dir=RIGA_TEST_BASE_PATH, transform=transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    model = UNet(n_class=N_CLASSES)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    if TRAIN:
        t0 = time.time()
        for epoch in range(EPOCHS):

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data['raw'], data['mask']

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()
                running_loss += loss.item()

                print('[%d, %3d] loss: %.3f' % (epoch, i + 1, running_loss))
                running_loss = 0.0

                if i == 10: print_step_images(inputs, labels, outputs)

        print('Finished Training: Elapsed time: %.3f secs' % (time.time() - t0))
        torch.save(model.state_dict(), TRAIN_MODEL_PATH)

    else:
        # Load the trained model
        model.load_state_dict(torch.load(TRAIN_MODEL_PATH))

        correct = 0
        total = 0
        index = 0
        with torch.no_grad():
            for data in test_loader:
                print('[%d] ' % index)
                inputs, labels = data['raw'], data['mask']
                # Test on test data
                outputs = model(inputs)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                index += 1

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


if __name__ == "__main__":
    main()
