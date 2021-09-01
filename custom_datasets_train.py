import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
import torch.nn.functional as F
from cnn_model import MnistCNNModel
import time

start = time.time()
print("hello")


class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Third column is for an operation indicator
        self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)

        # Check if there is an operation
        some_operation = self.operation_arr[index]
        # If there is an operation
        if some_operation:
            # Do some operation on image
            # ...
            # ...
            pass
        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height, width, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = pd.read_csv(csv_path, header=None)
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.height = height
        self.width = width
        self.transform = transform

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        # Read each 784 pixels and reshape the 1D array ([784]) to 2D array ([28,28])
        img_as_np = np.asarray(self.data.iloc[index][1:]).reshape(300, 300).astype('uint8')
        # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        # Transform image to tensor
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data.index)


batch_size = 10
transformations = transforms.Compose([transforms.ToTensor()])

# custom_mnist_from_images =  \
#     CustomDatasetFromImages('../data/mnist_labels.csv')

custom_mnist_from_csv = CustomDatasetFromCSV('data_set.txt', 300, 300, transformations)

mn_dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist_from_csv,
                                                batch_size=batch_size,
                                                shuffle=True)

mn_dataset_loader_test = torch.utils.data.DataLoader(dataset=custom_mnist_from_csv,
                                                batch_size=593,
                                                shuffle=False)


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in mn_dataset_loader_test:
        # print('Data shape:', data.shape)
        # print('Target shape:', target.shape)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        # print('Predict:', pred.shape)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        pred_int = np.asarray(pred)
        target_int = np.asarray(target)
        # print(type(pred_int))
        # print(pred_int.shape)
        # print(pred_int[0:5])
        # print(type(target_int))
        # print(target_int.shape)
        # print(target_int[0:5])
        con_fus = confusion_matrix(target_int, pred_int)
        fig = plt.figure()
        con_fus_map = sns.heatmap(con_fus)
        # plt.show()
        filename = 'con_1/'+'con_fus'+str(epoch)
        fig.savefig(filename)

    test_loss /= len(mn_dataset_loader_test.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(mn_dataset_loader_test.dataset),
        100. * correct / len(mn_dataset_loader_test.dataset)))
    test_acc.append(correct)
    print(test_acc[-1])


test_acc = []

if __name__ == "__main__":

    model = MnistCNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(1000):
        for i, (images, labels) in enumerate(mn_dataset_loader):
            print("Epoch:", epoch)
            images = Variable(images)
            labels = Variable(labels)
            # Clear gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            # Calculate loss
            loss = criterion(outputs, labels)
            # print('Training Loss:', loss)
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            break

        # test_acc = 0
        # for i, (images, labels) in enumerate(mn_dataset_loader_test):
        #     print("Test Epoch:", epoch)
        #     images = Variable(images)
        #     labels = Variable(labels)
        #     print('labels:', labels)
        #     # print(type(labels))
        #     # Clear gradients
        #     # optimizer.zero_grad()
        #     # Forward pass
        #     outputs = model(images)
        #     outputs = outputs[0]
        #     # print('outputs:', outputs)
        #     max_data = outputs.max(0)
        #     # print('outputs.max:', outputs.max(0))
        #     print('max_data[1]:', max_data[1])
        #     # print(type(max_data[1]))
        #     if max_data[1] == labels:
        #         test_acc = test_acc + 1
        #         print('test_acc:', test_acc)
        #     # Calculate loss
        #     # loss = criterion(outputs, labels)
        #     # print('Test loss:', loss)
        #     break
        # print('Test accuracy:', test_acc)
        test()

end = time.time()
print(end - start)
