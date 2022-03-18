import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
from io import open
import os
from PIL import Image
import pathlib
import glob

train_path = ""
pred_path = ""
root = pathlib.Path(train_path)
classes = sorted([j.name.split("/")[-1] for j in root.iterdir()])


class ConvNet(nn.Module):
    def __init__(self, num_classes=5):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1
        )
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()

        self.fc = nn.Linear(in_features=75 * 75 * 32, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = output.view(-1, 32 * 75 * 75)

        output = self.fc(output)

        return output


checkpoint = torch.load("best_checkpoint.model")
model = ConvNet(num_classes=5)
model.load_state_dict(checkpoint)
model.eval()


transformer = transforms.Compose(
    [  # Remove transformations since we want to make predicition on original images (not processed images)
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def prediction(img_path, transformer):

    image = Image.open(img_path)  # Read Image utilizng pillow package

    # Feed image to transformer to conver image to tensor
    image_tensor = transformer(image).float()

    # Pytorch reads images as batches, add extra batch dimension
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():  # Device check
        image_tensor.cuda()

    input = Variable(image_tensor)

    output = model(input)  # Prediction using inputted image.

    # Argmax function get get the category ID with the max probablity (cnn thinks that this image is in this category)
    index = output.data.numpy().argmax()

    pred = classes[index]  # Index into classes

    return pred  # Returns Category name as an ouput


# Fetch all images and save it in the path with the pred name
images_path = glob.glob(pred_path + "/*.jpg")
pred_dict = {}  # Save image names with considred predictions

for i in images_path:
    pred_dict[i[i.rfind("/") + 1 :]] = prediction(i, transformer)
pred_dict  # Print it out.
