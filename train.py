import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import DEVICE, synthesize_data
from torchsummary import summary


"""
    Model Architecture - ResNet-FPN backbone 
    ResNet Encoder and FPN for Regional Proposal Network
"""


class StarModelResNet(nn.Module):
    def __init__(self, channels_in, channels_out, padding=1, stride=2):
        super(StarModelResNet, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=channels_out)
        self.act1 = nn.ReLU()  # Activation function
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=channels_out)
        self.dnsmpl = None  # Down-sampling

        if stride != 1:
            self.dnsmpl = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(channels_out))

    def forward(self, x):
        skip_connection = x
        x = self.bn1(self.conv1(x))
        x = self.act1(x)
        x = self.bn2(self.conv2(x))
        skip_connection = self.dnsmpl(skip_connection)
        x = x + skip_connection
        x = self.act1(x)
        return x


"""
    Paper: Feature Pyramid Networks for Object Detection
"""


class StarModelFPN(nn.Module):
    def __init__(self):
        super(StarModelFPN, self).__init__()

        output_filters = [16, 32, 64, 128, 256]

        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.Conv2d(8, output_filters[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_filters[0]),
            nn.MaxPool2d(2)
        )

        """
            Building the pyramid
        """
        # Pyramid channels
        pyramid_channels = 256

        # Bottom-up channels
        self.conv_2 = StarModelResNet(output_filters[0], output_filters[1], stride=2, padding=1)
        self.conv_3 = StarModelResNet(output_filters[1], output_filters[2], stride=2, padding=1)
        self.conv_4 = StarModelResNet(output_filters[2], output_filters[3], stride=2, padding=1)
        self.conv_5 = StarModelResNet(output_filters[3], output_filters[4], stride=2, padding=1)

        self.pyramid_bottleneck = nn.Conv2d(output_filters[4], pyramid_channels, 1)

        # Reduced and smoothing bottom-up channels
        self.conv_2_reduced = nn.Conv2d(output_filters[1], pyramid_channels, 1)
        self.conv_3_reduced = nn.Conv2d(output_filters[2], pyramid_channels, 1)
        self.conv_4_reduced = nn.Conv2d(output_filters[3], pyramid_channels, 1)
        self.conv_2_smooth = nn.Conv2d(pyramid_channels, pyramid_channels, 3, 1)
        self.conv_3_smooth = nn.Conv2d(pyramid_channels, pyramid_channels, 3, 1)
        self.conv_4_smooth = nn.Conv2d(pyramid_channels, pyramid_channels, 3, 1)

        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier to predict if the image has 'star'
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(pyramid_channels, 1))

        # Regressor to predict the 5 outputs
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(pyramid_channels, 5))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv_1(x)
        out2 = self.conv_2(out1)
        out3 = self.conv_3(out2)
        out4 = self.conv_4(out3)
        out5 = self.conv_5(out4)

        pyramid_out5 = self.pyramid_bottleneck(out5)
        pyramid_out4 = self.conv_4_reduced(out4) + F.interpolate(pyramid_out5, size=out4.shape[-2:], mode="nearest")
        pyramid_out3 = self.conv_3_reduced(out3) + F.interpolate(pyramid_out4, size=out3.shape[-2:], mode="nearest")
        pyramid_out2 = self.conv_2_reduced(out2) + F.interpolate(pyramid_out3, size=out2.shape[-2:], mode="nearest")

        classifier_features = self.average_pooling(pyramid_out5)
        regressor_features = self.average_pooling(pyramid_out5)

        classification_output = self.classifier(classifier_features)
        classification_output = self.sigmoid(classification_output)

        regression_output = self.regressor(regressor_features)
        regression_output = self.sigmoid(regression_output)

        prediction = classification_output.view(x.shape[0], 1)
        bounding_box = regression_output.view(x.shape[0], 5)

        combined_output = torch.cat((prediction, bounding_box), dim=1)

        return combined_output


class StarDataset(torch.utils.data.Dataset):
    """Return star image and labels"""

    def __init__(self, data_size=50000):
        self.data_size = data_size

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx) -> t.Tuple[torch.Tensor, torch.Tensor]:
        image, label = synthesize_data(has_star=True)
        return image[None], label


"""
    COMPUTING THE LOSS FUNCTION
    Here for the Loss function I have referenced the paper - 'Learning Modulated Loss for Rotated Object Detection (2019)'  
"""


def Modulated_Loss(target, preds):

    assert target.shape[-1] == 5
    assert preds.shape[-1] == 5

    actual_x, actual_y, actual_yaw, actual_width, actual_ht = target[:, 0], target[:, 1], target[:, 2], target[:, 3], target[:, 4]
    pred_x, pred_y, pred_yaw, pred_width, pred_ht = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3], preds[:, 4]

    m_loss = torch.abs(actual_x - pred_x) + torch.abs(actual_y - pred_y) + torch.abs(actual_width - pred_width) + torch.abs(
        actual_ht - pred_ht) + torch.min(torch.abs(actual_yaw - pred_yaw), torch.abs(0.5 - torch.abs(actual_yaw - pred_yaw)))

    return m_loss


def Loss_function(target, prediction):

    assert target.shape[-1] == 6
    assert prediction.shape[-1] == 6

    # Classification & Regression Loss
    no_star_index = torch.nonzero(target[:, 0] == 0, as_tuple=True)

    loss_bounding_box = Modulated_Loss(target[:, 1:], prediction[:, 1:])
    loss_bounding_box[no_star_index] = 0

    loss_classification = torch.nn.BCELoss(reduction='none')(prediction[:, 0], target[:, 0])
    loss_total = 0.5 * (loss_bounding_box + loss_classification)

    return loss_total, loss_bounding_box, loss_classification


def train(model: StarModelFPN, dl: StarDataset, num_epochs: int) -> StarModelFPN:

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in tqdm(range(num_epochs)):
        print(f"EPOCH: {epoch}")
        losses = []
        for image, label in tqdm(dl, total=len(dl)):
            image = image.to(DEVICE).float()
            label = label.to(DEVICE).float()

            optimizer.zero_grad()

            preds = model(image)
            loss = loss_fn(preds, label)
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            optimizer.step()
        print(np.mean(losses))

    return model


def main():

    model = StarModelFPN().to(DEVICE)
    inpt = torch.rand((2, 1, 200, 200))
    with open('/content/astronomy_cv_takehome/model_summary.txt', 'w') as f:
        summary_report = summary(model, inpt.shape[1:])
        f.write(str(summary_report))
    star_model = train(
        model,
        torch.utils.data.DataLoader(StarDataset(), batch_size=64, num_workers=8),
        num_epochs=30,
    )
    torch.save(star_model.state_dict(), "model.pickle")


if __name__ == "__main__":
    main()
