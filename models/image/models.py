import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from torchvision.models import vit_b_16

class DensNetWithHead(nn.Module):
    def __init__(self,  hidden_layer_sizes, dropout_rate, num_classes,freeze_backbone = False):
        super(DensNetWithHead, self).__init__()

        # Pretrained DenseNet backbone
        self.backbone = models.densenet121(pretrained=True)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        num_features = self.backbone.classifier.in_features

        # Remove the last classification layer of the backbone
        self.backbone.classifier = nn.Identity()

        # Custom head with hidden layers
        layers = []
        input_size = num_features

        for size in hidden_layer_sizes:
            linear_layer = nn.Linear(input_size, size)
            init.kaiming_uniform_(linear_layer.weight, nonlinearity='relu')
            layers.append(linear_layer)
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout_rate))
            input_size = size

        # Output layer
        layers.append(nn.Linear(input_size, num_classes))

        # Assemble the custom head
        self.custom_head = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the backbone
        features = self.backbone(x)
  

        # Forward pass through the custom head
        output = self.custom_head(features)

        return output
    


class ViTWithHead(nn.Module):
    def __init__(self, hidden_layer_sizes, dropout_rate, num_classes, pretrained=True):
        super(ViTWithHead, self).__init__()

        # Load the pretrained Vision Transformer backbone
    
        self.backbone = vit_b_16(pretrained=pretrained)

        # Assuming the ViT model ends with a linear layer and we only use the head output
        num_features = self.backbone.heads[0].in_features

        # Remove the last classification head of the backbone (if exists)
        self.backbone.heads = nn.Identity()

        # Custom head with hidden layers
        layers = []
        input_size = num_features
        for size in hidden_layer_sizes:
            linear_layer = nn.Linear(input_size, size)
            torch.nn.init.kaiming_uniform_(linear_layer.weight, nonlinearity='relu')
            layers.append(linear_layer)
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout_rate))
            input_size = size
                                                                   
        # Output layer
        layers.append(nn.Linear(input_size, num_classes))

        # Assemble the custom head
        self.custom_head = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the backbone
        features = self.backbone(x)

        # The output from ViT is usually a tuple, we need only the last hidden state
        if isinstance(features, tuple):
            features = features[0]  # Getting the last hidden state

        # Forward pass through the custom head
        output = self.custom_head(features)

        return output
    



import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from torchvision.models import ResNet101_Weights

class ResNetWithHead(nn.Module):
    def __init__(self, hidden_layer_sizes, dropout_rate, num_classes, freeze_backbone=False):
        super(ResNetWithHead, self).__init__()

        # Pretrained ResNet101 backbone
        self.backbone = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    module.requires_grad_(False)
        num_features = self.backbone.fc.in_features

        # Remove the last classification layer of the backbone
        self.backbone.fc = nn.Identity()

        # Custom head with hidden layers
        layers = []
        input_size = num_features

        for size in hidden_layer_sizes:
            linear_layer = nn.Linear(input_size, size)
            init.kaiming_uniform_(linear_layer.weight, nonlinearity='relu')
            nn.init.constant_(linear_layer.bias, 0)
            layers.append(linear_layer)
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = size

        # Output layer
        output_layer = nn.Linear(input_size, num_classes)
        init.kaiming_uniform_(output_layer.weight, nonlinearity='linear')
        nn.init.constant_(output_layer.bias, 0)
        layers.append(output_layer)

        # Assemble the custom head
        self.custom_head = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the backbone
        features = self.backbone(x)

        # Forward pass through the custom head
        output = self.custom_head(features)

        return output

    def train(self, mode=True):
        super(ResNetWithHead, self).train(mode)
        if self.freeze_backbone:
            self.backbone.eval()


if __name__ == '__main__':
    # Example parameters
    hidden_layer_sizes = [512, 256]
    dropout_rate = 0.5
    num_classes = 10
    freeze_backbone = False

    # Instantiate the model
    model = ResNetWithHead(hidden_layer_sizes, dropout_rate, num_classes, freeze_backbone)

    # Example input
    input_tensor = torch.randn(8, 3, 224, 224)  # Batch size of 8, image size 224x224

    # Forward pass
    output = model(input_tensor)
    print(output.shape)  # Should be [8, num_classes]
