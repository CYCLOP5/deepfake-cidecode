import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_fake_detect.utils import *
from utils import *
from data_utils.utils import *
from deep_fake_detect.features import *
class DeepFakeDetectModel(nn.Module):
    def __init__(self, frame_dim=None, encoder_name=None):
        super().__init__()
        self.image_dim = frame_dim
        self.num_of_classes = 1
        self.encoder = get_encoder(encoder_name)
        self.encoder_flat_feature_dim, _ = get_encoder_params(encoder_name)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_flat_feature_dim, int(self.encoder_flat_feature_dim * .10)),
            nn.Dropout(0.50),
            nn.ReLU(),
            nn.Linear(int(self.encoder_flat_feature_dim * .10), self.num_of_classes),
        )
        self.gradients = None
        self.activations = None
        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        def save_activation(module, input, output):
            self.activations = output
        self.encoder.conv_head.register_forward_hook(save_activation)
        self.encoder.conv_head.register_backward_hook(save_gradient)
    def forward(self, x, return_features=False):
        x = self.encoder.forward_features(x)
        if return_features:  
            return x
        x = self.avg_pool(x).flatten(1)
        x = self.classifier(x)
        return x
    def get_gradcam(self):
        if self.gradients is None or self.activations is None:
            raise RuntimeError("No gradients or activations found. Run a forward + backward pass first.")
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)  
        cam = torch.sum(weights * self.activations, dim=1).squeeze()  
        cam = F.relu(cam)  
        cam = cam / torch.max(cam)  
        return cam.detach().cpu().numpy()
