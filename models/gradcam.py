import torch
from torch.nn import functional as F
from torch.autograd import Function

# Custom ReLU for Guided Backpropagation
class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(ctx, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.relu(input)
        ctx.save_for_backward(positive_mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        positive_mask, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[grad_input < 0] = 0
        grad_input *= positive_mask
        return grad_input

# Utility class for applying Grad-CAM to GNN models
class GNN_GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        self.hooks = []

    def _register_hooks(self, target_layer):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Register hooks on the target layer
        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def generate_grad_cam(self, g, features, target_layer_index=0, task_type='classification', target_class=None):
        # Replace ReLU with GuidedReLU
        self._replace_relu_with_guidedrelu()

        # Register hooks
        self._register_hooks(self.model.layers[target_layer_index])

        # Forward pass
        output, _ = self.model(g, features)
        # Zero gradients
        self.model.zero_grad()

        # Backward pass based on task type
        if task_type == 'classification' and target_class is not None:
            one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_().to(features.device)
            one_hot_output[0][target_class] = 1
            output.backward(gradient=one_hot_output)
        elif task_type == 'regression':
            output.backward(torch.ones_like(output))
        else:
            raise ValueError("Invalid task_type: choose 'classification' or 'regression'")

        # Remove hooks (clean up)
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        return self.activations, self.gradients

    def _replace_relu_with_guidedrelu(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.forward = GuidedBackpropReLU.apply
