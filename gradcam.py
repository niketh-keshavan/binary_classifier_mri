"""
gradcam.py — Gradient-weighted Class Activation Mapping (Grad-CAM).

Paper: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
       Networks via Gradient-based Localization", ICCV 2017.

HOW IT WORKS
────────────
1. Run a forward pass and record the activations at a chosen
   convolutional layer (the "target layer").

2. Compute the gradient of the model's output (logit for class of
   interest) with respect to those activations.

3. Global-average-pool the gradients over the spatial dimensions
   to get a weight α_k for each channel k.

4. Compute the weighted sum of activation maps:
       cam = ReLU( Σ_k  α_k · A_k )
   ReLU keeps only features with POSITIVE influence on the class.

5. Resize the heatmap to the original image size and overlay.

WHY THE LAST CONV LAYER?
─────────────────────────
Earlier layers capture low-level features (edges, textures);
the last conv layer captures high-level semantic features
(tumour vs. normal tissue) while still retaining spatial info.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM for any CNN model.

    Parameters
    ----------
    model : nn.Module
        The trained model (in eval mode).
    target_layer : nn.Module
        The convolutional layer to visualize.
        For BinaryClassifier, use `model.encoder.stages[-1]`.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        # Storage for hooked tensors
        self._activations = None   # forward output of target layer
        self._gradients = None     # gradient w.r.t. target layer output

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ── Hook callbacks ──────────────────────────

    def _save_activation(self, module, input, output):
        """Called during forward pass — store the layer's output."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Called during backward pass — store the gradient."""
        self._gradients = grad_output[0].detach()

    # ── Main interface ──────────────────────────

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for a single input image.

        Parameters
        ----------
        input_tensor : torch.Tensor, shape (1, C, H, W)
            A single preprocessed image (batch dim = 1).

        Returns
        -------
        heatmap : np.ndarray, shape (H, W), values in [0, 1]
            The Grad-CAM heatmap resized to the input spatial size.
        """
        self.model.eval()

        # Step 1: Forward pass (hooks capture activations)
        logit = self.model(input_tensor)          # (1, 1)

        # Step 2: Backward pass from the logit
        self.model.zero_grad()
        logit.backward()

        # Step 3: Channel weights = GAP of gradients
        # gradients shape: (1, C, h, w)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Step 4: Weighted combination of activation maps
        # activations shape: (1, C, h, w)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)  # only positive influence

        # Step 5: Resize to input spatial dimensions
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],   # (H, W)
            mode="bilinear",
            align_corners=False,
        )

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        """Remove forward/backward hooks (call when done)."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()
