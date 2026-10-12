"""Minimal Grad-CAM implementation with Vision Transformer token support."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ReshapeTransform = Callable[[Tensor], Tensor]


def tokens_to_spatial(
    tokens: Tensor,
    *,
    grid_size: tuple[int, int] | None = None,
    num_prefix_tokens: int = 1,
) -> Tensor:
    """Convert ``[batch, tokens, channels]`` ViT activations to ``[B, C, H, W]``.

    Classification ViTs normally prepend one class token. Some architectures add
    more prefix tokens; pass the appropriate ``num_prefix_tokens`` for those.
    When ``grid_size`` is omitted, a square patch grid is inferred.
    """

    if tokens.ndim != 3:
        raise ValueError(f"Expected [batch, tokens, channels], got {tuple(tokens.shape)}")
    if num_prefix_tokens < 0 or num_prefix_tokens >= tokens.shape[1]:
        raise ValueError("num_prefix_tokens must leave at least one patch token")

    patches = tokens[:, num_prefix_tokens:, :]
    patch_count = patches.shape[1]

    if grid_size is None:
        side = math.isqrt(patch_count)
        if side * side != patch_count:
            raise ValueError(
                f"Cannot infer a square grid from {patch_count} patch tokens; "
                "pass grid_size=(height, width)"
            )
        grid_size = (side, side)

    height, width = grid_size
    if height * width != patch_count:
        raise ValueError(
            f"grid_size={grid_size} needs {height * width} tokens, got {patch_count}"
        )

    return patches.reshape(tokens.shape[0], height, width, tokens.shape[2]).permute(0, 3, 1, 2)


class ViTGradCAM:
    """Compute Grad-CAM from a ViT layer whose output is a token sequence.

    Args:
        model: A classifier returning logits shaped ``[batch, classes]``.
        target_layer: Late transformer layer to hook (often the final block's
            first LayerNorm).
        reshape_transform: Converts layer activations and gradients to spatial
            feature maps. ``tokens_to_spatial`` handles standard class-token ViTs.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        reshape_transform: ReshapeTransform = tokens_to_spatial,
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.reshape_transform = reshape_transform
        self._activations: Tensor | None = None
        self._gradients: Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._capture_activations)

    def _capture_activations(
        self, module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor
    ) -> None:
        del module, inputs
        if not isinstance(output, Tensor):
            raise TypeError("The selected target layer must return a Tensor")
        if not output.requires_grad:
            raise RuntimeError("Grad-CAM requires gradients; do not use torch.no_grad()")
        self._activations = output
        output.register_hook(self._capture_gradients)

    def _capture_gradients(self, gradients: Tensor) -> None:
        self._gradients = gradients

    def __call__(
        self,
        inputs: Tensor,
        target_classes: int | Sequence[int] | Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return normalized CAMs, model logits, and the selected class indices."""

        if self._handle is None:
            raise RuntimeError("This ViTGradCAM instance has been closed")

        self._activations = None
        self._gradients = None
        self.model.zero_grad(set_to_none=True)

        logits = self.model(inputs)
        if not isinstance(logits, Tensor) or logits.ndim != 2:
            raise ValueError("The model must return logits shaped [batch, classes]")

        targets = self._prepare_targets(logits, target_classes)
        selected_scores = logits.gather(1, targets[:, None]).sum()
        selected_scores.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("The target layer did not produce captured activations and gradients")

        activations = self.reshape_transform(self._activations)
        gradients = self.reshape_transform(self._gradients)
        if activations.ndim != 4 or activations.shape != gradients.shape:
            raise ValueError(
                "reshape_transform must return matching [batch, channels, height, width] tensors"
            )

        channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
        cams = torch.relu((channel_weights * activations).sum(dim=1, keepdim=True))
        cams = F.interpolate(cams, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        cams = self._normalize(cams).squeeze(1)

        return cams.detach(), logits.detach(), targets.detach()

    @staticmethod
    def _prepare_targets(
        logits: Tensor, target_classes: int | Sequence[int] | Tensor | None
    ) -> Tensor:
        batch_size, class_count = logits.shape
        if target_classes is None:
            targets = logits.argmax(dim=1)
        elif isinstance(target_classes, int):
            targets = torch.full((batch_size,), target_classes, device=logits.device)
        else:
            targets = torch.as_tensor(target_classes, device=logits.device)
            if targets.numel() != batch_size:
                raise ValueError(f"Expected {batch_size} target classes, got {targets.numel()}")
            targets = targets.reshape(batch_size)

        targets = targets.to(dtype=torch.long)
        if ((targets < 0) | (targets >= class_count)).any():
            raise ValueError(f"Target classes must be between 0 and {class_count - 1}")
        return targets

    @staticmethod
    def _normalize(cams: Tensor) -> Tensor:
        flattened = cams.flatten(start_dim=1)
        minimum = flattened.min(dim=1).values[:, None, None, None]
        maximum = flattened.max(dim=1).values[:, None, None, None]
        return (cams - minimum) / (maximum - minimum).clamp_min(1e-8)

    def close(self) -> None:
        """Remove the forward hook held by this instance."""

        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "ViTGradCAM":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def torchvision_vit_b_16_target_layer(model: nn.Module) -> nn.Module:
    """Return the conventional Grad-CAM target layer for torchvision ViT-B/16."""

    try:
        return model.encoder.layers[-1].ln_1  # type: ignore[attr-defined, no-any-return]
    except (AttributeError, IndexError) as error:
        raise ValueError("Expected a torchvision VisionTransformer model") from error
