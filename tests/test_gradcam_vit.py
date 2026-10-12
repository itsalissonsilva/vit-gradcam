import unittest

import torch
from torch import nn

from gradcam_vit import ViTGradCAM, tokens_to_spatial


class TinyTokenClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 8, kernel_size=4, stride=4)
        self.class_token = nn.Parameter(torch.zeros(1, 1, 8))
        self.target_norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(inputs).flatten(2).transpose(1, 2)
        class_tokens = self.class_token.expand(inputs.shape[0], -1, -1)
        tokens = self.target_norm(torch.cat((class_tokens, patches), dim=1))
        return self.head(tokens[:, 1:].mean(dim=1))


class ViTGradCAMTest(unittest.TestCase):
    def test_tokens_to_spatial_discards_class_token(self) -> None:
        tokens = torch.arange(2 * 17 * 8).reshape(2, 17, 8)
        spatial = tokens_to_spatial(tokens)
        self.assertEqual(spatial.shape, (2, 8, 4, 4))
        torch.testing.assert_close(spatial[:, :, 0, 0], tokens[:, 1, :])

    def test_gradcam_has_input_resolution_and_normalized_values(self) -> None:
        torch.manual_seed(7)
        model = TinyTokenClassifier().eval()
        original_hook_count = len(model.target_norm._forward_hooks)

        with ViTGradCAM(model, model.target_norm) as gradcam:
            cams, logits, targets = gradcam(torch.randn(2, 3, 16, 16), [0, 2])

        self.assertEqual(cams.shape, (2, 16, 16))
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(targets.tolist(), [0, 2])
        self.assertGreaterEqual(cams.min().item(), 0.0)
        self.assertLessEqual(cams.max().item(), 1.0)
        self.assertEqual(len(model.target_norm._forward_hooks), original_hook_count)

    def test_non_square_grid_requires_explicit_shape(self) -> None:
        tokens = torch.randn(1, 13, 4)
        with self.assertRaisesRegex(ValueError, "square grid"):
            tokens_to_spatial(tokens)
        self.assertEqual(tokens_to_spatial(tokens, grid_size=(3, 4)).shape, (1, 4, 3, 4))


if __name__ == "__main__":
    unittest.main()
