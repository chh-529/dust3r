"""
Receiver-side appearance decoder: reconstruct RGB from the RECEIVED features.

Why this exists
---------------
DUSt3R's geometry path never needs pixels at the receiver -- the ViT features carry
everything the pointmap heads use. But colouring a point cloud does need pixels, and
DUSt3R's demo takes them straight from the input images (dust3r/demo.py: the vertex
colours and the mesh face colours both come from `scene.imgs`). In a communication
setting the receiver must not have the images at all, so taking colours from there
would be cheating.

This head closes the gap without opening a side channel: colour is decoded from the
same feat_hat that came out of the channel. No extra bandwidth is used -- the
bandwidth ratio k/n is unchanged -- because the appearance information rides along in
the very same symbols the geometry does.

Design
------
MAE's decoder output layer: one Linear per token, from the token embedding to the raw
pixels of that token's patch, then unpatchify. That is the whole thing. Reconstructions
come out blurry (low frequencies dominate), which is the honest outcome: it shows
exactly how much appearance survived the compression and the noise, and the blur grows
as SNR drops -- a figure worth having.

Training
--------
Can be trained jointly with a small loss weight, but the recommended route is to train
it SEPARATELY after the geometry model is frozen. Then the 3D numbers are provably
unaffected by the appearance objective, and nobody can ask whether the RGB loss
degraded the geometry.

Colour space: DUSt3R normalizes images with mean=std=0.5 (dust3r/utils/image.py
ImgNorm), so `view['img']` lives in [-1, 1]. This head predicts in that same space, so
its output can be compared to `view['img']` directly.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn


def unpatchify(x: torch.Tensor, grid_hw: Tuple[int, int], patch_size: int,
               channels: int = 3) -> torch.Tensor:
    """
    Fold per-patch pixel predictions back into an image.

    Args:
        x: (B, S, patch_size^2 * channels), S == grid_h * grid_w
        grid_hw: (grid_h, grid_w), the patch grid, in tokens
        patch_size: pixels per patch side
        channels: 3 for RGB

    Returns:
        (B, channels, grid_h * patch_size, grid_w * patch_size)

    Note:
        croco's CroCoNet.unpatchify assumes a square grid (h = w = sqrt(S)) and cannot
        be used here -- DUSt3R runs at 512x384, a 32x24 grid.
    """
    B, S, _ = x.shape
    gh, gw = grid_hw
    if gh * gw != S:
        raise ValueError(f'{grid_hw = } does not match {S = } tokens')

    x = x.reshape(B, gh, gw, patch_size, patch_size, channels)
    x = torch.einsum('nhwpqc->nchpwq', x)
    return x.reshape(B, channels, gh * patch_size, gw * patch_size)


class RGBHead(nn.Module):
    """
    feat_hat (B, S, D) -> image (B, 3, H, W), in DUSt3R's normalized [-1, 1] colour space.

    Runs on the receiver. It only ever sees features that came out of the channel; it is
    never given an image, which is the point.
    """

    def __init__(self,
                 feat_dim: int = 1024,
                 patch_size: int = 16,
                 hidden_dim: Optional[int] = None):
        """
        Args:
            feat_dim: D of the (reconstructed) ViT features.
            patch_size: DUSt3R's patch size, 16.
            hidden_dim: if given, insert one Linear+GELU before the pixel projection.
                None (default) is the plain MAE-style single projection.
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.patch_size = patch_size
        out_dim = patch_size * patch_size * 3

        if hidden_dim is None:
            self.proj = nn.Linear(feat_dim, out_dim)
        else:
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )
        # no output activation: the target is in [-1, 1] but a tanh here would saturate
        # gradients on the (common) near-white and near-black patches. Clamp for display.

    def extra_repr(self) -> str:
        return f'feat_dim={self.feat_dim}, patch_size={self.patch_size}'

    def forward(self, feat: torch.Tensor, img_hw: Tuple[int, int],
                true_shape: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            feat: (B, S, D) reconstructed features for ONE view.
            img_hw: (H, W) of the padded image in pixels -- geometry side information the
                receiver knows, exactly like the patch positions. NOT pixel content.
            true_shape: (B, 2) per-sample (height, width). Only used to undo the
                landscape/portrait transposition that ManyAR_PatchEmbed applies. If None,
                every image is assumed landscape.

        Returns:
            (B, 3, H, W) in the [-1, 1] space of DUSt3R's ImgNorm.
        """
        H, W = img_hw
        gh, gw = H // self.patch_size, W // self.patch_size

        pixels = self.proj(feat)                             # (B, S, p*p*3)

        if true_shape is None:
            return unpatchify(pixels, (gh, gw), self.patch_size)

        # ManyAR_PatchEmbed lays out portrait images on a transposed grid, so undo it
        height, width = true_shape.T
        is_landscape = (width >= height)
        if bool(is_landscape.all()):
            return unpatchify(pixels, (gh, gw), self.patch_size)
        if bool((~is_landscape).all()):
            return unpatchify(pixels, (gw, gh), self.patch_size).swapaxes(-1, -2)

        out = pixels.new_zeros((pixels.size(0), 3, H, W))
        out[is_landscape] = unpatchify(pixels[is_landscape], (gh, gw), self.patch_size)
        out[~is_landscape] = unpatchify(
            pixels[~is_landscape], (gw, gh), self.patch_size).swapaxes(-1, -2)
        return out


def to_display_rgb(img: torch.Tensor) -> torch.Tensor:
    """
    [-1, 1] normalized image -> [0, 1] for saving / colouring a point cloud.
    Clamps, because the head has no output activation and can overshoot.
    """
    return (img.clamp(-1, 1) + 1) / 2


def rgb_loss(pred: torch.Tensor, target: torch.Tensor,
             mode: str = 'l1') -> torch.Tensor:
    """
    Appearance loss between the reconstructed and the original image, both in [-1, 1].

    L1 (default) gives sharper reconstructions than L2 at the same PSNR; L2 is what MAE
    uses. Neither is perceptual -- add LPIPS on top if the figures need it.
    """
    if mode == 'l1':
        return (pred - target).abs().mean()
    if mode == 'l2':
        return (pred - target).pow(2).mean()
    raise ValueError(f'invalid {mode = }, expected "l1" or "l2"')


if __name__ == '__main__':
    torch.manual_seed(0)

    B, D, P = 2, 1024, 16
    H, W = 384, 512
    S = (H // P) * (W // P)
    print(f'image {H}x{W}, patch {P} -> {H//P}x{W//P} = {S} tokens')

    head = RGBHead(D, P)
    print(head)
    print(f'parameters: {sum(p.numel() for p in head.parameters()):,}')

    feat = torch.randn(B, S, D)

    # landscape
    ts_land = torch.tensor([[H, W]] * B)
    img = head(feat, (H, W), ts_land)
    assert img.shape == (B, 3, H, W), img.shape
    print(f'[ok] landscape: {tuple(feat.shape)} -> {tuple(img.shape)}')

    # portrait, and a mixed batch
    ts_port = torch.tensor([[W, H]] * B)
    assert head(feat, (H, W), ts_port).shape == (B, 3, H, W)
    ts_mix = torch.tensor([[H, W], [W, H]])
    assert head(feat, (H, W), ts_mix).shape == (B, 3, H, W)
    print('[ok] portrait and mixed batches keep the output shape')

    # unpatchify must be the exact inverse of a patchify on a non-square grid
    target = torch.randn(B, 3, H, W)
    gh, gw = H // P, W // P
    patches = target.reshape(B, 3, gh, P, gw, P)
    patches = torch.einsum('nchpwq->nhwpqc', patches).reshape(B, gh * gw, P * P * 3)
    assert torch.allclose(unpatchify(patches, (gh, gw), P), target, atol=1e-6)
    print('[ok] unpatchify inverts patchify on the 24x32 grid')

    # loss and display range
    loss = rgb_loss(img, target)
    loss.backward()
    assert torch.isfinite(loss) and head.proj.weight.grad is not None
    disp = to_display_rgb(img)
    assert disp.min() >= 0 and disp.max() <= 1
    print(f'[ok] l1 loss = {loss.item():.4f}, gradients flow, display range [0, 1]')
