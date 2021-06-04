import torch
from torch import nn, Tensor
from torch import tensor
import torch.nn.functional as F


class TransformerModel(nn.Module):
    def __init__(
        self,
        *,
        image_size: int = 28,
        image_channels: int = 1,
        patch_size: int = 7,
        num_classes: int = 10,
        dim: int = 64,
        nhead: int = 1,
        dim_feedforward: int = 64,
        depth: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        assert (
            image_size % patch_size == 0
        ), f'{image_size}, {patch_size}, {image_size % patch_size}'

        self.input_shape = (image_channels, image_size, image_size)

        self.image_size = image_size
        self.image_channels = image_channels
        self.patch_size = patch_size
        self.dim = dim

        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = image_channels * patch_size ** 2

        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.patch_embedding = nn.Linear(self.patch_dim, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=depth
        )

        self.mlp_head = nn.Linear(dim, num_classes)

    def unfold_patch(self, image: Tensor) -> Tensor:
        batch_size = len(image)

        verify_shape = torch.Size(
            [batch_size, self.image_channels, self.image_size, self.image_size]
        )
        assert image.shape == verify_shape, f'{image.shape}, {verify_shape}'

        x = image.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )

        verify_shape = torch.Size(
            [
                batch_size,
                self.image_channels,
                self.image_size // self.patch_size,
                self.image_size // self.patch_size,
                self.patch_size,
                self.patch_size,
            ]
        )
        assert x.shape == verify_shape, f'{x.shape}, {verify_shape}'

        x = x.permute(0, 2, 3, 1, 4, 5).reshape(
            batch_size,
            self.num_patches,
            self.image_channels,
            self.patch_size,
            self.patch_size,
        )
        return x

    def forward(self, image: Tensor) -> Tensor:
        batch_size = len(image)

        patches = self.unfold_patch(image)
        x = patches.flatten(2)

        verify_shape = torch.Size([batch_size, self.num_patches, self.patch_dim])
        assert x.shape == verify_shape, f'{x.shape}, {verify_shape}'

        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(1)

        verify_shape = torch.Size([batch_size, self.dim])
        assert x.shape == verify_shape, f'{x.shape}, {verify_shape}'

        x = self.mlp_head(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary

    image_size = 6
    image_channels = 3
    patch_size = 2

    model = TransformerModel(
        image_size=image_size,
        image_channels=image_channels,
        patch_size=patch_size,
        num_classes=3,
        dim=13,
        nhead=13,
        dim_feedforward=17,
        depth=2,
    )

    image_batch = torch.rand(2, *model.input_shape)
    image_batch = torch.arange(image_batch.numel()).reshape(image_batch.shape).float()
    model(image_batch)

    summary(model, (2, *model.input_shape), device='cpu')
