
import torch
import torch.nn as nn


class ECGTransformerBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=0.1,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        normalized = self.norm1(x)

        attended, weights = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=True,
            average_attn_weights=False
        )

        x = x + attended
        x = x + self.feed_forward(self.norm2(x))

        return x, weights


class ECGTransformer(nn.Module):
    def __init__(
        self,
        input_channels=12,
        sequence_length=1000,
        patch_size=20,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        num_classes=2
    ):
        super().__init__()

        self.patch_embedding = nn.Conv1d(
            input_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        num_patches = sequence_length // patch_size

        self.position_embedding = nn.Parameter(
            torch.randn(1, num_patches, embed_dim) * 0.02
        )

        self.layers = nn.ModuleList([
            ECGTransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.transpose(1, 2)
        x = x + self.position_embedding

        attentions = []

        for layer in self.layers:
            x, weights = layer(x)
            attentions.append(weights)

        x = self.norm(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)

        return logits, attentions
