
import torch
import sys

sys.path.append(".")

from models.transformer import ECGTransformer

model = ECGTransformer()

dummy_ecg = torch.randn(4, 12, 1000)

logits, attentions = model(dummy_ecg)

print("Input shape:", dummy_ecg.shape)
print("Output shape:", logits.shape)
print("Attention layers:", len(attentions))
print("Attention shape:", attentions[0].shape)

print("\nTransformer test passed!")
