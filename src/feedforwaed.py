"""Task 3: Position-wise Feed-Forward Network (Section 3.3, Equation 2).

FFN(x) = max(0, x W1 + b1) W2 + b2
"""
import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)   # x W1 + b1  (expand 512 -> 2048)
        self.relu = nn.ReLU()                     # max(0, ...)
        self.linear2 = nn.Linear(d_ff, d_model)   # (...) W2 + b2 (shrink 2048 -> 512)

    def forward(self, x):
        # x: (B, L, d_model) -> (B, L, d_model)
        # nn.Linear acts on the last dimension only, so every word (position)
        # is processed separately but with the same weights.
        return self.linear2(self.relu(self.linear1(x)))


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, d_model = 2, 6, 512
    ffn = PositionwiseFeedForward(d_model, d_ff=2048)
    x = torch.randn(B, L, d_model)

    out = ffn(x)
    assert out.shape == (B, L, d_model)

    # "Position-wise": each word is processed on its own.
    # Processing word 0 alone must give the same answer as inside the full sentence.
    alone = ffn(x[:, 0:1, :])
    assert torch.allclose(out[:, 0:1, :], alone, atol=1e-5)

    # Shuffling the words just shuffles the outputs the same way.
    perm = torch.tensor([3, 1, 0, 5, 2, 4])
    assert torch.allclose(ffn(x[:, perm, :]), out[:, perm, :], atol=1e-5)

    params = sum(p.numel() for p in ffn.parameters())
    print("All checks passed.")
    print("Parameters:", params)  # expected 2,099,712