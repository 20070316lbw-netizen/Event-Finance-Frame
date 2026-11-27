"""模型训练基类"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Iterable, Sequence

import torch


class Train(ABC):
    """定义训练模型代码的基类"""

    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def fit_epoch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batches: Iterable[torch.Tensor | Sequence[int]],
    ) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for rows in batches:
            self.optimizer.zero_grad(set_to_none=True)
            value = self.loss(self.model(x[rows]), y[rows])
            value.backward()
            self.optimizer.step()
            total_loss += float(value.detach().item())
            n_batches += 1
        return total_loss / n_batches if n_batches > 0 else 0.0

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batches: Sequence[torch.Tensor | Sequence[int]],
        epochs: int = 1,
    ) -> list[float]:
        return [self.fit_epoch(x, y, batches) for _ in range(epochs)]

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(x)
