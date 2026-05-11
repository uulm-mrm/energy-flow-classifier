# pylint: disable=E1101

import torch
from torch import Tensor, nn


class EMA(nn.Module):
    """Implements the exponential moving average for parameters of a model,
    Applied as:
        EMA_t = rate * EMA_{t-1} + (1 - rate) * model_param
    To be used as a model wrapper that updates the ema parameters or on its own
    """

    def __init__(self, model: nn.Module, rate: float = 0.999) -> None:
        """
        Args:
            model (nn.Module): model for which to do EMA on
            rate (float, optional): decay rate of EMA. rate -> 1 means slower updates.
                Defaults to 0.999.
        """
        super().__init__()

        self.model = model
        self.rate = rate

        # register for state dict
        self.register_buffer("num_updates", torch.tensor(0))

        # copy of model params
        self.ema_t = nn.ParameterList(
            [
                nn.Parameter(p.clone().detach(), requires_grad=False)
                for p in model.parameters()
                if p.requires_grad
            ]
        )

    def forward(self, *args, **kwargs) -> Tensor:
        """Does the wrapped model's forward"""

        return self.model.forward(*args, **kwargs)

    def update_ema_t(self) -> None:
        """Updates the internal ema_t state"""

        self.num_updates += 1
        num_updates = self.num_updates.item()  # type: ignore

        rate = min(self.rate, (1 + num_updates) / (10 + num_updates))

        # dont accidentally update model params
        with torch.no_grad():
            params = [p for p in self.model.parameters() if p.requires_grad]
            for ema_p, p in zip(self.ema_t, params):
                # shorthand for ema_p = rate * ema_p + (1 - rate) * p
                # also inplace ema_p update
                ema_p.sub_((1 - rate) * (ema_p - p))

    def to_model(self) -> None:
        """Copy EMA_t into the model. Use once training is done"""

        params = [p for p in self.model.parameters() if p.requires_grad]
        for ema_p, p in zip(self.ema_t, params):
            p.data.copy_(ema_p.data)
