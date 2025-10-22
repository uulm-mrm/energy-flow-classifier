import torch
from torch import Tensor


def closest_anchor(time_predictions: Tensor, anchor_times: tuple[float, ...]) -> Tensor:
    """Maps time predictions to their closest anchor times"""

    anc_times = torch.tensor(
        anchor_times, dtype=time_predictions.dtype, device=time_predictions.device
    )

    return anc_times[
        torch.argmin((time_predictions[:, None] - anc_times[None]).abs(), dim=1)
    ]


def anchors_to_class(
    anchor_predictions: Tensor, time_class_map: dict[float, int]
) -> Tensor:
    keys = torch.tensor(list(time_class_map.keys()), device=anchor_predictions.device)
    values = torch.tensor(
        list(time_class_map.values()), device=anchor_predictions.device
    )

    return values[(anchor_predictions.unsqueeze(1) == keys).int().argmax(dim=1)]


def main():
    preds = torch.tensor([0.14, 0.45, 0.89, 0.12])
    times = (0.0, 0.33, 0.66, 1.0)

    res = closest_anchor(preds, times)
    print(res)

    pred_cls = anchors_to_class(res, dict(zip(times, (-1, 0, 1, 2))))
    print(pred_cls)


if __name__ == "__main__":
    main()
