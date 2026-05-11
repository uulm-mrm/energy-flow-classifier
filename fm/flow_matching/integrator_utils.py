from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class BucherTableau:
    """A Butcher Tableau for RK integration methods"""

    a: Tensor  # the A matrix of size (S, S)
    b: Tensor  # the b vector of size (S,)
    c: Tensor  # the c vector of size (S,)


# ordinary RK4 table
RK4_TABLEAU = BucherTableau(
    a=torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    ),
    b=torch.tensor([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]),
    c=torch.tensor([0.0, 0.5, 0.5, 1.0]),
)

# 3/8 RK4 table
RK4_38_TABLEAU = BucherTableau(
    a=torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0 / 3.0, 0.0, 0.0, 0.0],
            [-1.0 / 3.0, 1.0, 0.0, 0.0],
            [1.0, -1.0, 1.0, 0.0],
        ]
    ),
    b=torch.tensor([1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0]),
    c=torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]),
)

# Midpoint table
RK2_TABLEAU = BucherTableau(
    a=torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 0.0],
        ]
    ),
    b=torch.tensor([0.0, 1.0]),
    c=torch.tensor([0.0, 0.5]),
)

# Dormand-Prince 5th order Table
DOPRI_TABLEAU = BucherTableau(
    a=torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0],
            [
                19372.0 / 6561.0,
                -25360.0 / 2187.0,
                64448.0 / 6561.0,
                -212.0 / 729,
                0.0,
                0.0,
                0.0,
            ],
            [
                9017.0 / 3168.0,
                -355.0 / 33.0,
                46732.0 / 5247.0,
                49.0 / 176.0,
                -513.0 / 18656.0,
                0.0,
                0.0,
            ],
            [
                35.0 / 384.0,
                0.0,
                500.0 / 1113.0,
                125.0 / 192.0,
                -2187.0 / 6784.0,
                11.0 / 84.0,
                0.0,
            ],
        ]
    ),
    b=torch.tensor(
        [
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
            0.0,
        ]
    ),
    c=torch.tensor([0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0]),
)
