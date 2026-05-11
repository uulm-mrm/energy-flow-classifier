from typing import Callable
import torch
from torch import Tensor

from .integrator_utils import BucherTableau


def _broadcast_to(x: Tensor, y: Tensor) -> Tensor:
    return x.view(-1, *[1] * (len(y.shape) - 1))


class Integrator:
    """Base class for integrating an ODE in time given some starting conditon"""

    def _step(
        self,
        func: Callable[[Tensor, list[Tensor]], list[Tensor]],
        tn: Tensor,
        xn: list[Tensor],
        dt: Tensor,
    ) -> tuple[Tensor, list[Tensor]]:
        """Does one step of the numerical ODE solver

        Args:
            func (Callable[[Tensor, list[Tensor]], list[Tensor]]): f from dx/dt = f(t, x)
            tn (Tensor): current time, size (B,)
            xn (list[Tensor]): current solution, size (B, D...)
            dt (Tensor): delta time increment, size (B, )

        Returns:
            tuple[Tensor, list[Tensor]]: next iteration time and solution
        """

        raise NotImplementedError

    def integrate(
        self,
        func: Callable[[Tensor, list[Tensor]], list[Tensor]],
        x0: list[Tensor],
        ints: Tensor,
        steps: int = 20,
    ) -> tuple[Tensor, list[Tensor]]:
        """Integrates the function `func` over a batch of time intervals `ints`,
        starting from some initial condition `x0`, iterating `steps` number of times

        Args:
            func (Callable[[Tensor, list[Tensor]], list[Tensor]]): dx/dt = f(t, x) ODE to solve
            x0 (list[Tensor]): batch of initial conditions, size (B, D...)
            ints (Tensor): batch of intervals in ascending or descending orders, size (B, 2)
            steps (int, optional): number of steps to numerically solve for. Defaults to 20.

        Returns:
            tuple[Tensor, list[Tensor]]: trajectory of time and solutions given the parameters,
            size (steps+1, B) for time and [(steps+1, B, D...), ...] for solutions
        """
        t0, t1 = ints[:, 0].unsqueeze(1), ints[:, 1].unsqueeze(1)
        dt = (t1 - t0) / steps

        x_traj = [
            torch.empty(
                size=(steps + 1, *_x0.shape), dtype=_x0.dtype, device=_x0.device
            )
            for _x0 in x0
        ]

        t_traj = torch.empty(
            size=(steps + 1, *t0.shape), dtype=ints.dtype, device=ints.device
        )

        # initial state
        for i, _x0 in enumerate(x0):
            x_traj[i][0] = _x0

        t_traj[0] = t0

        xn = x0
        tn = t0
        for e in range(steps):
            tn, xn = self._step(func, tn, xn, dt)

            for i, _xn in enumerate(xn):
                x_traj[i][e + 1] = _xn
            t_traj[e + 1] = tn

        return t_traj, x_traj


class EulerIntegrator(Integrator):
    """
    Euler ODE solver.

    For a function dx/dt = f(t, x) it calculates the solution numerically as:
    xn+1 = xn + dt * f(tn, xn)
    """

    def _step(
        self,
        func: Callable[[Tensor, list[Tensor]], list[Tensor]],
        tn: Tensor,
        xn: list[Tensor],
        dt: Tensor,
    ) -> tuple[Tensor, list[Tensor]]:
        states = func(tn, xn)

        for i, state in enumerate(states):
            _dt = _broadcast_to(dt, state)
            xn[i] = xn[i] + _dt * state

        tn = tn + dt

        return tn, xn


class MidpointIntegrator(Integrator):
    """
    Explicit Extended Euler ODE Solver (Midpoint Solver)

    For a function dx/dt = f(t, x) it calculates the solution numerically as:
    xn+1 = xn + dt * f(tn + dt / 2, xn + dt / 2 * f(tn, xn))
    """

    def _step(
        self,
        func: Callable[[Tensor, list[Tensor]], list[Tensor]],
        tn: Tensor,
        xn: list[Tensor],
        dt: Tensor,
    ) -> tuple[Tensor, list[Tensor]]:
        half_dt = dt / 2

        mid_states = func(tn, xn)
        for i, state in enumerate(mid_states):
            _half_dt = _broadcast_to(half_dt, state)
            mid_states[i] = xn[i] + _half_dt * state

        states = func(tn + half_dt, mid_states)
        for i, state in enumerate(states):
            _dt = _broadcast_to(dt, state)
            xn[i] = xn[i] + _dt * state

        tn = tn + dt

        return tn, xn


class RungeKuttaIntegrator(Integrator):

    def __init__(
        self,
        tableu: BucherTableau,
        device: str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        # recast the tableu
        self.tableu = BucherTableau(
            tableu.a.to(device).to(dtype),
            tableu.b.to(device).to(dtype),
            tableu.c.to(device).to(dtype),
        )

    def _step(
        self,
        func: Callable[[Tensor, list[Tensor]], list[Tensor]],
        tn: Tensor,
        xn: list[Tensor],
        dt: Tensor,
    ) -> tuple[Tensor, list[Tensor]]:

        # [(B, D...), (B, D', ...), ...]
        k0 = func(tn, xn)

        # make a list of zero tensors of the shape of each state
        # [(S, B, D...), (B, D', ...), ...]
        states = [
            torch.zeros(
                size=(self.tableu.a.shape[0], *_k0.shape),
                dtype=_k0.dtype,
                device=_k0.device,
            )
            for _k0 in k0
        ]

        # k0 is the first state in each
        for i, state in enumerate(k0):
            states[i][0] = state

        # loop over tableau rows length and calc k_i for each state
        for row_idx in range(1, self.tableu.a.shape[0]):
            # loop over states
            _xn = []  # -> [(B, D, ...), (B, D', ...)]
            _tn = tn + dt * self.tableu.c[row_idx]

            for i, state in enumerate(states):
                # state = (S, B, D...)
                _dt = _broadcast_to(dt, state[0])
                _a = _broadcast_to(self.tableu.a[row_idx], state)
                _xn.append(xn[i] + _dt * (_a * state).sum(dim=0))

            # [(B, D, ...), (B, D', ...)]
            _xn = func(_tn, _xn)
            for i, state in enumerate(_xn):
                states[i][row_idx] = state

        # return final k sum
        for i, state in enumerate(states):
            # state = (S, B, D...)
            # state_t+1 = xn + (b * k)
            _b = _broadcast_to(self.tableu.b, state)
            _dt = _broadcast_to(dt, state[0])
            xn[i] = xn[i] + (_b * state).sum(dim=0) * _dt

        return tn + dt, xn
