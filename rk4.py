import numpy as np
from typing import Callable


class RK4:
    def __init__(
        self,
        rhs_func: Callable[
            [np.float64, 'np.ndarray[int, np.dtype[np.float64]]'],
            'np.ndarray[int, np.dtype[np.float64]]',
        ],
        h0: np.float64,
        h_func: Callable[
            [np.float64, 'np.ndarray[int, np.dtype[np.float64]]', np.float64], np.float64
        ],
        q0: 'np.ndarray[int, np.dtype[np.float64]]',
        x0=np.float64(0.0),
        tol=np.float64(1.0e-12),
        MAX_ITER=100000,
    ) -> None:
        self.xi: list[np.float64] = []
        self.result: list['np.ndarray[int, np.dtype[np.float64]]'] = []
        self.q0 = q0
        self.x0 = x0
        self.h0 = h0
        self.rhs_func = rhs_func
        self.h_func = h_func
        self.tol = tol
        self.MAX_ITER = MAX_ITER
        self.integrate(h0, tol, q0, x0, MAX_ITER)

    def integrate(
        self,
        h0: np.float64,
        tol: np.float64,
        q0: 'np.ndarray[int, np.dtype[np.float64]]',
        x0: np.float64,
        MAX_ITER: int,
    ) -> None:
        # q = np.zeros_like(q0, dtype=np.float64)
        x = x0
        h = h0
        q = q0.copy()
        iter = 0
        while h > tol and iter < MAX_ITER:
            self.result.append(q.copy())
            self.xi.append(x)
            # fourth order runge-kutta
            k1 = self.rhs_func(x, q)
            k2 = self.rhs_func(x + 0.5 * h, q + 0.5 * h * k1)
            k3 = self.rhs_func(x + 0.5 * h, q + 0.5 * h * k2)
            k4 = self.rhs_func(x + h, q + h * k3)

            q += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            x += h

            h = self.h_func(x, q, h)
            iter += 1
        self.result.append(q.copy())
        self.xi.append(x)
        self.result = list(np.array(self.result).transpose())
