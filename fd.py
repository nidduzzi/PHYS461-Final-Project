import numpy as np


class FiniteDifferencing:
    def __init__(self, lhs_coeffs, lhs_nonlinear, rhs) -> None:
        if len(lhs_coeffs) % 2:
            C = np.zeros(len(rhs))
            for coeff in lhs_coeffs:
                C = C + np.diagflat(coeff)


