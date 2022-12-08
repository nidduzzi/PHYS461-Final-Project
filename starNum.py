import numpy as np
from scipy.constants import G, Boltzmann, atomic_mass, proton_mass
from scipy.special import logsumexp
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from rk4 import RK4


def ThetaXi(n=3.0, dxi=0.01, MAX_XI=6.9, MAX_ITER=10000):
    # N = int(np.floor(MAX_XI / dxi))

    # # Calculate polytropic temperature and dimensionless radius
    # theta: "np.ndarray[int, np.dtype[np.float32]]" = np.linspace(1.0, 0.0, N)
    # theta[:1] = 1.0
    # dxi2 = dxi**2
    # xi: "np.ndarray[int, np.dtype[np.float32]]" = np.array([dxi * i for i in range(N)])
    # thetanew: "np.ndarray[int, np.dtype[np.float32]]" = np.zeros(N)
    # thetanew[1] = 1.0
    # thetanew[0] = 1.0
    # thetaold: "np.ndarray[int, np.dtype[np.float32]]" = theta
    # rhs: "np.ndarray[int, np.dtype[np.float32]]" = np.zeros(N - 2)
    # rhs[0] = 1 / dxi  # at the other boundary theta is 0 so the boundary is also 0
    # iter = 0
    # print(theta)

    # while np.amax(np.abs(thetaold - thetanew)) > 0.01 and iter < MAX_ITER:
    #     coeff = np.zeros((N - 2, N - 2))
    #     np.fill_diagonal(coeff[1:], (xi[1:-2] + dxi) / dxi2)
    #     np.fill_diagonal(coeff, np.full(N - 2, -2 / dxi2))
    #     np.fill_diagonal(coeff[:, 1:], (xi[2:-1] - dxi) / dxi2)
    #     print(coeff)
    #     nonlinear: "np.ndarray[int, np.dtype[np.float32]]" = np.array(xi[1:-1] * theta[1:-1] ** n)
    #     thetanew[1:-1] = np.matmul(np.linalg.inv(coeff), rhs - nonlinear)
    #     thetaold = theta
    #     theta = thetanew
    #     iter += 1

    # return {"theta": theta, "xi": xi}

    def h_func(x: np.float64, q: "np.ndarray[int, np.dtype[np.float64]]", h: np.float64):
        R_est = x - q[0] / q[1]
        if x + h > R_est:
            h = -q[0] / q[1]
        return h

    def rhs_func(x: np.float64, q: "np.ndarray[int, np.dtype[np.float64]]"):
        """the righthand side of the LE system, q' = f"""
        f = np.zeros_like(q)
        # y' = z
        f[0] = q[1]
        # for z', we need to use the expansion if we are at x = 0,
        # to avoid dividing by 0
        if x == 0.0:
            f[1] = (2.0 / 3.0) - q[0] ** n
        else:
            f[1] = -2.0 * q[1] / x - q[0] ** n

        return f

    solver = RK4(
        rhs_func,
        np.float64(1.0e-5),
        h_func,
        np.array([1.0, 0.0]),
        x0=np.float64(0.0),
        tol=np.float64(1.0e-12),
        MAX_ITER=MAX_ITER,
    )
    xi = np.array(solver.xi)
    theta, dthetadxi = np.array(solver.result)

    return xi, theta, dthetadxi


def Xi2dThetadXi(n, xi, theta, dxi=0.001):
    xi2dthdxi = [0.0, -(xi[1] ** 2 * theta[1] ** n)]
    for i in range(2, len(xi)):
        xi2dthdxi.append(-2.0 * dxi * xi[i] ** 2.0 * theta[i] ** n + xi2dthdxi[-2])
    return xi2dthdxi


def polytropicNn(n, xiR, xi2dthdxiR):
    return (
        (((4 * np.pi) ** (1 / n)) / (n + 1))
        * ((-xi2dthdxiR) ** ((1 - n) / n))
        * (xiR ** ((n - 3) / n))
    )


# def temp(xi, dthetadxi, P_xi, ):
#     def rad():

#     def ad(P, T, ):
#         pass
#     def rhs_func(x: np.float64, q: "np.ndarray[int, np.dtype[np.float64]]"):
#         pass
#     def h_func(x: np.float64, q: "np.ndarray[int, np.dtype[np.float64]]", h: np.float64):
#         pass


MSUN = 1.989 * 10.0**30.0  # Mass of the sun
RSUN = 696.342 * 10.0**6.0  # Radius of the sun


def run(
    PIN=3.0,  # Polytropic index number
    M=1.0 * MSUN,  # Mass of star
    R=1.0 * RSUN,  # Radius of star
    X=0.73,  # ratio of hydrogen
    Y=0.26,  # ratio of helium
    Z=0.01,  # ratio  of "metalic" elements
    MAX_ITER = 10000000000
):
    MU = 1 / (2 * X + 0.75 * Y + 0.5 * Z)  # Mean molecular wheight
    # Calculate Polytropic temperatures and dimensionless radius coordinate (Lane Emden)
    xi, theta, dthetadxi = ThetaXi(n=PIN, MAX_ITER=MAX_ITER)
    xi2dthdxi = xi**2 * dthetadxi
    # Calculate constant N_n
    Nn = polytropicNn(PIN, xi[-1], xi2dthdxi[-1])
    # Calculate Polytropic constant k
    k = R ** ((3 - PIN) / PIN) * M ** ((PIN - 1) / PIN) * G * Nn
    # Calculate radius constant r_n
    r_n = R / xi[-1]
    # Calculate core density and pressure
    rho_c = (3 * M) / (4 * np.pi * R**3) * (xi[-1]) / (3 * -dthetadxi[-1])
    # Calculate density
    rho_xi = rho_c * theta**PIN
    # Calculate Mass_r
    M_xi = 4 * np.pi * r_n**3 * rho_c * -xi2dthdxi
    # Calculate Pressure
    P_c = k * rho_c ** ((PIN + 1.0) / PIN)
    P_xi = k * np.sign(rho_xi) * np.abs(rho_xi) ** ((PIN + 1.0) / PIN)
    # Calculate Temperature
    T_c = k * rho_c ** (1 / PIN) * (MU * atomic_mass) / Boltzmann
    T_xi = T_c * theta
    # Calculate radius
    R_xi = r_n * xi

    # Plot
    df_scaled = pd.DataFrame(
        {
            "xi": xi / xi[-1],
            "dthetadxi": dthetadxi,
            "theta": theta,
            "rho": rho_xi / rho_c,
            "M": M_xi / M,
            "P": P_xi / P_c,
            "T": T_xi / T_c,
            "R": R_xi,
        }
    )
    df_scaled.set_index("R")
    sns.lineplot(pd.melt(df_scaled, ["R"]), x="R", y="value", hue="variable")
    plt.savefig("out.png")
    plt.cla()
    df = pd.DataFrame(
        {
            "xi": xi,
            "dthetadxi": dthetadxi,
            "theta": theta,
            "rho": rho_xi,
            "M": M_xi,
            "P": P_xi,
            "T": T_xi,
            "R": R_xi,
        }
    )
    df.to_csv("./sim_LE.csv")
    df.set_index("R")
    sns.lineplot(df, x="xi", y="theta")
    plt.savefig("xitheta.png")
    plt.cla()
    sns.lineplot(df, x="R", y="rho")
    plt.savefig("RRho.png")
    plt.cla()
    sns.lineplot(df, x="R", y="M")
    plt.savefig("RM.png")
    plt.cla()
    sns.lineplot(df, x="R", y="P")
    plt.savefig("RP.png")
    plt.cla()
    sns.lineplot(df, x="R", y="T")
    plt.savefig("RT.png")
    plt.cla()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Star Polytrope",
        description="Models a star with the Lane-Emden Equation. By default this uses the mass, radius, and composition of the sun with a polytropic index of 3",
        epilog="by Ahmad Izzuddin",
    )
    parser.add_argument(
        "-M",
        help="Total mass of the star in units of mass of the sun (default %(default)s MSUN)",
        dest="M",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "-R",
        help="Total radius of the star in units of radius of the sun (default %(default)s RSUN)",
        dest="R",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "-X",
        help="Hydrogen mass fraction of the star (default %(default)s)",
        dest="X",
        default=0.73,
        type=float,
    )
    parser.add_argument(
        "-Y",
        help="Helium mass fraction of the star (default %(default)s)",
        dest="Y",
        default=0.26,
        type=float,
    )
    parser.add_argument(
        "-Z",
        help="Metalic mass fraction of the star (default %(default)s)",
        dest="Z",
        default=0.01,
        type=float,
    )
    parser.add_argument(
        "-n", help="Polytropic index (default %(default)s)", dest="n", default=3.0, type=float
    )
    parser.add_argument(
        "-tol",
        help="Tolerance between model radius approximation and actual radius for a given polytropic index (default %(default)s)",
        dest="tol",
        default=1.0e-13,
        type=float,
    )
    parser.add_argument(
        "-h0",
        help="Initial integration step size for RK4 (default %(default)s)",
        dest="h0",
        default=1.0e-5,
        type=float,
    )
    parser.add_argument(
        "-MAXITER",
        help="Maximum number of steps for RK4 integration (default %(default)s)",
        dest="MAX_ITER",
        default=10000000000,
        type=int,
    )
    args = parser.parse_args()
    run(args.n, args.M*MSUN, args.R*RSUN, args.X, args.Y, args.Z, args.MAX_ITER)
