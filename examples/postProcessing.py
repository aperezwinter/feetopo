import sys, subprocess
import gmsh, random
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress
from operator import itemgetter
from scipy.integrate import simps
from scipy.interpolate import griddata
from math import sqrt, cos, sin, pi, isclose

# command to add feenox to the $PATH, run on terminal...
# export PATH=$PATH:/home/alan/Desktop/feenox/bin

src_path = "/home/alan/Desktop/MI_IB/thesis_code-master/"
if src_path not in sys.path:
    sys.path.append(src_path)

import src.lib.postProcessing as pp


def main():
    # Set file path
    filePath = "examples/reflector/volumeHeatFlux/tol_d/14mm/result.dat"
    filePath = src_path + filePath

    # Extract data
    data = pp.process(filePath)
    print(data.keys())
    x = np.arange(data["numH"])
    x_plus = np.arange(data["numH"] + 1)
    cf = 100 * data["cf"] / data["cf"][0]
    mc = data["mC"]
    mr = data["mH"]
    dp = data["dp"]
    Qratio = 100 * np.array(data["Qhole"]) / np.array(data["Qsrc"])
    Tmax = np.array([T[1] for T in data["maxT"]])
    mCmax = 425 * np.ones(data["mC"].size)
    mCmin = 410 * np.ones(data["mC"].size)

    # Create figure object
    fig, ax = plt.subplots(3, 2, figsize=(12, 17), dpi=300)

    # Plot cost function
    ax[0][0].plot(x, cf, "s--", color="blue")
    ax[0][0].set_xlabel("# Canales", color="k", fontsize=14)
    ax[0][0].set_ylabel("Función costo [%]", color="k", fontsize=14)
    ax[0][0].set_ylim(0, 105)
    ax[0][0].set_title("(a)", color="k", fontsize=14)
    ax[0][0].grid()

    # Plot core & reflector mass flow
    ax2 = ax[0][1].twinx()
    ax[0][1].plot(x_plus, mc, "s--", color="blue", label="Núcleo")
    ax2.plot(x_plus, mr, "^--", color="orange", label="Reflector")
    ax[0][1].plot(x_plus, mCmax, "-", color="green", linewidth=2.0)
    ax[0][1].plot(x_plus, mCmin, "-", color="red", linewidth=2.0)
    ax[0][1].set_xlabel("# Canales", color="k", fontsize=14)
    ax[0][1].set_ylabel("Flujo másico - Núcleo [kg/s]", color="k", fontsize=14)
    ax2.set_ylabel("Flujo másico - Reflector [kg/s]", color="k", fontsize=14)
    ax[0][1].set_ylim(407, 426)
    ax2.set_ylim(-1, 16)
    ax[0][1].set_title("(b)", color="k", fontsize=14)
    ax[0][1].legend(loc="upper center", fontsize=14)
    ax2.legend(loc="lower center", fontsize=14)
    ax[0][1].grid()

    # Delta pressure
    ax[1][0].plot(x, dp, "o--", color="red")
    ax[1][0].set_xlabel("# Canales", color="k", fontsize=14)
    ax[1][0].set_ylabel("Salto de presión [Pa]", color="k", fontsize=14)
    ax[1][0].set_title("(c)", color="k", fontsize=14)
    ax[1][0].grid()

    # Power dissipated
    ax[1][1].plot(x, Qratio, "o--", color="purple")
    ax[1][1].set_xlabel("# Canales", color="k", fontsize=14)
    ax[1][1].set_ylabel("Qch/Qr [%]", color="k", fontsize=14)
    ax[1][1].set_title("(d)", color="k", fontsize=14)
    ax[1][1].grid()

    # Maximun reflector temperature
    ax[2][0].plot(x, Tmax, "s--", color="green")
    ax[2][0].set_xlabel("# Canales", color="k", fontsize=14)
    ax[2][0].set_ylabel("Temperatura máxima [⁰C]", color="k", fontsize=14)
    ax[2][0].set_title("(e)", color="k", fontsize=14)
    ax[2][0].grid()

    # Core temperature
    trefC_mu = np.mean(np.array(data["TrefC"]), axis=0)
    trefC_std = np.std(np.array(data["TrefC"]), axis=0)
    z = np.linspace(0, 1.8, trefC_mu.size, True)
    ax[2][1].errorbar(
        trefC_mu,
        z,
        xerr=trefC_std,
        c="blue",
        ls="--",
        marker="o",
    )
    ax[2][1].set_xlabel("Temperatura Núcleo [⁰C]", color="k", fontsize=14)
    ax[2][1].set_ylabel("z [m]", color="k", fontsize=14)
    ax[2][1].set_title("(f)", color="k", fontsize=14)
    ax[2][1].grid()

    # Save figure
    fig.savefig("data_14mm.png")

    # ------------------------------------------------------ #

    # Create figure object
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Convection coefficient
    c = ["b", "m", "y", "g", "r", "c", "brown", "orange", "purple"]
    mk = ["s", "o", "v", "^", "*", "X", "P", "D", "<", ">"]
    hc_mu = np.array([np.mean(hc) for hc in data["hcH"][:-1]])
    hc_std = np.array([np.std(hc) for hc in data["hcH"][:-1]])
    hc_idx = [1, 3, 6, 9, 12, 15, 18, 21, 24]
    hc = [data["hcH"][idx] for idx in hc_idx]
    ax2 = ax.twinx()
    for i, hc_i in enumerate(hc):
        nch = hc_idx[i] + 1
        x_i = np.arange(nch) + 1
        ax.plot(x_i, hc_i, "--", color=c[i], marker=mk[i], label=f"{nch} canales")
    ax2.errorbar(
        x_i,
        hc_mu,
        yerr=hc_std,
        c="k",
        ls="-",
        marker="o",
        mfc="none",
        label="Promedio",
    )
    ax.set_xticks(x_i)
    ax.set_xlabel("Índice de canal", color="k", fontsize=10)
    ax.set_ylabel("Coeficiente convección [W/m²K]", color="k", fontsize=10)
    ax2.set_ylabel("<hc> [W/m²K]", color="k", fontsize=10)
    ax.set_title("(a)", color="k", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax2.legend(loc="upper center", fontsize=10)
    ax.grid()

    # Save figure
    fig.savefig("hc_14mm.png")

    # Create figure object
    fig, ax = plt.subplots(2, 2, figsize=(10, 12), dpi=300)
    # Reflector temperature
    c = ["b", "m", "y", "g", "r", "c", "brown", "orange", "purple"]
    mk = ["s", "o", "v", "^", "*", "X", "P", "D", "<", ">"]
    tref_idx = [1, 3, 5, 8, 12, 15, 18, 20, 25]
    tref_mu = {k: np.mean(np.array(tref), axis=0) for k, tref in data["TrefH"].items()}
    tref_std = {k: np.std(np.array(tref), axis=0) for k, tref in data["TrefH"].items()}

    ## Plot all cases
    z = np.linspace(0, 1.8, tref_mu[1].size, True)
    for i, idx in enumerate(tref_idx):
        label = f"{idx} Canal" if idx == 1 else f"{idx} Canales"
        ax[0][0].plot(tref_mu[idx], z, c=c[i], ls="--", marker=mk[i], label=label)
    ax[0][0].set_xlim(284, 306)
    ax[0][0].set_xlabel("<T> [⁰C]", color="k", fontsize=14)
    ax[0][0].set_ylabel("z [m]", color="k", fontsize=14)
    ax[0][0].set_title("(a)", color="k", fontsize=16)
    ax[0][0].legend(loc="lower right", fontsize=10)
    ax[0][0].grid()

    ## Plot case: 3 channels
    tref = data["TrefH"][3]
    for i, tref_i in enumerate(tref):
        ax[0][1].plot(tref_i, z, c=c[i], ls="--", marker=mk[i], label=f"Canal {i+1}")
    ax[0][1].set_xlim(284, 306)
    ax[0][1].set_xlabel("Temperatura [⁰C]", color="k", fontsize=14)
    ax[0][1].set_ylabel("z [m]", color="k", fontsize=14)
    ax[0][1].set_title("(b)", color="k", fontsize=16)
    ax[0][1].legend(loc="lower right", fontsize=10)
    ax[0][1].grid()

    ## Plot case: 6 channels
    tref = data["TrefH"][6]
    for i, tref_i in enumerate(tref):
        ax[1][0].plot(tref_i, z, c=c[i], ls="--", marker=mk[i], label=f"Canal {i+1}")
    ax[1][0].set_xlim(284, 306)
    ax[1][0].set_xlabel("Temperatura [⁰C]", color="k", fontsize=14)
    ax[1][0].set_ylabel("z [m]", color="k", fontsize=14)
    ax[1][0].set_title("(c)", color="k", fontsize=16)
    ax[1][0].legend(loc="lower right", fontsize=10)
    ax[1][0].grid()

    ## Plot case: 9 channels
    tref = data["TrefH"][9]
    for i, tref_i in enumerate(tref):
        ax[1][1].plot(tref_i, z, c=c[i], ls="--", marker=mk[i], label=f"Canal {i+1}")
    ax[1][1].set_xlim(284, 306)
    ax[1][1].set_xlabel("Temperatura [⁰C]", color="k", fontsize=14)
    ax[1][1].set_ylabel("z [m]", color="k", fontsize=14)
    ax[1][1].set_title("(d)", color="k", fontsize=16)
    ax[1][1].legend(loc="lower right", fontsize=10)
    ax[1][1].grid()

    # Save figure
    fig.savefig("tref_14mm.png")


# Run main...
if __name__ == "__main__":
    main()
