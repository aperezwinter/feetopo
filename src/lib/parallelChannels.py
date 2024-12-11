import math
import numpy as np
import matplotlib.ticker
import matplotlib.pyplot as plt
from typing import Tuple, Any
from scipy.optimize import fsolve, root
from scipy.integrate import dblquad, quad, simps
from scipy import interpolate
from pyXSteam.XSteam import XSteam


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(
            self, useOffset=offset, useMathText=mathText
        )

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r"$\mathdefault{%s}$" % self.format


# Configure units
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)  # m/kg/sec/°C/bar/W


class parallelChannels(object):
    def __init__(
        self,
        num: int,
        Dh: Tuple[float, list, np.ndarray],
        area: Tuple[float, list, np.ndarray],
        height: float = 1,
    ) -> None:

        # Define main variables.
        self.g = 9.81  # gravity acceleration in [m/s²]
        self.N = num  # channel's number
        self.L = height  # channel's length
        self.dp = 0  # delta pressure in [Pa]
        self.mn = np.zeros(num)  # channel's mass flow in [kg/s]
        self.z = np.linspace(0, self.L, num=25, endpoint=True)

        # Set hydraulic diameter.
        if isinstance(Dh, float):
            self.Dh = Dh * np.ones(num, dtype=float)
        elif isinstance(Dh, list):
            if len(Dh) != num:
                raise Exception(
                    f"Size mismatch! Dh must be of size {num} instead of {len(Dh)}."
                )
            else:
                self.Dh = np.array(Dh, dtype=float)
        elif isinstance(Dh, np.ndarray):
            if Dh.size != num:
                raise Exception(
                    f"Size mismatch! Dh must be of size {num} instead of {Dh.size}."
                )
            else:
                self.Dh = Dh
        else:
            self.Dh = None

        # Set mask depends on where Dh = -1
        self.mask = np.where(self.Dh == -1, np.zeros(num), np.ones(num))
        # Refresh Dh variable. Change Dh from -1 to 1, for calculus purpose.
        self.Dh = np.where(self.Dh == -1, np.ones(num), self.Dh)

        # Set cross section area.
        if isinstance(area, float):
            self.A = area * np.ones(num, dtype=float)
        elif isinstance(area, list):
            if len(area) != num:
                raise Exception(
                    f"Size mismatch. Dh must be of size {num} instead of {len(area)}!"
                )
            else:
                self.A = np.array(area, dtype=float)
        elif isinstance(area, np.ndarray):
            if area.size != num:
                raise Exception(
                    f"Size mismatch. Dh must be of size {num} instead of {area.size}!"
                )
            else:
                self.A = area
        else:
            self.A = None

    def __del__(self) -> None:
        pass

    def setConstraints(self, m: float, p_m: float, T_m: float, k: float = 1) -> None:
        self.m = m  # total mass flow in [kg/s]
        self.p_m = p_m  # mean pressure in [bar]
        self.T_m = T_m  # mean temperature in [°C]
        self.k = k  # thermal conductivity in [W/mK]
        self.Cp = 1000 * steamTable.Cp_pt(p_m, T_m)  # heat capacity in [J/Kg°C]
        self.mu = steamTable.my_pt(p_m, T_m)  # dynamic viscosity in [Pa.s]
        self.rho_m = steamTable.rho_pt(p_m, T_m)  # mean density in [kg/m³]
        self.Pr = self.mu * self.Cp / self.k  # Prandtl number

    def setInletTemperature(self, T_in: Tuple[float, list, np.ndarray]) -> None:
        if isinstance(T_in, float):
            self.T_in = T_in * np.ones(self.N, dtype=float)
        elif isinstance(T_in, list):
            if len(T_in) != self.N:
                raise Exception(
                    f"Size mismatch. Dh must be of size {self.N} instead of {len(T_in)}!"
                )
            else:
                self.T_in = np.array(T_in, dtype=float)
        elif isinstance(T_in, np.ndarray):
            if T_in.size != self.N:
                raise Exception(
                    f"Size mismatch. Dh must be of size {self.N} instead of {T_in.size}!"
                )
            else:
                self.T_in = T_in
        # Set inlet density from inlet temperature
        self.rho_in = self.rho_m * (1 - self.beta * (self.T_in - self.T_m))

    def setLocalFrictionFactor(self, K: Tuple[float, list, np.ndarray]) -> None:
        if isinstance(K, float):
            self.K = K * np.ones(self.N, dtype=float)
        elif isinstance(K, list):
            if len(K) != self.N:
                raise Exception(
                    f"Size mismatch. Dh must be of size {self.N} instead of {len(K)}!"
                )
            else:
                self.K = np.array(K, dtype=float)
        elif isinstance(K, np.ndarray):
            if K.size != self.N:
                raise Exception(
                    f"Size mismatch. Dh must be of size {self.N} instead of {K.size}!"
                )
            else:
                self.K = K

    def setBeta(self, beta: Tuple[float, list, np.ndarray]) -> None:
        if isinstance(beta, float):
            self.beta = beta * np.ones(self.N, dtype=float)
        elif isinstance(beta, list):
            if len(beta) != self.N:
                raise Exception(
                    f"Size mismatch. Dh must be of size {self.N} instead of {len(beta)}!"
                )
            else:
                self.beta = np.array(beta, dtype=float)
        elif isinstance(beta, np.ndarray):
            if beta.size != self.N:
                raise Exception(
                    f"Size mismatch. Dh must be of size {self.N} instead of {beta.size}!"
                )
            else:
                self.beta = beta

    def compDarcyFactor(self, Re: float) -> float:
        return (-1.8 * np.log10(6.9 / Re)) ** (-2)

    def compConstants(self) -> None:
        self.constants = []
        # Compute: rho_ref*(1-beta*(Tin-Tref))
        self.constants.append(self.rho_m * (1 - self.beta * (self.T_in - self.T_m)))
        # Compute: rho_ref*beta/Cp
        self.constants.append(self.rho_m * self.beta / self.Cp)
        # Compute: K/(2*A²)
        self.constants.append(self.K / (2 * self.A**2))
        # Compute: Dh/(mu*A)
        self.constants.append(self.Dh / (self.mu * self.A))
        # Compute: L/(2*Dh*A²)
        self.constants.append(self.L / (2 * self.Dh * self.A**2))
        # Compute: 1/A²
        self.constants.append(1 / self.A**2)

    def compProfileFactor(self, num_z: int, qn: list) -> None:
        z = np.linspace(0, self.L, num_z, endpoint=True)
        self.qn = [interpolate.interp1d(z, qn_i) for qn_i in qn]
        self.Qn_func, self.Qn, self.Sn = [], [], []
        for qn_i in self.qn:
            Qn_z = np.array(
                [0] + [simps(qn_i(z[: i + 1]), z[: i + 1]) for i in range(1, num_z)]
            )
            Qn_func_z = interpolate.interp1d(z, Qn_z)
            self.Qn_func.append(Qn_func_z)
            Qn = Qn_func_z(self.L)
            self.Qn.append(Qn)
            self.Sn.append(simps(Qn_func_z(z), z) / (Qn * self.L))
        self.Qn = np.array(self.Qn)
        self.Sn = np.array(self.Sn)

        """
        z = np.linspace(0, self.L, num_z, endpoint=True)
        self.qn = [interpolate.interp1d(z, qn_i) for qn_i in qn]
        self.Sn_func = [lambda z, z_hat: float(qn_func(z_hat)) for qn_func in self.qn]
        # Compute Qn, Sn factors
        self.Qn, self.Sn = [], []
        for qn_i, Sn_func in zip(self.qn, self.Sn_func):
            # Qn, _ = quad(qn_i, 0, self.L)
            Qn = simps(qn_i(z), z)
            Sn, _ = dblquad(Sn_func, 0, self.L, 0, lambda z: z)
            self.Qn.append(Qn)
            self.Sn.append(Sn / (Qn * self.L))
        self.Qn = np.array(self.Qn)
        self.Sn = np.array(self.Sn)
        """

    def compAcumProfile(self, z: Tuple[int, float]) -> np.ndarray:
        # z_vec = np.linspace(0, z, endpoint=True)
        # qn_z = [qn_i(z_vec) for qn_i in self.qn]
        # return np.array([simps(qn_z_i, z_vec) for qn_z_i in qn_z])
        return np.array([Qn_func(z) for Qn_func in self.Qn_func])

        # return np.array([quad(qn_i, 0, z)[0] for qn_i in self.qn])

    def compDensity(self, z: float = 0) -> np.ndarray:
        return self.constants[0] - self.constants[1] * self.compAcumProfile(z) / self.mn

    def compDensityByChannel(self, id_ch: int = 0, z: float = 0) -> float:
        Qn_z = quad(self.qn[id_ch], 0, z)[0]
        rho_z = -self.constants[1][id_ch] * Qn_z / self.mn[id_ch]
        rho_z += self.constants[0][id_ch]
        return rho_z

    def compOutDensity(self) -> np.ndarray:
        return self.constants[0] - self.constants[1] * self.Qn / self.mn

    def compAverageDensity(self) -> np.ndarray:
        return self.constants[0] - self.constants[1] * self.Sn * self.Qn / self.mn

    def compAverageInvDensity(self) -> np.ndarray:
        InvRho_z = np.array([1 / self.compDensity(zi) for zi in self.z]).T
        InvRho = np.array([self.L / simps(InvRho_ch, self.z) for InvRho_ch in InvRho_z])

        return InvRho

    def compMass(self) -> float:
        return self.m - np.sum(self.mn)

    def compMomentum(self) -> np.ndarray:
        rho_out = self.compOutDensity()
        rho_avg = self.compAverageDensity()
        rho_avg_inv = self.compAverageInvDensity()
        self.Re = np.abs(self.mn) * self.constants[3]
        f = self.compDarcyFactor(self.Re)

        # Compute buoyant term
        buoyant = self.g * self.L * rho_avg
        # Compute shear stress friction term
        ssFriction = self.mn * np.abs(self.mn)
        ssFriction *= self.mask * self.constants[4] * f / rho_avg_inv
        # Compute local friction term
        localFriction = self.mn * np.abs(self.mn)
        localFriction *= self.constants[2] / rho_avg
        # Compute acceleration term
        acceleration = (
            (1 / rho_out - 1 / self.rho_in) * self.mn**2 * self.constants[5]
        )

        return -self.dp + buoyant + ssFriction + localFriction + acceleration

    def compSystem(self, x) -> np.ndarray:
        self.dp = x[0]
        self.mn = x[1:]
        nullMass = np.array([self.compMass()])
        nullMomentum = self.compMomentum()

        return np.concatenate((nullMass, nullMomentum), axis=0)

    def compSimplifySystemJacobian(self, x) -> np.ndarray:
        # Approximation: Inverse average density as
        # constant against mass flow.
        mn = x[1:]
        rho_out = self.compOutDensity()
        rho_avg = self.compAverageDensity()
        rho_avg_inv = self.compAverageInvDensity()
        Re = np.abs(mn) * self.constants[3]
        f = self.compDarcyFactor(Re)

        dRhoOut_dmn = self.constants[1] * self.Qn / mn**2
        dRhoAvg_dmn = self.constants[1] * self.Sn * self.Qn / mn**2
        dfn_dRe = 2 / (1.8**2 * math.log(10) * Re) * (np.log10(6.9 / Re) ** (-3))
        dRe_dmn = self.constants[3] * mn / np.abs(mn)
        dfn_dmn = dfn_dRe * dRe_dmn

        # Compute partial differential dFn_dmn by terms:
        # Term 1: Buoyant term
        buoyant = self.g * self.L * dRhoAvg_dmn
        # Term 2: Shear stress friction factor
        ssFriction = self.mask * self.constants[4]
        ssFriction *= np.abs(mn) / rho_avg_inv
        ssFriction *= mn * dfn_dmn + 2 * f
        # Term 3: Local friction factor
        localFriction = np.abs(mn) * (2 - mn * dRhoAvg_dmn / rho_avg) / rho_avg
        localFriction *= self.constants[2]
        # Term 4: Acceleration
        acceleration = mn * self.constants[5]
        acceleration *= (
            2 * (1 / rho_out - 1 / self.rho_in) - mn * dRhoOut_dmn / rho_out**2
        )

        dF_dDP = np.array([-1])
        dF_dmn = buoyant + ssFriction + localFriction + acceleration

        gradF = np.concatenate((dF_dDP, dF_dmn), axis=0)
        return np.diag(gradF)

    def compSystemJacobian(self, x) -> np.ndarray:
        mn = x[1:]
        rho_in = self.compInDensity()
        rho_out = self.compOutDensity()
        rho_avg = self.compAverageDensity()
        rho_avg_inv = self.compAverageInvDensity()
        Re = np.abs(mn) * self.Dh / (self.mu * self.A)
        f = self.compDarcyFactor(Re)

        g_z_zhat = [
            lambda z, z_hat: float(
                self.compDensityByChannel(id_ch, z) ** (-2) * qn_z(z_hat)
            )
            for id_ch, qn_z in enumerate(self.qn)
        ]
        int_qnRho = np.array(
            [dblquad(gn, 0, self.L, 0, lambda z: z)[0] for gn in g_z_zhat]
        )

        dRhoOut_dmn = self.rho_m * self.beta * self.Qn / (self.Cp * mn**2)
        dRhoAvg_dmn = self.rho_m * self.beta * self.Sn * self.Qn / (self.Cp * mn**2)
        dfn_dRe = 2 / (1.8**2 * math.log(10) * Re) * (np.log10(6.9 / Re) ** (-3))
        dRe_dmn = mn * self.Dh / (self.mu * self.A * np.abs(mn))
        dfn_dmn = dfn_dRe * dRe_dmn
        dRhoAvgInv_dmn = (
            self.beta * self.rho_m * rho_avg_inv**2 / (self.Cp * self.L * mn**2)
        ) * int_qnRho

        # Compute partial diferential dFn_dmn by terms:
        # Term 1: Buoyant term
        buoyant = self.g * self.L * dRhoAvg_dmn
        # Term 2: Distribiut friction factor
        distFrict = self.L / (2 * self.Dh * self.A**2)
        distFrict *= np.abs(mn) / rho_avg_inv
        distFrict *= mn * (dfn_dmn - f * dRhoAvgInv_dmn / rho_avg_inv) + 2 * f
        distFrict *= self.mask
        # Term 3: Local friction factor
        localFrict = self.K / (2 * self.A**2)
        localFrict *= np.abs(mn) * (2 - mn * dRhoAvg_dmn / rho_avg) / rho_avg
        # Term 4: Acceleration
        acceleration = mn / self.A**2
        acceleration *= 2 * (1 / rho_out - 1 / rho_in) - mn * dRhoOut_dmn / rho_out**2

        dF_dDP = np.array([-1])
        dF_dmn = buoyant + distFrict + localFrict + acceleration

        gradF = np.concatenate((dF_dDP, dF_dmn), axis=0)
        return np.diag(gradF)

    def compTemp(self, z) -> float:
        # Qn_z = [quad(qn_i, 0, z)[0] for qn_i in self.qn]
        Qn_z = self.compAcumProfile(z)
        return self.T_in + Qn_z / (self.Cp * self.mn)

    def compConvectionCoeff(self) -> float:
        return self.k * 0.023 * (self.Re**0.8) * (self.Pr**0.4) / self.Dh

    def solve(
        self,
        dp_0: float = 0,
        m_0: Tuple[float, list, np.ndarray] = 0,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:

        # Check initial flow mass.
        if isinstance(m_0, float):
            m_0 = m_0 * np.ones(self.N, dtype=float)
        elif isinstance(m_0, list):
            if len(m_0) != self.N:
                raise Exception(
                    f"Size mismatch. Dh must be of size {self.N} instead of {len(m_0)}!"
                )
            else:
                m_0 = np.array(m_0, dtype=float)
        elif isinstance(m_0, np.ndarray):
            if m_0.size != self.N:
                raise Exception(
                    f"Size mismatch. Dh must be of size {self.N} instead of {m_0.size}!"
                )

        guess = np.concatenate((np.array([dp_0]), m_0), axis=0)
        sol = root(
            self.compSystem,
            guess,
            jac=self.compSimplifySystemJacobian,
            tol=tol,
            method="hybr",  # 'lm'
        )
        if not sol.success:
            raise Exception(
                f"Solution not found! {sol.message} with {sol.nfev} evaluations."
            )

        # Save solution.
        self.dp = sol.x[0]
        self.mn = sol.x[1:]