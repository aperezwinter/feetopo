import numpy as np
from math import pi
from pyXSteam.XSteam import XSteam
from scipy.optimize import root
from scipy.integrate import simpson

# Configure units
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)  # m/kg/sec/°C/bar/W

class Channel(object):

    g = 9.81  # gravity acceleration [m/s²]

    @staticmethod
    def darcyFactor(Re: float=1e5):
        return (-1.8 * np.log10(6.9 / Re)) ** (-2)
    
    @staticmethod
    def darcyFactorDerivative(Re: float=1e5):
        return -3.6 * Channel.darcyFactor(Re)**1.5 / Re

    def __init__(self, dh: float, area: float, height: float=1, nz: int=20):
        self.dh = dh                            # hidraulic diameter [m]
        self.area = area                        # area [m²]
        self.height = height                    # height [m]
        self.nz = nz                            # number of points in the z direction
        self.z = np.linspace(0, height, nz)     # z-grid [m]

    def setMeanProperties(self, Tm: float, pm: float) -> None:
        self.Tm = Tm                                # mean temperature [°C]
        self.pm = pm                                # mean pressure [bar]
        self.cp = 1000 * steamTable.Cp_pt(pm, Tm)   # specific heat [J/kgK]
        self.mu = steamTable.my_pt(pm, Tm)          # dynamic viscosity [Pa.s]
        self.rhom = steamTable.rho_pt(pm, Tm)       # mean density [kg/m³]
    
    def setProperties(self, beta: float, K: float, k: float=1) -> None:
        self.beta = beta                            # thermal expansion coefficient [1/°C]
        self.K = K                                  # local friction factor [~]
        self.k = k                                  # thermal conductivity [W/mK]
        self.Pr = self.cp * self.mu / k             # Prandtl number

        # Compute some constants derived from the properties and initial variables
        self.constants = [
            self.rhom * self.beta / self.cp,
            self.K / (2 * self.area**2),
            self.dh / (self.mu * self.area),
            self.height / (2 * self.dh * self.area**2),
            self.area**(-2),
        ]

    def setInletDensity(self, Tin: float=25) -> None:
        # Use mean density as reference density
        self.Tin = Tin
        self.rhoi = self.rhom * (1 - self.beta * (Tin - self.Tm))

    def computeOuletTemperature(self, qz: np.ndarray, m: float=1) -> float:
        # Compute the outlet temperature Tout = Tin + ∫_[0,L] qz dz / (m * cp)
        # where qz is the heat flux profile [W/m], m is the mass flow rate [kg/s]
        return self.Tin + simpson(qz, x=self.z) / (m * self.cp)
    
    def computeOutletDensity(self, qz: np.ndarray, m: float=1) -> float:
        # Compute the outlet density ρout = ρ_ref * (1 - β * (Tout - T_ref))
        # where qz is the heat flux profile [W/m], m is the mass flow rate [kg/s]
        Tout = self.computeOuletTemperature(qz, m)
        return self.rhom * (1 - self.beta * (Tout - self.Tm))

    def computeGeometricFactor(self, qz: np.ndarray) -> float:
        # Compute the geometric factor S = 1/(Q*L) ∫_[0,L]∫_[0,z] qz' dz' dz
        # where qz is the heat flux profile [W/m] and Q is the total heat flow [W]
        first_int = np.array([simpson(qz[:i], x=self.z[:i]) for i in range(1, self.nz+1)])
        second_int = simpson(first_int, x=self.z)
        return second_int / (first_int[-1] * self.height)
    
    def computeTemperatureProfile(self, qz: np.ndarray, m: float=1) -> np.ndarray:
        # Compute the temperature profile Tz = Tin + ∫_[0,z] qz' dz' / (m * cp)
        # where qz is the heat flux profile [W/m] and m is the mass flow rate [kg/s]
        first_int = np.array([simpson(qz[:i], x=self.z[:i]) for i in range(1, self.nz+1)])
        return self.Tin + first_int / (m * self.cp)
    
    def computeDensityProfile(self, qz: np.ndarray, m: float=1) -> np.ndarray:
        # Compute the density profile ρz = ρ_ref * (1 - β * (Tz - T_ref))
        # where qz is the heat flux profile [W/m], m is the mass flow rate [kg/s]
        Tz = self.computeTemperatureProfile(qz, m)
        return self.rhom * (1 - self.beta * (Tz - self.Tm))
    
    def computeAverageDensity(self, qz: np.ndarray, m: float=1) -> float:
        # Compute the average density ρ = ∫_[0,L] ρz dz / L
        # where qz is the heat flux profile [W/m], m is the mass flow rate [kg/s]
        rhoz = self.computeDensityProfile(qz, m)
        return simpson(rhoz, x=self.z) / self.height
    
    def computeInverseAverageDensity(self, qz: np.ndarray, m: float=1) -> float:
        # Compute the inverse of the average density ρ = L / ∫_[0,L] 1/ρz dz
        # where qz is the heat flux profile [W/m], m is the mass flow rate [kg/s]
        rhoz = self.computeDensityProfile(qz, m)
        return self.height / simpson(1 / rhoz, x=self.z)
    
    def computePower(self, qz: np.ndarray) -> float:
        # Compute the power P = ∫_[0,L] qz dz
        return simpson(qz, x=self.z)
    
    def computeJacobian(self, qz: np.ndarray, dp: float=0, m: float=1, friction: bool=True) -> tuple[float, float]:
        # Compute the jacobian matrix of the momentum balance equation J = (∂F/∂dp, ∂F/∂m)
        # where F is the momentum balance equation, qz is the heat flux profile [W/m]
        # m is the mass flow rate [kg/s] and dp is the pressure drop [Pa]
        Q = self.computePower(qz)
        S = self.computeGeometricFactor(qz)
        rhoo = self.computeOutletDensity(qz, m)
        rho_avg = self.computeAverageDensity(qz, m)
        rho_avg_inv = self.computeInverseAverageDensity(qz, m)
        drhoo_dm = self.constants[0] * Q / m**2
        drho_avg_dm = self.constants[0] * Q * S / m**2

        bouyant = Channel.g * self.height * drho_avg_dm
        local_friction = self.constants[1] * abs(m) * (2 - m*drho_avg_dm/rho_avg) / rho_avg
        local_acceleration = self.constants[4] * m * (2*(1/rhoo - 1/self.rhoi) - m*drhoo_dm/rhoo**2)

        if friction:
            Re = self.constants[2] * abs(m)
            Darcy = Channel.darcyFactor(Re)
            df_dRe = Channel.darcyFactorDerivative(Re)
            df_dm = df_dRe * self.constants[2] * np.sign(m)
            ss_friction = self.constants[3] * abs(m) * (m*df_dm + 2*Darcy) / rho_avg_inv
            dF_dm = bouyant + ss_friction + local_friction + local_acceleration
        else:
            dF_dm = bouyant + local_friction + local_acceleration
        
        dF_ddp = -1

        return dF_ddp, dF_dm
        
    def computeMomentumBalance(self, qz: np.ndarray, m: float=1, dp: float=0, friction: bool=True) -> float:
        # Compute the momentum balance equation, returns the residual
        # F(dp,m) = -dp + bouyant + shear_stress_friction + local_friction + local_acceleration
        # where dp is the pressure drop [Pa], m is the mass flow rate [kg/s]
        # and qz is the heat flux profile [W/m].
        rhoo = self.computeOutletDensity(qz, m)
        rho_avg = self.computeAverageDensity(qz, m)
        rho_avg_inv = self.computeInverseAverageDensity(qz, m)
        
        bouyant = rho_avg * Channel.g * self.height
        local_friction = m**2 * self.constants[1] / rho_avg
        local_acceleration = m**2 * self.constants[4] * (1/rhoo - 1/self.rhoi)

        if friction:
            Re = self.constants[2] * abs(m)
            Darcy = Channel.darcyFactor(Re)
            ss_friction = m**2 * self.constants[3] * Darcy /rho_avg_inv
            return -dp + bouyant + ss_friction + local_friction + local_acceleration
        else:
            return -dp + bouyant + local_friction + local_acceleration
    
    def computeHeatConvectionCoeff(self, m: float=1) -> float:
        # Compute the heat convection coefficient h, based on Dittus-Boelter equation
        # h = 0.023 * Re^0.8 * Pr^0.4 * k / dh
        Re = self.constants[2] * m
        return 0.023 * Re**0.8 * self.Pr**0.4 * self.k / self.dh
    

class Plenum(object):

    def __init__(self, m: float, num_ch: int, dh: tuple[float, list], 
                 area: tuple[float, list], height: float=1, nz : int=20):
        
        # Fill in the missing data
        dh = [dh] if isinstance(dh, float) else dh
        area = [area] if isinstance(area, float) else area
        dh = dh + [dh[-1]]*(num_ch - len(dh)) if len(dh) < num_ch else dh
        area = area + [area[-1]]*(num_ch - len(area)) if len(area) < num_ch else area
        ss_friction_mask = np.where(np.array(dh) > 0, True, False)
        dh = np.where(np.array(dh) > 0, np.array(dh), np.ones(num_ch)*1e-2)
        
        self.m = m              # total mass flow rate [kg/s]
        self.num_ch = num_ch    # number of channels
        self.dh = dh            # hydraulic diameter [m]
        self.area = area        # cross-sectional area [m^2]
        self.height = height    # height of the plenum [m]
        self.nz = nz            # points in the z-direction

        self.ss_friction_mask = ss_friction_mask
        self.channels = [Channel(dh[i], area[i], height, nz) for i in range(num_ch)]

    def setMeanProperties(self, Tm: float, pm: float) -> None:
        for i in range(self.num_ch):
            self.channels[i].setMeanProperties(Tm, pm)

    def setProperties(self, beta: tuple[float, list], K: tuple[float, list], k: tuple[float, list]=1.) -> None:
        # Fill in the missing data
        beta = [beta] if isinstance(beta, float) else beta
        K = [K] if isinstance(K, float) else K
        k = [k] if isinstance(k, float) else k
        beta = beta + [beta[-1]]*(self.num_ch - len(beta)) if len(beta) < self.num_ch else beta
        K = K + [K[-1]]*(self.num_ch - len(K)) if len(K) < self.num_ch else K
        k = k + [k[-1]]*(self.num_ch - len(k)) if len(k) < self.num_ch else k

        # Set the properties for each channel
        for i in range(self.num_ch):
            self.channels[i].setProperties(beta[i], K[i], k[i])

    def setInletDensity(self, Ti: tuple[float, list]=25.) -> None:
        # Fill in the missing data
        Ti = [Ti] if isinstance(Ti, float) else Ti
        Ti = Ti + [Ti[-1]]*(self.num_ch - len(Ti)) if len(Ti) < self.num_ch else Ti
        # Set the inlet density for each channel
        for i in range(self.num_ch):
            self.channels[i].setInletDensity(Ti[i])

    def computeOutletTemperature(self, qz: list, m: list) -> np.ndarray:
        # Fill in the missing data
        qz = [qz] if isinstance(qz, np.ndarray) else qz
        m = [m] if isinstance(m, float) else m
        qz = qz + [qz[-1]]*(self.num_ch - len(qz)) if len(qz) < self.num_ch else qz
        m = m + [m[-1]]*(self.num_ch - len(m)) if len(m) < self.num_ch else m
        # Compute the outlet temperature for each channel
        return np.array([ch.computeOuletTemperature(qz_ch, m_ch) for ch, qz_ch, m_ch in zip(self.channels, qz, m)])

    def computeOutletDensity(self, qz: list, m: list) -> np.ndarray:
        # Fill in the missing data
        qz = [qz] if isinstance(qz, np.ndarray) else qz
        m = [m] if isinstance(m, float) else m
        qz = qz + [qz[-1]]*(self.num_ch - len(qz)) if len(qz) < self.num_ch else qz
        m = m + [m[-1]]*(self.num_ch - len(m)) if len(m) < self.num_ch else m
        # Compute the outlet density for each channel
        return np.array([ch.computeOutletDensity(qz_ch, m_ch) for ch, qz_ch, m_ch in zip(self.channels, qz, m)])
    
    def computeTemperatureProfile(self, qz: list, m: list) -> np.ndarray:
        # Fill in the missing data
        qz = [qz] if isinstance(qz, np.ndarray) else qz
        m = [m] if isinstance(m, float) else m
        qz = qz + [qz[-1]]*(self.num_ch - len(qz)) if len(qz) < self.num_ch else qz
        m = m + [m[-1]]*(self.num_ch - len(m)) if len(m) < self.num_ch else m
        # Compute the temperature profile for each channel
        return np.array([ch.computeTemperatureProfile(qz_ch, m_ch) for ch, qz_ch, m_ch in zip(self.channels, qz, m)])

    def computeDensityProfile(self, qz: list, m: list) -> np.ndarray:
        # Fill in the missing data
        qz = [qz] if isinstance(qz, np.ndarray) else qz
        m = [m] if isinstance(m, float) else m
        qz = qz + [qz[-1]]*(self.num_ch - len(qz)) if len(qz) < self.num_ch else qz
        m = m + [m[-1]]*(self.num_ch - len(m)) if len(m) < self.num_ch else m
        # Compute the density profile for each channel
        return np.array([ch.computeDensityProfile(qz_ch, m_ch) for ch, qz_ch, m_ch in zip(self.channels, qz, m)])
    
    def computeHeatConvectionCoeff(self, m: list) -> np.ndarray:
        # Fill in the missing data
        m = [m] if isinstance(m, float) else m
        m = m + [m[-1]]*(self.num_ch - len(m)) if len(m) < self.num_ch else m
        # Compute the heat convection coefficient for each channel
        return np.array([ch.computeHeatConvectionCoeff(m_ch) for ch, m_ch in zip(self.channels, m)])

    def computeMassBalance(self, m: list) -> float:
        # Fill in the missing data
        m = [m] if isinstance(m, float) else m
        m = m + [m[-1]]*(self.num_ch - len(m)) if len(m) < self.num_ch else m
        # Compute the mass balance
        return self.m - sum(m)

    def computeMomentumBalance(self, qz: list, m: list, dp: float=0) -> np.ndarray:
        # Fill in the missing data
        qz = [qz] if isinstance(qz, np.ndarray) else qz
        m = [m] if isinstance(m, float) else m
        qz = qz + [qz[-1]]*(self.num_ch - len(qz)) if len(qz) < self.num_ch else qz
        m = m + [m[-1]]*(self.num_ch - len(m)) if len(m) < self.num_ch else m
        # Compute the momentum balance for each channel
        momentum = np.array([ch.computeMomentumBalance(qz[i], m[i], dp, self.ss_friction_mask[i]) 
                             for i, ch in enumerate(self.channels)])
        return momentum
        
    def computeJacobian(self, x, qz: list) -> np.ndarray:
        # dp = x[0], m = x[1:] 
        # Fill in the missing data
        qz = [qz] if isinstance(qz, np.ndarray) else qz
        qz = qz + [qz[-1]]*(self.num_ch - len(qz)) if len(qz) < self.num_ch else qz
        # Compute the jacobian for each channel
        mom_jacobian = np.array([ch.computeJacobian(qz[i], x[0], x[i+1], self.ss_friction_mask[i]) 
                             for i, ch in enumerate(self.channels)])
        mom_jacobian = np.hstack((mom_jacobian[:,0].reshape(-1,1), np.diag(mom_jacobian[:,1])))
        # Build the jacobian matrix
        mass_jacobian = np.array([[0] + [-1]*self.num_ch])
        jacobian = np.vstack((mass_jacobian, mom_jacobian))
        return jacobian
    
    def computeResidual(self, x, qz: list) -> np.ndarray:
        # Fill in the missing data
        qz = [qz] if isinstance(qz, np.ndarray) else qz
        qz = qz + [qz[-1]]*(self.num_ch - len(qz)) if len(qz) < self.num_ch else qz
        # Compute the residual of equations for F(dp,m) = 0
        mass_balance = self.computeMassBalance(x[1:])
        momentum_balance = self.computeMomentumBalance(qz, x[1:], x[0])
        residual = np.hstack((mass_balance, momentum_balance))
        return residual
    
    def solveSystem(self, qz: list, dp0: float=0, m0: tuple[float, list]=0, max_iter: int=100, tol: float=1e-6) -> np.ndarray:
        # Fill in the missing data
        qz = [qz] if isinstance(qz, np.ndarray) else qz
        m0 = [m0] if isinstance(m0, float) else m0
        qz = qz + [qz[-1]]*(self.num_ch - len(qz)) if len(qz) < self.num_ch else qz
        m0 = m0 + [m0[-1]]*(self.num_ch - len(m0)) if len(m0) < self.num_ch else m0
        # Initial guess
        x0 = [dp0] + m0
        # Solve the system of equations
        func = self.computeResidual
        jac = self.computeJacobian
        sol = root(func, x0, args=(qz,), method='hybr', jac=jac, options={'xtol': tol})
        return sol.x