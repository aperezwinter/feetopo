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

import src.lib.mesh as mesh
import src.lib.parallelChannels as pllCh
import src.lib.reflector as rfl


def compVirtualHolesBC(
    root: str,
    case,
    meshParams: dict,
    diameter: float,
    centers: list,
    qn_centers: list,
    qn_core: np.ndarray,
    qn_holes: list = [],
    holes: list = [],
    dp_0: float = 15000.0,
    mN_0: float = 420.0,
    relax: float = 0.5,
    maxIter: int = 100,
    tol: float = 1e-4,
):
    hc_centers, Tref_centers = [], []
    for rc, qn_rc in zip(centers, qn_centers):
        # Set holes and qn
        holes_virtual = holes + [(rc, diameter)]
        qn_virtual = qn_holes + [qn_rc]
        # Create domain
        carem = rfl.Reflector(
            root=root,
            folderName=f"virtual",
            fileName="virtual",
            height=case.L,
            nz=meshParams["nz"],
            holes=holes_virtual,
            lc=meshParams["lc"],
            lch=meshParams["lch"],
            Ntheta=meshParams["Ntheta"],
            Nr=meshParams["Nr"],
        )
        # Compute hc and Tref
        hc_virtual, Tref_virtual = compHolesBC(
            case=case,
            carem=carem,
            qn_core=qn_core,
            qn_holes=qn_virtual,
            dp_0=dp_0,
            mN_0=mN_0,
            relax=relax,
            maxIter=maxIter,
            tol=tol,
        )
        hc_centers.append(hc_virtual[-1])
        Tref_centers.append(Tref_virtual[-1])
        # Delete/empty virtual folder
        subprocess.run(["rm", "-r", root + "virtual"])

    return hc_centers, Tref_centers


def compHolesBC(
    case,
    carem,
    qn_core: np.ndarray,
    qn_holes: list,
    dp_0: float = 15000,
    mN_0: float = 420,
    relax: float = 0.5,
    maxIter: int = 100,
    tol: float = 1e-4,
) -> list:
    # El reflector viene solamente creado!
    # Los canales paralelos ya vienen iniciados, hasta antes de Qn,SN.
    hc_holes, Tref_holes = [], []
    centers = [hole[0] for hole in carem.holes]
    d = carem.holes[0][1]
    iter, error = 0, 1
    Nz = qn_core.size
    z = np.linspace(0, carem.height, num=Nz, endpoint=True)

    while (error > tol) and iter < maxIter:
        # Prepare qn and m lists
        qn = [qn_core] + qn_holes * 6
        m_0 = [mN_0] + [(case.m - mN_0) / (case.N - 1)] * (case.N - 1)
        # Compute power profile factors
        case.compProfileFactor(Nz, qn)
        # Solve paralallel channels problem
        case.solve(dp_0, m_0)
        # Compute hc for each channel
        hc_holes = case.compConvectionCoeff()
        hc_holes = list(hc_holes[1 : len(qn_holes) + 1])
        # Compute Tref(z) for each channel
        Tref_holes = np.array([case.compTemp(zi) for zi in z]) + 273.15
        Tref_holes = Tref_holes[:, 1 : len(qn_holes) + 1]
        Tref_holes = list(Tref_holes.T)
        # Set BC on FEM problem
        carem.setRobinBC(hc_holes=hc_holes, Tref_holes=Tref_holes, Nz=Nz)
        # Solve direct problem
        carem.writeStateFeenox()
        carem.solveState()
        # Compute power profiles from FEM
        qn_holes_prime, _ = carem.compHeatProfiles(centers, d, Nz)
        # Compute error
        error = max(
            [
                np.linalg.norm(q - q_prime) / np.linalg.norm(q)
                for q, q_prime in zip(qn_holes, qn_holes_prime)
            ]
        )
        # Refresh iterative variables
        iter += 1
        qn_holes = [
            q * relax + q_prime * (1 - relax)
            for q, q_prime in zip(qn_holes, qn_holes_prime)
        ]

    return hc_holes, Tref_holes


def main():
    # Set root path
    root_path = src_path + "examples/reflector/volumeHeatFlux/tol_d/14mm/"

    # Geometric parameters
    H = 1.8  # reflector's height in [m]
    d = 14e-3  # channel's diameter in [m]
    tol = d  # minimun channel's distance in [m]
    A_core = 0.804371  # [m²], core cross section area

    # Numeric variables
    w = 0.05  # relaxation parameter, xn+1=xn (w=0)
    Nz = 15  # discretisation along z axis (q'(z) profile)
    Ntheta = 6  # discretization along theta axis
    fraction = 0.15  # percentage of nodes to filter (q weight)
    filterThold = 10  # max number of virtual holes
    lc_embed = 1e-3  # characteristic length of embed points
    lc_virtual = 5e-3  # characteristic length of virtual holes

    # Optimization variables
    i = 0  # iteration variable, default to 0
    holes = []  # holes[i] = (rc, d)
    flag = True  # loop stop criteria variable
    currVol = 1  # current volume, default to start volume = 1, in [%]
    tholdVol = 0.9  # threshold volume in [%]
    initVol = 0  # initial reflector volume in [m³]
    mN_thold = 410  # threshold mass flow in [kg/s]

    # Thermodynamic variables
    Q = 100e6  # total core power in [W]
    m = 425  # total mass flow in [kg/s]
    P = 123  # operation pressure value in [bar]
    Tm = 308.8  # mean temperature in [°C]
    Ti = 285.2  # inlet temperature in [°C]
    k_water = 0.545  # heat conduction in [W/mK]
    beta_core = 0.0032  # volume expansion coefficient in [1/°C]
    beta_ch = 0.0025  # volume expansion coefficient in [1/°C]
    mN_guess = 420  # initial core mass flow in [kg/s]
    dp_guess = 15000  # initial reflector delta pressure in [Pa]
    qn_core = Q * np.ones(Nz) / H  # core power profile in [W/m]

    # Friction factors
    K_elbow = 0.21  # 90 degrees elbow friction factor in [~]
    K_cont = 0.42  # maximum contraction nuzzle factor in [~]
    K_exp = 1.0  # maximum expansion nuzzle factor in [~]
    K_ch = 2 * K_elbow + K_cont + K_exp  # [1/m⁴], channel's K
    K_core = 13.361  # [1/m⁴], core friction factor per unit area
    K_core *= A_core**2

    # Solution variables
    holes_coord = []  # rc in [m]
    hc_holes_i, Tref_holes_i = [], []  # on step i
    hc_holes = []  # channel's BC hc coefficient
    hc_core = []  # core's BC hc coefficient (inner wall)
    Tref_holes = []  # channel's BC Tref(z) variable
    Tref_core = []  # core's BC Tref(z) (inner wall)
    mN, mR = [m], [0]  # core and by-pass mass flow in [kg/s]
    volume = []  # volume[i] = %vol on i-th iteration
    costFunc = []  # cost function
    powerSrc = []  # volume heat source in [W]
    powerCh = []  # power dissipated on channels in [W]
    deltaP = []  # delta pressure on each step i, in [Pa]
    maxTemp = []  # maximum temperature, (maxT, x,y,z) in [K]

    # Define unenable physical groups
    adjBounds = ["inner", "outer", "top"]
    boundOff = ["inner", "outer", "side", "boundGap"]
    domainOff = ["gap", "barrel"]

    # Mesh parameters
    virtParam = {"lc": 15e-3, "lch": 5e-3, "nz": 5, "Nr": 3, "Ntheta": 50}
    coarseParam = {"lc": 25e-3, "lch": 0.5 * d, "nz": 8, "Nr": 3, "Ntheta": 50}
    refineParam = {"lc": 10e-3, "lch": 0.25 * d, "nz": 20, "Nr": 3, "Ntheta": 75}

    # Save file parameters
    sep = " | "
    headers = {
        "some": ("#FEATURES", "#ENDFEATURES"),
        "dp": ("#DELTAP", "#ENDDELTAP"),
        "hcC": ("#HCCORE", "#ENDHCCORE"),
        "hcH": ("#HCHOLE", "ENDHCHOLE"),
        "TrefC": ("#TREFCORE", "#ENDTREFCORE"),
        "TrefH": ("#TREFHOLE", "#ENDTREFHOLE"),
        "mC": ("#MASSCORE", "#ENDMASSCORE"),
        "mH": ("#MASSHOLE", "#ENDMASSHOLE"),
        "qSrc": ("#POWERSOURCE", "#ENDPOWERSOURCE"),
        "qH": ("#POWERHOLE", "#ENDPOWERHOLE"),
        "vol": ("#VOLUME", "#ENDVOLUME"),
        "fc": ("#COST", "#ENDCOST"),
        "holes": ("#HOLES", "#ENDHOLES"),
        "maxT": ("#MAXT", "#ENDMAXT"),
    }

    # First domain i=0
    # Create dom_i(refine) and solve -D²u=b for step i
    carem_ref = rfl.Reflector(
        root=root_path,
        folderName=f"{i}",
        fileName="design",
        height=H,
        nz=refineParam["nz"],
        holes=holes,
        lc=refineParam["lc"],
        lch=refineParam["lch"],
        Ntheta=refineParam["Ntheta"],
        Nr=refineParam["Nr"],
        form="nonuniform",
    )

    # ------START: Optimization loop------ #
    while flag and (currVol >= tholdVol):
        print(f"INFO:  Step {i}")
        # Refresh boundOff for each step
        chBounds = [f"channel{j+1}" for j in range(len(holes))]
        bounds = adjBounds + chBounds
        stepBoundOff = boundOff + chBounds

        # Solve -D²u=b for the i-th step
        carem_ref.setRobinBC({}, {}, hc_holes_i, Tref_holes_i, Ntheta, Nz)
        carem_ref.writeStateFeenox()
        result = carem_ref.solveState()

        # Refine mesh and resolve state equation
        # Refine -> (x,y,z) on max T -> embeded node
        ## Get maximum temperature
        _, maxT = carem_ref.getTemp()
        point = np.array([maxT[0][0], maxT[0][1], 0])
        ## Create refined mesh
        subprocess.run(["rm", "-r", root_path + f"{i}"])
        carem_ref = rfl.Reflector(
            root=root_path,
            folderName=f"{i}",
            fileName="design",
            height=H,
            nz=refineParam["nz"],
            holes=holes,
            lc=refineParam["lc"],
            lch=refineParam["lch"],
            Ntheta=refineParam["Ntheta"],
            Nr=refineParam["Nr"],
            embed=[(point, lc_embed)],
            form="nonuniform",
        )
        ## Solve -D²u=b for the i-th step
        carem_ref.setRobinBC({}, {}, hc_holes_i, Tref_holes_i, Ntheta, Nz)
        carem_ref.writeStateFeenox()
        result = carem_ref.solveState()
        print(f"INFO:  Direct problem has been solved!")

        # Build adjoint BC for the i-th step
        carem_ref.writeTemperature(groups=bounds, domain=True)

        # Build hc(r) and Tref(r) for the i-th step
        ## Filter nodes from base mesh (z=0)
        gFilterNodeTags = carem_ref.geomFilter(tol, stepBoundOff, domainOff, base=True)
        # fFilterNodeTags = carem_ref.fieldFilter("T", fraction, gFilterNodeTags, True)
        fFilterNodeTags = random.choices(
            gFilterNodeTags, k=int(fraction * len(gFilterNodeTags))
        )
        if len(fFilterNodeTags) > filterThold:
            filterNodeTags = fFilterNodeTags[:filterThold]
        elif len(gFilterNodeTags) < filterThold:
            filterNodeTags = gFilterNodeTags
        elif len(fFilterNodeTags) < filterThold:
            newFrac = filterThold / len(gFilterNodeTags)
            filterNodeTags = carem_ref.fieldFilter("T", newFrac, gFilterNodeTags, True)
        print(f"INFO:  BC map should be computed on {len(filterNodeTags)} nodes")
        ## Get its centers and qn(z)
        filterCenters = [carem_ref.baseMesh.nodes[tag].r for tag in filterNodeTags]
        qn_filtCters, z = carem_ref.compHeatProfiles(filterCenters, d, Nz, Ntheta)
        qn_filtCters = [abs(simps(qn_i, z)) / H * np.ones(Nz) for qn_i in qn_filtCters]
        ## Mesh virtual holes preview
        holes_preview = [(rc, d) for rc in filterCenters]
        rfl.Reflector(
            root=root_path,
            folderName=f"{i}_preview",
            fileName="bcVirtual",
            height=H,
            nz=virtParam["nz"],
            holes=holes_preview,
            lc=virtParam["lc"],
            lch=virtParam["lch"],
            Ntheta=virtParam["Ntheta"],
            Nr=virtParam["Nr"],
        )
        ## Get qn(z) from i-th step holes
        if i == 0:
            holeCenters, qn_holeCters = [], []
        else:
            holeCenters = [hole[0] for hole in holes]
            qn_holeCters, z = carem_ref.compHeatProfiles(holeCenters, d, Nz, Ntheta)
        ## Create paralallel channel's case
        ## i holes + 1 virtual hole
        N = 6 * (i + 1) + 1  # number of channels
        K = [K_core] + [K_ch] * (N - 1)
        D = [-1] + [d] * (N - 1)
        A = [A_core] + [0.25 * pi * d**2] * (N - 1)
        beta = [beta_core] + [beta_ch] * (N - 1)
        case = pllCh.parallelChannels(N, D, A, H)
        case.setConstraints(m, P, Tm, k_water)
        case.setBeta(beta)
        case.setLocalFrictionFactor(K)
        case.setInletTemperature(Ti)
        case.compConstants()
        ## Compute virtual BC
        hc_virtual, Tref_virtual = compVirtualHolesBC(
            root=root_path,
            case=case,
            meshParams=coarseParam,
            diameter=d,
            centers=filterCenters,
            qn_centers=qn_filtCters,
            qn_core=qn_core,
            qn_holes=qn_holeCters,
            holes=holes,
            dp_0=dp_guess,
            mN_0=mN_guess,
            relax=w,
            tol=1e-2,
        )
        ## Write hc and Tref ascii files
        carem_ref.writeBCMap(filterCenters, hc_virtual, Tref_virtual)
        print(f"INFO:  BC map has been computed!")

        # Solve adjoint D²lambda=2b for step i
        # Optimize and find best next hole (refine mesh)
        carem_ref.writeAdjointFeenox()
        carem_ref.solveAdjoint()
        _, holeCenter = carem_ref.optimize(tol, "max", stepBoundOff, domainOff)
        print(f"INFO:  Adjoint problem has been solved!")

        # Refresh and save some variables
        initVol += result[0] if i == 0 else 0  # set initial reflector volume
        currVol = result[0] / initVol  # set current volume
        volume.append(currVol)
        costFunc.append(result[1])
        powerSrc.append(result[2])
        powerCh.append(sum(result[6:]) if len(holes) > 0 else 0)
        holes_coord.append(holeCenter)
        holes.append((holeCenter, d))
        i += 1

        # Solve hc & Tref for step i+1 (with new hole)
        ## Compute q channel profiles (guess)
        centers = [hole[0] for hole in holes]
        qn_holes, z = carem_ref.compHeatProfiles(centers, d, Nz=Nz, Ntheta=Ntheta)
        qn_holes = [abs(simps(qn_i, z)) / H * np.ones(Nz) for qn_i in qn_holes]
        ## Create i+1-th domain
        carem_ref = rfl.Reflector(
            root=root_path,
            folderName=f"{i}",
            fileName="design",
            height=H,
            nz=refineParam["nz"],
            holes=holes,
            lc=refineParam["lc"],
            lch=refineParam["lch"],
            Ntheta=refineParam["Ntheta"],
            Nr=refineParam["Nr"],
            form="nonuniform",
        )
        ## Compute hc, Tref for i+1-th step (with new hole)
        hc_holes_i, Tref_holes_i = compHolesBC(
            case,
            carem_ref,
            qn_core,
            qn_holes,
            dp_0=dp_guess,
            mN_0=mN_guess,
            relax=w,
        )

        # Refresh and save some variables
        dp_guess = case.dp
        mN_guess = case.mn[0]
        flag = True if case.mn[0] > mN_thold else False
        hc_holes.append(hc_holes_i)
        hc_core.append(case.compConvectionCoeff()[0])
        Tref_holes.append(Tref_holes_i)
        Tref_core_i = np.array([case.compTemp(zi) for zi in z]) + 273.15
        Tref_core_i = Tref_core_i[:, 0]
        Tref_core.append(Tref_core_i.T)
        mN.append(case.mn[0])
        mR.append(sum(case.mn[1:]))
        deltaP.append(case.dp)
        maxTemp.append(maxT)

        # Write down partial results on ascii file
        lines = {}
        ## Add feature variables
        features = [d, tol, tholdVol, mN_thold, len(holes_coord)]
        features = [str(feature) for feature in features]
        lines["some"] = [f"{sep}".join(features) + "\n"]
        ## Add delta pressure
        lines["dp"] = [f"{sep}".join([str(dp) for dp in deltaP]) + "\n"]
        ## Add core's hc
        lines["hcC"] = [f"{sep}".join([str(hc) for hc in hc_core]) + "\n"]
        ## Add holes hc
        lines["hcH"] = [
            f"{sep}".join([str(hc) for hc in hc_holes_j]) + "\n"
            for hc_holes_j in hc_holes
        ]
        ## Add core's Tref
        lines["TrefC"] = [
            f"{sep}".join([str(Tref) for Tref in Tref_core_j]) + "\n"
            for Tref_core_j in Tref_core
        ]
        ## Add holes Tref
        lines["TrefH"] = []
        for nH, Tref_holes_n in enumerate(Tref_holes):
            lines["TrefH"].append(f"{nH+1}\n")
            for Tref_holes_ni in Tref_holes_n:
                lines["TrefH"].append(
                    f"{sep}".join([str(Tref) + sep for Tref in Tref_holes_ni]) + "\n"
                )
        ## Add core mass flow
        lines["mC"] = [f"{sep}".join([str(mN_i) for mN_i in mN]) + "\n"]
        ## Add by-pass mass flow
        lines["mH"] = [f"{sep}".join([str(mR_i) for mR_i in mR]) + "\n"]
        ## Add source power
        lines["qSrc"] = [f"{sep}".join([str(ps_i) for ps_i in powerSrc]) + "\n"]
        ## Add hole power
        lines["qH"] = [f"{sep}".join([str(pch_i) for pch_i in powerCh]) + "\n"]
        ## Add volume percentage
        lines["vol"] = [f"{sep}".join([str(vol * 100) for vol in volume]) + "\n"]
        ## Add cost functional
        lines["fc"] = [f"{sep}".join([str(fc) for fc in costFunc]) + "\n"]
        ## Add center's hole coordinates
        lines["holes"] = [
            f"{sep}".join([str(xi) for xi in coord]) + "\n" for coord in holes_coord
        ]
        ## Add maximum temperature and its location
        lines["maxT"] = [
            f"{sep}".join([str(xi) for xi in rc]) + f"{sep}{temp}\n"
            for (rc, temp) in maxTemp
        ]
        ## Create file and save it
        savePath = root_path + f"{int(i-1)}/result.dat"
        open(savePath, "w").close()  # Erase content...
        f = open(savePath, "wt")
        for key in headers.keys():
            start, end = headers[key][0], headers[key][1]
            f.write(start + "\n")
            f.writelines(lines[key])
            f.write(end + "\n")
        f.close()
    # ------END: Optimization loop------ #

    # Write down final results on ascii file (post processing)
    lines = {}
    ## Add feature variables
    features = [d, tol, tholdVol, mN_thold, len(holes_coord)]
    features = [str(feature) for feature in features]
    lines["some"] = [f"{sep}".join(features) + "\n"]
    ## Add delta pressure
    lines["dp"] = [f"{sep}".join([str(dp) for dp in deltaP]) + "\n"]
    ## Add core's hc
    lines["hcC"] = [f"{sep}".join([str(hc) for hc in hc_core]) + "\n"]
    ## Add holes hc
    lines["hcH"] = [
        f"{sep}".join([str(hc) for hc in hc_holes_j]) + "\n" for hc_holes_j in hc_holes
    ]
    ## Add core's Tref
    lines["TrefC"] = [
        f"{sep}".join([str(Tref) for Tref in Tref_core_j]) + "\n"
        for Tref_core_j in Tref_core
    ]
    ## Add holes Tref
    lines["TrefH"] = []
    for nH, Tref_holes_n in enumerate(Tref_holes):
        lines["TrefH"].append(f"{nH+1}\n")
        for Tref_holes_ni in Tref_holes_n:
            lines["TrefH"].append(
                f"{sep}".join([str(Tref) + sep for Tref in Tref_holes_ni]) + "\n"
            )
    ## Add core mass flow
    lines["mC"] = [f"{sep}".join([str(mN_i) for mN_i in mN]) + "\n"]
    ## Add by-pass mass flow
    lines["mH"] = [f"{sep}".join([str(mR_i) for mR_i in mR]) + "\n"]
    ## Add source power
    lines["qSrc"] = [f"{sep}".join([str(ps_i) for ps_i in powerSrc]) + "\n"]
    ## Add hole power
    lines["qH"] = [f"{sep}".join([str(pch_i) for pch_i in powerCh]) + "\n"]
    ## Add volume percentage
    lines["vol"] = [f"{sep}".join([str(vol * 100) for vol in volume]) + "\n"]
    ## Add cost functional
    lines["fc"] = [f"{sep}".join([str(fc) for fc in costFunc]) + "\n"]
    ## Add center's hole coordinates
    lines["holes"] = [
        f"{sep}".join([str(xi) for xi in coord]) + "\n" for coord in holes_coord
    ]
    ## Add maximum temperature and its location
    lines["maxT"] = [
        f"{sep}".join([str(xi) for xi in rc]) + f"{sep}{temp}\n"
        for (rc, temp) in maxTemp
    ]
    ## Create file and save it
    savePath = root_path + "result.dat"
    open(savePath, "w").close()  # Erase content...
    f = open(savePath, "wt")
    for key in headers.keys():
        start, end = headers[key][0], headers[key][1]
        f.write(start + "\n")
        f.writelines(lines[key])
        f.write(end + "\n")
    f.close()


# Run main...
if __name__ == "__main__":
    main()
