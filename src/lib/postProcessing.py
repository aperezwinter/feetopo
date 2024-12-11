# command to add feenox to the $PATH, run on terminal...
# export PATH=$PATH:/home/alan/Desktop/feenox/bin

# importing libraries, packages, etc...
import re
import numpy as np


def process(filePath: str) -> dict:
    # Open file
    with open(filePath, "r") as f:
        lines = f.readlines()
        lines = [line.replace(" ", "") for line in lines]
        lines = [re.split(r"[|\n||]", line) for line in lines]
        lines = [[s for s in line if s != ""] for line in lines]
    f.close()

    # Set some variables
    keys = [
        "dp",
        "hcC",
        "hcH",
        "TrefC",
        "TrefH",
        "mC",
        "mH",
        "vol",
        "cf",
        "rc",
        "maxT",
        "Qsrc",
        "Qhole",
    ]
    headers = [
        ["#DELTAP"],
        ["#HCCORE"],
        ["#HCHOLE"],
        ["#TREFCORE"],
        ["#TREFHOLE"],
        ["#MASSCORE"],
        ["#MASSHOLE"],
        ["#VOLUME"],
        ["#COST"],
        ["#HOLES"],
        ["#MAXT"],
        ["#POWERSOURCE"],
        ["#POWERHOLE"],
    ]
    headersIdx = {key: lines.index(header) for key, header in zip(keys, headers)}
    data = dict.fromkeys(keys)

    # Extract delta pressure
    dp = lines[headersIdx["dp"] + 1]
    dp = np.array([float(dp_i) if dp_i != "nan" else np.nan for dp_i in dp])
    data["dp"] = dp
    data["numH"] = dp.size

    # Extract source's power
    Qsrc = lines[headersIdx["Qsrc"] + 1]
    Qsrc = np.array([float(Qsrc_i) if Qsrc_i != "nan" else np.nan for Qsrc_i in Qsrc])
    data["Qsrc"] = Qsrc

    # Extract hole's power
    Qhole = lines[headersIdx["Qhole"] + 1]
    Qhole = np.array(
        [float(Qhole_i) if Qhole_i != "nan" else np.nan for Qhole_i in Qhole]
    )
    data["Qhole"] = Qhole

    # Extract core hc coefficient
    hcC = lines[headersIdx["hcC"] + 1]
    hcC = np.array([float(hcC_i) if hcC_i != "nan" else np.nan for hcC_i in hcC])
    data["hcC"] = hcC

    # Extract core mass flow
    mC = lines[headersIdx["mC"] + 1]
    mC = np.array([float(mC_i) if mC_i != "nan" else np.nan for mC_i in mC])
    data["mC"] = mC

    # Extract holes mass flow
    mH = lines[headersIdx["mH"] + 1]
    mH = np.array([float(mH_i) if mH_i != "nan" else np.nan for mH_i in mH])
    data["mH"] = mH

    # Extract volume's percentage
    vol = lines[headersIdx["vol"] + 1]
    vol = np.array([float(vol_i) if vol_i != "nan" else np.nan for vol_i in vol])
    data["vol"] = vol

    # Extract cost function
    cf = lines[headersIdx["cf"] + 1]
    cf = np.array([float(cf_i) if cf_i != "nan" else np.nan for cf_i in cf])
    data["cf"] = cf

    # Extract hole's coordinates
    rc = [
        lines[i] for i in range(headersIdx["rc"] + 1, headersIdx["rc"] + data["numH"])
    ]
    rc = [
        np.array([float(rc_i) if rc_i != "nan" else np.nan for rc_i in rc_h])
        for rc_h in rc
    ]
    data["rc"] = rc

    # Extract maximum temperature
    maxT = [
        lines[i]
        for i in range(headersIdx["maxT"] + 1, headersIdx["maxT"] + data["numH"] + 1)
    ]
    maxT = [
        np.array([float(rc_i) if rc_i != "nan" else np.nan for rc_i in maxT_i])
        for maxT_i in maxT
    ]
    maxT = [[maxT_i[:3], maxT_i[3] - 273.15] for maxT_i in maxT]
    data["maxT"] = maxT

    # Extract core reference temperature
    TrefC = [
        lines[i]
        for i in range(headersIdx["TrefC"] + 1, headersIdx["TrefC"] + data["numH"] + 1)
    ]
    TrefC = [
        np.array(
            [float(TrefC_ij) if TrefC_ij != "nan" else np.nan for TrefC_ij in TrefC_i]
        )
        - 273.15
        for TrefC_i in TrefC
    ]
    data["TrefC"] = TrefC

    # Extract hole's hc coefficient
    hcH = [
        lines[i]
        for i in range(headersIdx["hcH"] + 1, headersIdx["hcH"] + data["numH"] + 1)
    ]
    hcH = [
        np.array([float(hcH_ij) if hcH_ij != "nan" else np.nan for hcH_ij in hcH_i])
        - 273.15
        for hcH_i in hcH
    ]
    data["hcH"] = hcH

    # Extract hole's reference temperature
    TrefH = {}
    Tref_hole_idx = [1 + sum(range(i + 1)) + i for i in range(data["numH"])]
    for idx, n in zip(Tref_hole_idx, range(1, data["numH"] + 1)):
        start = headersIdx["TrefH"] + idx + 1
        end = headersIdx["TrefH"] + idx + n + 1
        TrefH_n = lines[start:end]
        TrefH_n = [
            np.array(
                [
                    float(TrefH_ij) if TrefH_ij != "nan" else np.nan
                    for TrefH_ij in TrefH_i
                ]
            )
            - 273.15
            for TrefH_i in TrefH_n
        ]
        TrefH[n] = TrefH_n
    data["TrefH"] = TrefH

    return data
