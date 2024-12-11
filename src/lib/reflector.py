# command to add feenox to the $PATH, run on terminal...
# export PATH=$PATH:/home/alan/Desktop/feenox/bin

# importing libraries, packages, etc...
import sys, subprocess, random, gmsh
import numpy as np
from typing import Tuple, Any

# import gmsh_api.gmsh as gmsh
from itertools import compress
from operator import itemgetter
from scipy.integrate import simps
from scipy.interpolate import griddata
from math import sqrt, sin, cos, tan, pi, isclose

# CNEA source path...
# src_path = "/home/alan.perez/Escritorio/thesis_solver/"
# Home source path...
src_path = "/home/alan/Desktop/MI_IB/thesis_code-master/"
if src_path not in sys.path:
    sys.path.append(src_path)


import src.lib.geom.point as Point
import src.lib.mesh as mesh


# ----------------------- #
# MESH BUILDING FUNCTIONS #
# ----------------------- #

# 2D Mesh (base z=0) for optimization.
# Geometry extracted from CAREM's draws.
# It includes "boundGap" group.
# Design to be obtained for building purpose.
def baseMesh(
    root: str,  # root address
    fileName: str,  # file name
    modelName: str = "model_0",  # gmsh model name
    holes: list = [],  # hole[i] = (rc, d)
    lc: float = 10e-3,  # characteristic length
    lch: float = 1e-3,  # channel's characteristic length
    Ntheta: int = 100,  # number of elements on theta axis
    Nr: int = 5,  # number of elements on r axis
    embed: list = [],  # embed[i] = (rc, lc), embedded points
) -> None:
    # Initialize Gmsh.
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(modelName)

    # Some variables
    # Geometric parameters
    diam_screw = 0.042
    sep_ref = 0.08
    di_ref = 0.6484
    Ri_ref = 0.723
    Re_ref = 0.775
    Re_bar = 0.839
    g = 4e-3
    tol = 1e-4
    alpha = 26 * pi / 180
    beta = 30 * pi / 180
    gamma = 60 * pi / 180
    # Origin (x,y,z) = (0,0,0)
    p0 = gmsh.model.occ.addPoint(0, 0, 0)
    # Geom dictionaries, hook: cáncamo, screw: tornillo
    regions = ["zone1", "zone2", "gap", "barrel", "hook", "screw", "holes"]
    x = {region: [] for region in regions}
    y = {region: [] for region in regions}
    arcMass = {region: [] for region in regions}
    arcCoM = {region: [] for region in regions}
    surfMass = {region: [] for region in regions}
    surfCoM = {region: [] for region in regions}
    # Mesh dictionaries, hook: cáncamo, screw: tornillo
    points = {region: [] for region in regions}
    curves = {region: [] for region in regions}
    loops = {region: [] for region in regions}
    surfs = {region: [] for region in regions}

    # ----------------------------- #
    # ---------- ZONE 1 ----------- #
    # 9 points, 8 curves, 1 surface #
    # ----------------------------- #
    x["zone1"] = [
        Ri_ref * cos(alpha),  # x1
        Ri_ref * cos(alpha),  # x2
        di_ref - sep_ref / tan(gamma),  # x3
        di_ref,  # x4
        di_ref - sep_ref / tan(gamma),  # x5
        di_ref,  # x6
        di_ref - sep_ref / tan(gamma),  # x7
        di_ref,  # x8
        di_ref - sep_ref / tan(gamma),  # x9
    ]
    y["zone1"] = [
        -Ri_ref * sin(alpha),  # y1
        Ri_ref * sin(alpha),  # y2
        3 * sep_ref,  # y3
        2 * sep_ref,  # y4
        sep_ref,  # y5
        0,  # y6
        -sep_ref,  # y7
        -2 * sep_ref,  # y8
        -3 * sep_ref,  # y9
    ]
    arcMass["zone1"] = [
        2 * Ri_ref * alpha,  # l1
        sqrt(
            (x["zone1"][2] - x["zone1"][1]) ** 2 + (y["zone1"][2] - y["zone1"][1]) ** 2
        ),  # l2
        sep_ref / sin(gamma),  # l3
        sep_ref / sin(gamma),  # l4
        sep_ref / sin(gamma),  # l5
        sep_ref / sin(gamma),  # l6
        sep_ref / sin(gamma),  # l7
        sep_ref / sin(gamma),  # l8
        sqrt(
            (x["zone1"][0] - x["zone1"][8]) ** 2 + (y["zone1"][0] - y["zone1"][8]) ** 2
        ),  # l9
    ]
    arcCoM["zone1"] = [
        np.array([2 * Ri_ref**2 * sin(alpha), 0, 0]) / arcMass["zone1"][0],  # l1
        np.array([x["zone1"][1] + x["zone1"][2], y["zone1"][1] + y["zone1"][2], 0])
        / 2,  # l2
        np.array([x["zone1"][2] + x["zone1"][3], y["zone1"][2] + y["zone1"][3], 0])
        / 2,  # l3
        np.array([x["zone1"][3] + x["zone1"][4], y["zone1"][3] + y["zone1"][4], 0])
        / 2,  # l4
        np.array([x["zone1"][4] + x["zone1"][5], y["zone1"][4] + y["zone1"][5], 0])
        / 2,  # l5
        np.array([x["zone1"][5] + x["zone1"][6], y["zone1"][5] + y["zone1"][6], 0])
        / 2,  # l6
        np.array([x["zone1"][6] + x["zone1"][7], y["zone1"][6] + y["zone1"][7], 0])
        / 2,  # l7
        np.array([x["zone1"][7] + x["zone1"][8], y["zone1"][7] + y["zone1"][8], 0])
        / 2,  # l8
        np.array([x["zone1"][8] + x["zone1"][0], y["zone1"][8] + y["zone1"][0], 0])
        / 2,  # l9
    ]
    # MESH STARTS HERE #
    points["zone1"] = [
        gmsh.model.occ.addPoint(xi, yi, 0, lc) for xi, yi in zip(x["zone1"], y["zone1"])
    ]
    curves["zone1"] = [
        gmsh.model.occ.addCircleArc(points["zone1"][0], p0, points["zone1"][1])
    ] + [
        gmsh.model.occ.addLine(pi, pj)
        for pi, pj in zip(
            points["zone1"][1:], points["zone1"][2:] + [points["zone1"].pop(0)]
        )
    ]
    loops["zone1"] = [gmsh.model.occ.addCurveLoop(curves["zone1"])]
    surfs["zone1"] = [gmsh.model.occ.addPlaneSurface([cl]) for cl in loops["zone1"]]
    gmsh.model.occ.synchronize()  # synchronize model

    # ----------------------------- #
    # ---------- ZONE 2 ----------- #
    # 4 points, 4 curves, 1 surface #
    # After fragment operation:     #
    # 6 points, 6 curves, 1 surface #
    # ----------------------------- #
    x["zone2"] = [
        Ri_ref * cos(beta),  # x1
        Re_ref * cos(beta),  # x2
        Re_ref * cos(beta),  # x3
        Ri_ref * cos(beta),  # x4
    ]
    y["zone2"] = [
        -Ri_ref * sin(beta),  # y1
        -Re_ref * sin(beta),  # y2
        Re_ref * sin(beta),  # y3
        Ri_ref * sin(beta),  # y4
    ]
    arcMass["zone2"] = [
        Re_ref - Ri_ref,  # l1
        Re_ref * gamma,  # l2
        Re_ref - Ri_ref,  # l3
        Ri_ref * (beta - alpha),  # l4
        2 * Ri_ref * alpha,  # l5
        Ri_ref * (beta - alpha),  # l6
    ]
    arcCoM["zone2"] = [
        (Ri_ref + Re_ref) * np.array([cos(-beta), sin(-beta), 0]) / 2,  # l1
        np.array([2 * Re_ref**2 * sin(beta), 0, 0]) / arcMass["zone2"][1],  # l2
        (Ri_ref + Re_ref) * np.array([cos(beta), sin(beta), 0]) / 2,  # l3
        Ri_ref**2
        * np.array([sin(beta) - sin(alpha), cos(alpha) - cos(beta), 0])
        / arcMass["zone2"][3],  # l4
        np.array([2 * Ri_ref**2 * sin(alpha), 0, 0]) / arcMass["zone2"][4],  # l5
        Ri_ref**2
        * np.array([sin(-alpha) - sin(-beta), cos(-beta) - cos(-alpha), 0])
        / arcMass["zone2"][5],  # l6
    ]
    surfMass["zone2"] = [gamma * (Re_ref**2 - Ri_ref**2) / 2]
    surfCoM["zone2"] = [
        np.array(
            [
                2
                * sin(beta)
                * (Re_ref**3 - Ri_ref**3)
                / (3 * surfMass["zone2"][0]),
                0,
                0,
            ]
        )
    ]
    # MESH STARTS HERE #
    points["zone2"] = [
        gmsh.model.occ.addPoint(xi, yi, 0, lc) for xi, yi in zip(x["zone2"], y["zone2"])
    ]
    curves["zone2"] = [
        gmsh.model.occ.addLine(points["zone2"][0], points["zone2"][1]),
        gmsh.model.occ.addCircleArc(points["zone2"][1], p0, points["zone2"][2]),
        gmsh.model.occ.addLine(points["zone2"][2], points["zone2"][3]),
        gmsh.model.occ.addCircleArc(points["zone2"][0], p0, points["zone2"][3]),
    ]
    loops["zone2"] = [
        gmsh.model.occ.addCurveLoop(curves["zone2"][:-1] + [-curves["zone2"][3]])
    ]
    surfs["zone2"] = [gmsh.model.occ.addPlaneSurface([cl]) for cl in loops["zone2"]]
    gmsh.model.occ.synchronize()  # synchronize model

    # ----------------------------- #
    # ------------ GAP ------------ #
    # 4 points, 4 curves, 1 surface #
    # ----------------------------- #
    x["gap"] = [
        Re_ref * cos(beta),  # x1
        (Re_ref + g) * cos(beta),  # x2
        (Re_ref + g) * cos(beta),  # x3
        Re_ref * cos(beta),  # x4
    ]
    y["gap"] = [
        -Re_ref * sin(beta),  # y1
        -(Re_ref + g) * sin(beta),  # y2
        (Re_ref + g) * sin(beta),  # y3
        Re_ref * sin(beta),  # y4
    ]
    arcMass["gap"] = [
        g,  # l1
        (Re_ref + g) * gamma,  # l2
        g,  # l3
        Re_ref * gamma,  # l4
    ]
    arcCoM["gap"] = [
        np.array([x["gap"][0] + x["gap"][1], y["gap"][0] + y["gap"][1], 0]) / 2,  # l1
        np.array([2 * (Re_ref + g) ** 2 * sin(beta), 0, 0]) / arcMass["gap"][1],  # l2
        np.array([x["gap"][2] + x["gap"][3], y["gap"][2] + y["gap"][3], 0]) / 2,  # l3
        np.array([2 * Re_ref**2 * sin(beta), 0, 0]) / arcMass["gap"][3],  # l4
    ]
    surfMass["gap"] = [gamma * ((Re_ref + g) ** 2 - Re_ref**2) / 2]
    surfCoM["gap"] = [
        np.array(
            [
                2
                * sin(beta)
                * ((Re_ref + g) ** 3 - Re_ref**3)
                / (3 * surfMass["gap"][0]),
                0,
                0,
            ]
        )
    ]
    # MESH STARTS HERE #
    points["gap"] = [
        gmsh.model.occ.addPoint(xi, yi, 0, lc) for xi, yi in zip(x["gap"], y["gap"])
    ]
    curves["gap"] = [
        gmsh.model.occ.addLine(points["gap"][0], points["gap"][1]),
        gmsh.model.occ.addCircleArc(points["gap"][1], p0, points["gap"][2]),
        gmsh.model.occ.addLine(points["gap"][2], points["gap"][3]),
        gmsh.model.occ.addCircleArc(points["gap"][0], p0, points["gap"][3]),
    ]
    loops["gap"] = [
        gmsh.model.occ.addCurveLoop(curves["gap"][:-1] + [-curves["gap"][3]])
    ]
    surfs["gap"] = [gmsh.model.occ.addPlaneSurface([cl]) for cl in loops["gap"]]
    gmsh.model.occ.synchronize()  # synchronize model

    # ----------------------------- #
    # ---------- BARREL ----------- #
    # 4 points, 4 curves, 1 surface #
    # ----------------------------- #
    x["barrel"] = [
        (Re_ref + g) * cos(beta),  # x1
        Re_bar * cos(beta),  # x2
        Re_bar * cos(beta),  # x3
        (Re_ref + g) * cos(beta),  # x4
    ]
    y["barrel"] = [
        -(Re_ref + g) * sin(beta),  # y1
        -Re_bar * sin(beta),  # y2
        Re_bar * sin(beta),  # y3
        (Re_ref + g) * sin(beta),  # y4
    ]
    arcMass["barrel"] = [
        Re_bar - Re_ref - g,  # l1
        Re_bar * gamma,  # l2
        Re_bar - Re_ref - g,  # l3
        (Re_ref + g) * gamma,  # l4
    ]
    arcCoM["barrel"] = [
        np.array([x["barrel"][0] + x["barrel"][1], y["barrel"][0] + y["barrel"][1], 0])
        / 2,  # l1
        np.array([2 * Re_bar**2 * sin(beta), 0, 0]) / arcMass["barrel"][1],  # l2
        np.array([x["barrel"][2] + x["barrel"][3], y["barrel"][2] + y["barrel"][3], 0])
        / 2,  # l3
        np.array([2 * (Re_ref + g) ** 2 * sin(beta), 0, 0])
        / arcMass["barrel"][3],  # l4
    ]
    surfMass["barrel"] = [gamma * (Re_bar**2 - (Re_ref + g) ** 2) / 2]
    surfCoM["barrel"] = [
        np.array(
            [
                2
                * sin(beta)
                * (Re_bar**3 - (Re_ref + g) ** 3)
                / (3 * surfMass["barrel"][0]),
                0,
                0,
            ]
        )
    ]
    # MESH STARTS HERE #
    points["barrel"] = [
        gmsh.model.occ.addPoint(xi, yi, 0, lc)
        for xi, yi in zip(x["barrel"], y["barrel"])
    ]
    curves["barrel"] = [
        gmsh.model.occ.addLine(points["barrel"][0], points["barrel"][1]),
        gmsh.model.occ.addCircleArc(points["barrel"][1], p0, points["barrel"][2]),
        gmsh.model.occ.addLine(points["barrel"][2], points["barrel"][3]),
        gmsh.model.occ.addCircleArc(points["barrel"][0], p0, points["barrel"][3]),
    ]
    loops["barrel"] = [
        gmsh.model.occ.addCurveLoop(curves["barrel"][:-1] + [-curves["barrel"][3]])
    ]
    surfs["barrel"] = [gmsh.model.occ.addPlaneSurface([cl]) for cl in loops["barrel"]]
    gmsh.model.occ.synchronize()  # synchronize model

    # ----------------------------- #
    # ----------- HOOK ------------ #
    # 4 points, 4 curves, 1 surface #
    # ----------------------------- #
    x["hook"] = [
        Ri_ref - 0.057,  # x1
        Ri_ref - 0.025,  # x2
        Ri_ref - 0.025,  # x3
        Ri_ref - 0.057,  # x4
    ]
    y["hook"] = [
        -0.056,  # y1
        -0.056,  # y2
        0.056,  # y3
        0.056,  # y4
    ]
    arcMass["hook"] = [
        0.032,  # l1
        0.112,  # l2
        0.032,  # l3
        0.112,  # l4
    ]
    arcCoM["hook"] = [
        np.array([x["hook"][0] + x["hook"][1], y["hook"][0] + y["hook"][1], 0])
        / 2,  # l1
        np.array([x["hook"][1] + x["hook"][2], y["hook"][1] + y["hook"][2], 0])
        / 2,  # l2
        np.array([x["hook"][2] + x["hook"][3], y["hook"][2] + y["hook"][3], 0])
        / 2,  # l3
        np.array([x["hook"][3] + x["hook"][0], y["hook"][3] + y["hook"][0], 0])
        / 2,  # l4
    ]
    surfMass["hook"] = [0.112 * 0.032]
    surfCoM["hook"] = [np.array([Ri_ref - 0.041, 0, 0])]
    # MESH STARTS HERE #
    points["hook"] = [
        gmsh.model.occ.addPoint(xi, yi, 0, lc) for xi, yi in zip(x["hook"], y["hook"])
    ]
    curves["hook"] = [
        gmsh.model.occ.addLine(pi, pj)
        for pi, pj in zip(
            points["hook"][0:], points["hook"][1:] + [points["hook"].pop(0)]
        )
    ]
    loops["hook"] = [gmsh.model.occ.addCurveLoop(curves["hook"])]
    surfs["hook"] = [gmsh.model.occ.addPlaneSurface([cl]) for cl in loops["hook"]]
    gmsh.model.occ.synchronize()  # synchronize model

    # ----------------------------- #
    # ----------- Screw ----------- #
    # 0 point, 0 curve, 6 surfaces  #
    # ----------------------------- #
    x["screw"] = [
        Ri_ref - 0.067,  # xc_1
        Ri_ref - 0.067,  # xc_2
        Ri_ref - 0.067,  # xc_3
        Ri_ref - 0.067,  # xc_4
        Ri_ref - 0.037,  # xc_5
        Ri_ref - 0.037,  # xc_6
    ]
    y["screw"] = [
        0.235,  # yc_1
        0.085,  # yc_2
        -0.085,  # yc_3
        -0.235,  # yc_4
        0.125,  # yc_5
        -0.125,  # yc_6
    ]
    arcMass["screw"] = [
        pi * diam_screw,  # l1
        pi * diam_screw,  # l2
        pi * diam_screw,  # l3
        pi * diam_screw,  # l4
        pi * diam_screw,  # l5
        pi * diam_screw,  # l6
    ]
    arcCoM["screw"] = [
        np.array([Ri_ref - 0.067, 0.235, 0]),  # l1
        np.array([Ri_ref - 0.067, 0.085, 0]),  # l2
        np.array([Ri_ref - 0.067, -0.085, 0]),  # l3
        np.array([Ri_ref - 0.067, -0.235, 0]),  # l4
        np.array([Ri_ref - 0.037, 0.125, 0]),  # l5
        np.array([Ri_ref - 0.037, -0.125, 0]),  # l6
    ]
    surfMass["screw"] = [
        0.25 * pi * diam_screw**2,  # s1
        0.25 * pi * diam_screw**2,  # s2
        0.25 * pi * diam_screw**2,  # s3
        0.25 * pi * diam_screw**2,  # s4
        0.25 * pi * diam_screw**2,  # s5
        0.25 * pi * diam_screw**2,  # s6
    ]
    surfCoM["screw"] = [
        np.array([Ri_ref - 0.067, 0.235, 0]),  # s1
        np.array([Ri_ref - 0.067, 0.235, 0]),  # s2
        np.array([Ri_ref - 0.067, 0.235, 0]),  # s3
        np.array([Ri_ref - 0.067, 0.235, 0]),  # s4
        np.array([Ri_ref - 0.067, 0.235, 0]),  # s5
        np.array([Ri_ref - 0.067, 0.235, 0]),  # s6
    ]
    # MESH STARTS HERE #
    surfs["screw"] = [
        gmsh.model.occ.addDisk(xc, yc, 0, diam_screw / 2, diam_screw / 2)
        for xc, yc in zip(x["screw"], y["screw"])
    ]
    # Synchronize model.
    gmsh.model.occ.synchronize()


    # --------------------------- #
    # BOOLEAN OPERATION: FRAGMENT #
    # --------------------------- #
    # CUT: make holes.
    if len(holes) > 0:
        sHoles = []
        for rc, d in holes:
            sHoles.append(gmsh.model.occ.addDisk(rc[0], rc[1], rc[2], d / 2, d / 2))
        # Perform boolean operation...
        gmsh.model.occ.cut(
            [(2, tag) for tag in surfs["zone1"] + surfs["zone2"]],
            [(2, tag) for tag in sHoles],
        )
    # FRAGMENT: fuse parts of the domain.
    gmsh.model.occ.fragment([(2, surfs["gap"][0])], [(2, surfs["barrel"][0])])
    gmsh.model.occ.fragment([(2, surfs["zone2"][0])], [(2, surfs["gap"][0])])
    gmsh.model.occ.fragment(
        [(2, surfs["zone1"][0])],
        [(2, s) for s in surfs["screw"]]
        + [(2, surfs["hook"][0]), (2, surfs["zone2"][0])],
    )
    gmsh.model.occ.synchronize()  # synchronize model

    # --------------- #
    # SEARCH ENTITIES #
    # --------------- #
    # HOLES ENTITIES
    points["holes"], curves["holes"] = [], []
    for rc, d in holes:
        points["holes"] += gmsh.model.occ.getEntitiesInBoundingBox(
            xmin=rc[0] - d / 2 - tol,
            ymin=rc[1] - d / 2 - tol,
            zmin=-tol,
            xmax=rc[0] + d / 2 + tol,
            ymax=rc[1] + d / 2 + tol,
            zmax=tol,
            dim=0,
        )
        curves["holes"] += gmsh.model.occ.getEntitiesInBoundingBox(
            xmin=rc[0] - d / 2 - tol,
            ymin=rc[1] - d / 2 - tol,
            zmin=-tol,
            xmax=rc[0] + d / 2 + tol,
            ymax=rc[1] + d / 2 + tol,
            zmax=tol,
            dim=1,
        )
    points["holes"] = list(set([tag for _, tag in points["holes"]]))
    curves["holes"] = list(set([tag for _, tag in curves["holes"]]))
    # HOOK ENTITIES
    ## curves, dim = 1
    curves["hook"], arcMass["hook"], arcCoM["hook"] = [], [], []
    dimTags = gmsh.model.occ.getEntitiesInBoundingBox(
        xmin=x["hook"][0] - tol,
        ymin=y["hook"][0] - tol,
        zmin=-tol,
        xmax=x["hook"][1] + tol,
        ymax=y["hook"][2] + tol,
        zmax=tol,
        dim=1,
    )
    for dim, tag in dimTags:
        rc = gmsh.model.occ.getCenterOfMass(dim, tag)
        mass = gmsh.model.occ.getMass(dim, tag)
        curves["hook"].append(tag)
        arcMass["hook"].append(mass)
        arcCoM["hook"].append(np.array(rc))
    ## surfaces, dim = 2
    entities = gmsh.model.occ.getEntitiesInBoundingBox(
        xmin=x["hook"][0] - tol,
        ymin=y["hook"][0] - tol,
        zmin=-tol,
        xmax=x["hook"][1] + tol,
        ymax=y["hook"][2] + tol,
        zmax=tol,
        dim=2,
    )
    surfs["hook"] = [tag for dim, tag in entities]
    # SCREW ENTITIES
    ## curves, dim = 1
    points["screw"], curves["screw"], surfs["screw"] = [], [], []
    for rc in arcCoM["screw"]:
        points["screw"] += gmsh.model.occ.getEntitiesInBoundingBox(
            xmin=rc[0] - diam_screw / 2 - tol,
            ymin=rc[1] - diam_screw / 2 - tol,
            zmin=-tol,
            xmax=rc[0] + diam_screw / 2 + tol,
            ymax=rc[1] + diam_screw / 2 + tol,
            zmax=tol,
            dim=0,
        )
        curves["screw"] += gmsh.model.occ.getEntitiesInBoundingBox(
            xmin=rc[0] - diam_screw / 2 - tol,
            ymin=rc[1] - diam_screw / 2 - tol,
            zmin=-tol,
            xmax=rc[0] + diam_screw / 2 + tol,
            ymax=rc[1] + diam_screw / 2 + tol,
            zmax=tol,
            dim=1,
        )
        surfs["screw"] += gmsh.model.occ.getEntitiesInBoundingBox(
            xmin=rc[0] - diam_screw / 2 - tol,
            ymin=rc[1] - diam_screw / 2 - tol,
            zmin=-tol,
            xmax=rc[0] + diam_screw / 2 + tol,
            ymax=rc[1] + diam_screw / 2 + tol,
            zmax=tol,
            dim=2,
        )
    points["screw"] = [tag for _, tag in points["screw"]]
    curves["screw"] = [tag for _, tag in curves["screw"]]
    surfs["screw"] = [tag for _, tag in surfs["screw"]]
    # REMAINING ENTITIES
    ## curves, dim = 1
    currCurves = curves["hook"] + curves["screw"] + curves["holes"]
    allCurves = [tag for _, tag in gmsh.model.getEntities(dim=1)]
    remCurves = list(set(allCurves) - set(currCurves))
    remCurves_mass_com = [
        (
            gmsh.model.occ.getMass(1, tag),
            np.array(gmsh.model.occ.getCenterOfMass(1, tag)),
        )
        for tag in remCurves
    ]
    ### ZONE 1
    points["zone1"], curves["zone1"] = [], []
    for mass, rc in zip(arcMass["zone1"], arcCoM["zone1"]):
        for i, (other_mass, other_rc) in enumerate(remCurves_mass_com):
            if isclose(mass, other_mass) and all(np.isclose(rc, other_rc)):
                curves["zone1"].append(remCurves[i])
                _, downward = gmsh.model.getAdjacencies(1, remCurves[i])
                points["zone1"] += list(downward)
                continue
    points["zone1"] = list(set(points["zone1"]))
    ### ZONE 2
    points["zone2"], curves["zone2"] = [], []
    for mass, rc in zip(arcMass["zone2"], arcCoM["zone2"]):
        for i, (other_mass, other_rc) in enumerate(remCurves_mass_com):
            if isclose(mass, other_mass) and all(np.isclose(rc, other_rc)):
                curves["zone2"].append(remCurves[i])
                _, downward = gmsh.model.getAdjacencies(1, remCurves[i])
                points["zone2"] += list(downward)
                continue
    points["zone2"] = list(set(points["zone2"]))
    ### GAP
    points["gap"], curves["gap"] = [], []
    for mass, rc in zip(arcMass["gap"], arcCoM["gap"]):
        for i, (other_mass, other_rc) in enumerate(remCurves_mass_com):
            if isclose(mass, other_mass) and all(np.isclose(rc, other_rc)):
                curves["gap"].append(remCurves[i])
                _, downward = gmsh.model.getAdjacencies(1, remCurves[i])
                points["gap"] += list(downward)
                continue
    points["gap"] = list(set(points["gap"]))
    ### BARREL
    points["barrel"], curves["barrel"] = [], []
    for mass, rc in zip(arcMass["barrel"], arcCoM["barrel"]):
        for i, (other_mass, other_rc) in enumerate(remCurves_mass_com):
            if isclose(mass, other_mass) and all(np.isclose(rc, other_rc)):
                curves["barrel"].append(remCurves[i])
                _, downward = gmsh.model.getAdjacencies(1, remCurves[i])
                points["barrel"] += list(downward)
                continue
    points["barrel"] = list(set(points["barrel"]))
    ## surfaces, dim = 2
    currSurfs = surfs["hook"] + surfs["screw"]
    allSurfs = [tag for _, tag in gmsh.model.getEntities(dim=2)]
    remSurfs = list(set(allSurfs) - set(currSurfs))
    remSurfs_mass_com = [
        (
            gmsh.model.occ.getMass(2, tag),
            np.array(gmsh.model.occ.getCenterOfMass(2, tag)),
        )
        for tag in remSurfs
    ]
    ### ZONE 1
    surfs["zone1"], _ = gmsh.model.getAdjacencies(dim=1, tag=curves["zone1"][1])
    ### ZONE 2
    surfs["zone2"], _ = gmsh.model.getAdjacencies(dim=1, tag=curves["zone2"][0])
    ### GAP
    surfs["gap"] = []
    for mass, rc in zip(surfMass["gap"], surfCoM["gap"]):
        for i, (other_mass, other_rc) in enumerate(remSurfs_mass_com):
            if isclose(mass, other_mass) and all(np.isclose(rc, other_rc)):
                surfs["gap"].append(remSurfs[i])
                continue
    ### BARREL
    surfs["barrel"] = []
    for mass, rc in zip(surfMass["barrel"], surfCoM["barrel"]):
        for i, (other_mass, other_rc) in enumerate(remSurfs_mass_com):
            if isclose(mass, other_mass) and all(np.isclose(rc, other_rc)):
                surfs["barrel"].append(remSurfs[i])
                continue

    # ----------------------------- #
    # TRANSFINITE CURVES & SURFACES #
    # ----------------------------- #
    # Set structured grid parameters on hook, barrel and gap zones.
    ## Hook zone curves.
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["hook"][0], numNodes=10)  # l1
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["hook"][1], numNodes=20)  # l2
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["hook"][2], numNodes=10)  # l3
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["hook"][3], numNodes=20)  # l4
    ## Gap zone curves.
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["gap"][0], numNodes=Nr)  # l1
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["gap"][1], numNodes=Ntheta)  # l2
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["gap"][2], numNodes=Nr)  # l3
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["gap"][3], numNodes=Ntheta)  # l4
    ## Barrel zone curves.
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["barrel"][0], numNodes=Nr)  # l1
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["barrel"][1], numNodes=Ntheta)  # l2
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["barrel"][2], numNodes=Nr)  # l3
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["barrel"][3], numNodes=Ntheta)  # l4

    ## Hook zone surface
    gmsh.model.mesh.setTransfiniteSurface(tag=surfs["hook"][0])
    ## Gap zone surface
    gmsh.model.mesh.setTransfiniteSurface(tag=surfs["gap"][0])
    ## Barrel zone surface
    gmsh.model.mesh.setTransfiniteSurface(tag=surfs["barrel"][0])

    # Synchronize model...
    gmsh.model.occ.synchronize()

    # Remove duplicate surfaces...
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    # --------------- #
    # PHYSICAL GROUPS #
    # --------------- #
    # Inner BC -> Robin type.
    inner = curves["zone1"][1:] + [curves["zone2"][3], curves["zone2"][5]]
    innerTag = gmsh.model.addPhysicalGroup(1, inner)
    gmsh.model.setPhysicalName(1, innerTag, "inner")

    # Outer BC -> Robin type.
    outer = [curves["barrel"][1]]
    outerTag = gmsh.model.addPhysicalGroup(1, outer)
    gmsh.model.setPhysicalName(1, outerTag, "outer")

    # Side BC -> Neumann type.
    side = [
        curves["zone2"][0],
        curves["zone2"][2],
        curves["gap"][0],
        curves["gap"][2],
        curves["barrel"][0],
        curves["barrel"][2],
    ]
    sideTag = gmsh.model.addPhysicalGroup(1, side)
    gmsh.model.setPhysicalName(1, sideTag, "side")

    # Channels BC -> Robin type.
    for i, channel in enumerate(curves["holes"]):
        chTag = gmsh.model.addPhysicalGroup(1, [channel])
        gmsh.model.setPhysicalName(1, chTag, "channel" + str(int(i + 1)))

    # Complete boundary: internal and external
    bndEntities = [tag for _, tag in gmsh.model.occ.getEntities(dim=1)]
    bndTag = gmsh.model.addPhysicalGroup(1, bndEntities)
    gmsh.model.setPhysicalName(1, bndTag, "bound")

    # Domain part: Zone 1
    zOneTag = gmsh.model.addPhysicalGroup(2, surfs["zone1"])
    gmsh.model.setPhysicalName(2, zOneTag, "zoneOne")

    # Domain part: Zone 2
    zTwoTag = gmsh.model.addPhysicalGroup(2, surfs["zone2"])
    gmsh.model.setPhysicalName(2, zTwoTag, "zoneTwo")

    # Domain part: Screw
    screwTag = gmsh.model.addPhysicalGroup(2, surfs["screw"])
    gmsh.model.setPhysicalName(2, screwTag, "screw")

    # Domain part: Hook
    hookTag = gmsh.model.addPhysicalGroup(2, surfs["hook"])
    gmsh.model.setPhysicalName(2, hookTag, "hook")

    # Domain part: Gap
    gapTag = gmsh.model.addPhysicalGroup(2, surfs["gap"])
    gmsh.model.setPhysicalName(2, gapTag, "gap")

    # Domain part: Barrel
    barrelTag = gmsh.model.addPhysicalGroup(2, surfs["barrel"])
    gmsh.model.setPhysicalName(2, barrelTag, "barrel")

    # Assign a mesh size to hole points:
    gmsh.model.mesh.setSize([(0, tag) for tag in points["zone1"]], lc)
    gmsh.model.mesh.setSize([(0, tag) for tag in points["screw"]], 1.5 * lc)
    gmsh.model.mesh.setSize([(0, tag) for tag in points["holes"]], lch)

    # Set the order of the elements in the mesh.
    gmsh.model.mesh.setOrder(1)

    # Generate a 2D mesh...
    gmsh.model.mesh.generate(2)

    # ... and save it to disk
    gmsh.write(fileName=root + fileName + ".msh")

    # Launch the GUI to see the results:
    # if "-nopopup" not in sys.argv:
    #    gmsh.fltk.run()

    gmsh.model.remove()

    gmsh.finalize()

# 3D Mesh for FEM solver.
# Geometry extracted from CAREM's draws.
# Design to be obtained for building purpose.
# It NOT includes "boundGap" group.
def extrudeMesh(
    root: str,  # root address
    fileName: str,  # file name
    modelName: str = "model_0",  # gmsh model name
    height: float = 1,  # height to extrude in meters
    nz: int = 10,  # element's number along z axis
    holes: list = [],  # hole[i] = (rc, d)
    lc: float = 10e-3,  # characteristic length
    lch: float = 1e-3,  # channel's characteristic length
    Ntheta: int = 100,  # number of elements on theta axis
    Nr: int = 5,  # number of elements on r axis
    embed: list = [],  # embed[i] = (rc, lc), embedded points
    form: str = "uniform",  # distributed z elements
) -> None:
    # Initialize Gmsh.
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(modelName)

    # Some variables
    # Geometric parameters
    sep_ref = 0.08
    di_ref = 0.6484
    Ri_ref = 0.723
    Re_ref = 0.775
    Re_bar = 0.839
    g = 4e-3
    tol = 1e-4
    alpha = 26 * pi / 180
    beta = 30 * pi / 180
    gamma = 60 * pi / 180
    # Origin (x,y,z) = (0,0,0)
    p0 = gmsh.model.occ.addPoint(0, 0, 0)
    # Discretization among z axis
    nz = [nz] if form == "uniform" else [3, int(0.6 * nz), 5]
    # Geom dictionaries, hook: cáncamo, screw: tornillo
    regions = [
        "zone1",
        "zone2",
        "gap",
        "barrel",
        "top",
        "bottom",
        "holes",
    ]
    x = {region: [] for region in regions}
    y = {region: [] for region in regions}
    arcMass = {region: [] for region in regions}
    arcCoM = {region: [] for region in regions}
    surfMass = {region: [] for region in regions}
    surfCoM = {region: [] for region in regions}
    volMass = {region: [] for region in regions}
    volCoM = {region: [] for region in regions}
    # Mesh dictionaries, hook: cáncamo, screw: tornillo
    points = {region: [] for region in regions}
    curves = {region: [] for region in regions}
    loops = {region: [] for region in regions}
    surfs = {region: [] for region in regions}
    vols = {region: [] for region in regions}

    # ----------------------------- #
    # ---------- ZONE 1 ----------- #
    # 9 points, 8 curves, 1 surface #
    # ----------------------------- #
    x["zone1"] = [
        Ri_ref * cos(alpha),  # x1
        Ri_ref * cos(alpha),  # x2
        di_ref - sep_ref / tan(gamma),  # x3
        di_ref,  # x4
        di_ref - sep_ref / tan(gamma),  # x5
        di_ref,  # x6
        di_ref - sep_ref / tan(gamma),  # x7
        di_ref,  # x8
        di_ref - sep_ref / tan(gamma),  # x9
    ]
    y["zone1"] = [
        -Ri_ref * sin(alpha),  # y1
        Ri_ref * sin(alpha),  # y2
        3 * sep_ref,  # y3
        2 * sep_ref,  # y4
        sep_ref,  # y5
        0,  # y6
        -sep_ref,  # y7
        -2 * sep_ref,  # y8
        -3 * sep_ref,  # y9
    ]
    arcMass["zone1"] = [
        2 * Ri_ref * alpha,  # l1
        sqrt(
            (x["zone1"][2] - x["zone1"][1]) ** 2 + (y["zone1"][2] - y["zone1"][1]) ** 2
        ),  # l2
        sep_ref / sin(gamma),  # l3
        sep_ref / sin(gamma),  # l4
        sep_ref / sin(gamma),  # l5
        sep_ref / sin(gamma),  # l6
        sep_ref / sin(gamma),  # l7
        sep_ref / sin(gamma),  # l8
        sqrt(
            (x["zone1"][0] - x["zone1"][8]) ** 2 + (y["zone1"][0] - y["zone1"][8]) ** 2
        ),  # l9
    ]
    arcCoM["zone1"] = [
        np.array([2 * Ri_ref**2 * sin(alpha), 0, 0]) / arcMass["zone1"][0],  # l1
        np.array([x["zone1"][1] + x["zone1"][2], y["zone1"][1] + y["zone1"][2], 0])
        / 2,  # l2
        np.array([x["zone1"][2] + x["zone1"][3], y["zone1"][2] + y["zone1"][3], 0])
        / 2,  # l3
        np.array([x["zone1"][3] + x["zone1"][4], y["zone1"][3] + y["zone1"][4], 0])
        / 2,  # l4
        np.array([x["zone1"][4] + x["zone1"][5], y["zone1"][4] + y["zone1"][5], 0])
        / 2,  # l5
        np.array([x["zone1"][5] + x["zone1"][6], y["zone1"][5] + y["zone1"][6], 0])
        / 2,  # l6
        np.array([x["zone1"][6] + x["zone1"][7], y["zone1"][6] + y["zone1"][7], 0])
        / 2,  # l7
        np.array([x["zone1"][7] + x["zone1"][8], y["zone1"][7] + y["zone1"][8], 0])
        / 2,  # l8
        np.array([x["zone1"][8] + x["zone1"][0], y["zone1"][8] + y["zone1"][0], 0])
        / 2,  # l9
    ]
    surfMass["zone1"] = [height * length for length in arcMass["zone1"]]
    surfCoM["zone1"] = [rc + np.array([0, 0, height / 2]) for rc in arcCoM["zone1"]]
    # MESH STARTS HERE #
    points["zone1"] = [
        gmsh.model.occ.addPoint(xi, yi, 0, lc) for xi, yi in zip(x["zone1"], y["zone1"])
    ]
    curves["zone1"] = [
        gmsh.model.occ.addCircleArc(points["zone1"][0], p0, points["zone1"][1])
    ] + [
        gmsh.model.occ.addLine(pi, pj)
        for pi, pj in zip(
            points["zone1"][1:], points["zone1"][2:] + [points["zone1"].pop(0)]
        )
    ]
    loops["zone1"] = [gmsh.model.occ.addCurveLoop(curves["zone1"])]
    surfs["zone1"] = [gmsh.model.occ.addPlaneSurface([cl]) for cl in loops["zone1"]]
    gmsh.model.occ.synchronize()  # synchronize model

    # ----------------------------- #
    # ---------- ZONE 2 ----------- #
    # 4 points, 4 curves, 1 surface #
    # After fragment operation:     #
    # 6 points, 6 curves, 1 surface #
    # ----------------------------- #
    x["zone2"] = [
        Ri_ref * cos(beta),  # x1
        Re_ref * cos(beta),  # x2
        Re_ref * cos(beta),  # x3
        Ri_ref * cos(beta),  # x4
    ]
    y["zone2"] = [
        -Ri_ref * sin(beta),  # y1
        -Re_ref * sin(beta),  # y2
        Re_ref * sin(beta),  # y3
        Ri_ref * sin(beta),  # y4
    ]
    arcMass["zone2"] = [
        Re_ref - Ri_ref,  # l1
        Re_ref * gamma,  # l2
        Re_ref - Ri_ref,  # l3
        Ri_ref * (beta - alpha),  # l4
        2 * Ri_ref * alpha,  # l5
        Ri_ref * (beta - alpha),  # l6
    ]
    arcCoM["zone2"] = [
        (Ri_ref + Re_ref) * np.array([cos(-beta), sin(-beta), 0]) / 2,  # l1
        np.array([2 * Re_ref**2 * sin(beta), 0, 0]) / arcMass["zone2"][1],  # l2
        (Ri_ref + Re_ref) * np.array([cos(beta), sin(beta), 0]) / 2,  # l3
        Ri_ref**2
        * np.array([sin(beta) - sin(alpha), cos(alpha) - cos(beta), 0])
        / arcMass["zone2"][3],  # l4
        np.array([2 * Ri_ref**2 * sin(alpha), 0, 0]) / arcMass["zone2"][4],  # l5
        Ri_ref**2
        * np.array([sin(-alpha) - sin(-beta), cos(-beta) - cos(-alpha), 0])
        / arcMass["zone2"][5],  # l6
    ]
    surfMass["zone2"] = [height * length for length in arcMass["zone2"]]
    surfCoM["zone2"] = [rc + np.array([0, 0, height / 2]) for rc in arcCoM["zone2"]]
    # MESH STARTS HERE #
    points["zone2"] = [
        gmsh.model.occ.addPoint(xi, yi, 0, lc) for xi, yi in zip(x["zone2"], y["zone2"])
    ]
    curves["zone2"] = [
        gmsh.model.occ.addLine(points["zone2"][0], points["zone2"][1]),
        gmsh.model.occ.addCircleArc(points["zone2"][1], p0, points["zone2"][2]),
        gmsh.model.occ.addLine(points["zone2"][2], points["zone2"][3]),
        gmsh.model.occ.addCircleArc(points["zone2"][0], p0, points["zone2"][3]),
    ]
    loops["zone2"] = [
        gmsh.model.occ.addCurveLoop(curves["zone2"][:-1] + [-curves["zone2"][3]])
    ]
    surfs["zone2"] = [gmsh.model.occ.addPlaneSurface([cl]) for cl in loops["zone2"]]
    gmsh.model.occ.synchronize()  # synchronize model

    # ----------------------------- #
    # ------------ GAP ------------ #
    # 4 points, 4 curves, 1 surface #
    # ----------------------------- #
    x["gap"] = [
        Re_ref * cos(beta),  # x1
        (Re_ref + g) * cos(beta),  # x2
        (Re_ref + g) * cos(beta),  # x3
        Re_ref * cos(beta),  # x4
    ]
    y["gap"] = [
        -Re_ref * sin(beta),  # y1
        -(Re_ref + g) * sin(beta),  # y2
        (Re_ref + g) * sin(beta),  # y3
        Re_ref * sin(beta),  # y4
    ]
    arcMass["gap"] = [
        g,  # l1
        (Re_ref + g) * gamma,  # l2
        g,  # l3
        Re_ref * gamma,  # l4
    ]
    arcCoM["gap"] = [
        np.array([x["gap"][0] + x["gap"][1], y["gap"][0] + y["gap"][1], 0]) / 2,  # l1
        np.array([2 * (Re_ref + g) ** 2 * sin(beta), 0, 0]) / arcMass["gap"][1],  # l2
        np.array([x["gap"][2] + x["gap"][3], y["gap"][2] + y["gap"][3], 0]) / 2,  # l3
        np.array([2 * Re_ref**2 * sin(beta), 0, 0]) / arcMass["gap"][3],  # l4
    ]
    surfMass["gap"] = [height * length for length in arcMass["gap"]]
    surfCoM["gap"] = [rc + np.array([0, 0, height / 2]) for rc in arcCoM["gap"]]
    volMass["gap"] = [height * gamma * ((Re_ref + g) ** 2 - Re_ref**2) / 2]
    volCoM["gap"] = [
        np.array(
            [
                2
                * sin(beta)
                * ((Re_ref + g) ** 3 - Re_ref**3)
                / (3 * beta * ((Re_ref + g) ** 2 - Re_ref**2)),
                0,
                height / 2,
            ]
        )
    ]
    # MESH STARTS HERE #
    points["gap"] = [
        gmsh.model.occ.addPoint(xi, yi, 0, lc) for xi, yi in zip(x["gap"], y["gap"])
    ]
    curves["gap"] = [
        gmsh.model.occ.addLine(points["gap"][0], points["gap"][1]),
        gmsh.model.occ.addCircleArc(points["gap"][1], p0, points["gap"][2]),
        gmsh.model.occ.addLine(points["gap"][2], points["gap"][3]),
        gmsh.model.occ.addCircleArc(points["gap"][0], p0, points["gap"][3]),
    ]
    loops["gap"] = [
        gmsh.model.occ.addCurveLoop(curves["gap"][:-1] + [-curves["gap"][3]])
    ]
    surfs["gap"] = [gmsh.model.occ.addPlaneSurface([cl]) for cl in loops["gap"]]
    gmsh.model.occ.synchronize()  # synchronize model

    # ----------------------------- #
    # ---------- BARREL ----------- #
    # 4 points, 4 curves, 1 surface #
    # ----------------------------- #
    x["barrel"] = [
        (Re_ref + g) * cos(beta),  # x1
        Re_bar * cos(beta),  # x2
        Re_bar * cos(beta),  # x3
        (Re_ref + g) * cos(beta),  # x4
    ]
    y["barrel"] = [
        -(Re_ref + g) * sin(beta),  # y1
        -Re_bar * sin(beta),  # y2
        Re_bar * sin(beta),  # y3
        (Re_ref + g) * sin(beta),  # y4
    ]
    arcMass["barrel"] = [
        Re_bar - Re_ref - g,  # l1
        Re_bar * gamma,  # l2
        Re_bar - Re_ref - g,  # l3
        (Re_ref + g) * gamma,  # l4
    ]
    arcCoM["barrel"] = [
        np.array([x["barrel"][0] + x["barrel"][1], y["barrel"][0] + y["barrel"][1], 0])
        / 2,  # l1
        np.array([2 * Re_bar**2 * sin(beta), 0, 0]) / arcMass["barrel"][1],  # l2
        np.array([x["barrel"][2] + x["barrel"][3], y["barrel"][2] + y["barrel"][3], 0])
        / 2,  # l3
        np.array([2 * (Re_ref + g) ** 2 * sin(beta), 0, 0])
        / arcMass["barrel"][3],  # l4
    ]
    surfMass["barrel"] = [height * length for length in arcMass["barrel"]]
    surfCoM["barrel"] = [rc + np.array([0, 0, height / 2]) for rc in arcCoM["barrel"]]
    volMass["barrel"] = [height * gamma * (Re_bar**2 - (Re_ref + g) ** 2) / 2]
    volCoM["barrel"] = [
        np.array(
            [
                2
                * sin(beta)
                * (Re_bar**3 - (Re_ref + g) ** 3)
                / (3 * beta * (Re_bar**2 - (Re_ref + g) ** 2)),
                0,
                height / 2,
            ]
        )
    ]
    # MESH STARTS HERE #
    points["barrel"] = [
        gmsh.model.occ.addPoint(xi, yi, 0, lc)
        for xi, yi in zip(x["barrel"], y["barrel"])
    ]
    curves["barrel"] = [
        gmsh.model.occ.addLine(points["barrel"][0], points["barrel"][1]),
        gmsh.model.occ.addCircleArc(points["barrel"][1], p0, points["barrel"][2]),
        gmsh.model.occ.addLine(points["barrel"][2], points["barrel"][3]),
        gmsh.model.occ.addCircleArc(points["barrel"][0], p0, points["barrel"][3]),
    ]
    loops["barrel"] = [
        gmsh.model.occ.addCurveLoop(curves["barrel"][:-1] + [-curves["barrel"][3]])
    ]
    surfs["barrel"] = [gmsh.model.occ.addPlaneSurface([cl]) for cl in loops["barrel"]]
    gmsh.model.occ.synchronize()  # synchronize model

    # --------------------------- #
    # BOOLEAN OPERATION: FRAGMENT #
    # --------------------------- #
    # CUT: make holes.
    if len(holes) > 0:
        sHoles = []
        for rc, d in holes:
            sHoles.append(gmsh.model.occ.addDisk(rc[0], rc[1], rc[2], d / 2, d / 2))
        # Perform boolean operation...
        gmsh.model.occ.cut(
            [(2, tag) for tag in surfs["zone1"] + surfs["zone2"]],
            [(2, tag) for tag in sHoles],
        )
    # FRAGMENT: fuse parts of the domain.
    gmsh.model.occ.fragment([(2, surfs["gap"][0])], [(2, surfs["barrel"][0])])
    gmsh.model.occ.fragment([(2, surfs["zone2"][0])], [(2, surfs["gap"][0])])
    gmsh.model.occ.fragment([(2, surfs["zone1"][0])], [(2, surfs["zone2"][0])])
    gmsh.model.occ.synchronize()  # synchronize model

    # ------------------------------------ #
    # SEARCH ENTITIES: Transfinite purpose #
    # ------------------------------------ #
    # HOLES ENTITIES
    points["holes"], curves["holes"] = [], []
    for rc, d in holes:
        points["holes"] += gmsh.model.occ.getEntitiesInBoundingBox(
            xmin=rc[0] - d / 2 - tol,
            ymin=rc[1] - d / 2 - tol,
            zmin=-tol,
            xmax=rc[0] + d / 2 + tol,
            ymax=rc[1] + d / 2 + tol,
            zmax=tol,
            dim=0,
        )
        curves["holes"] += gmsh.model.occ.getEntitiesInBoundingBox(
            xmin=rc[0] - d / 2 - tol,
            ymin=rc[1] - d / 2 - tol,
            zmin=-tol,
            xmax=rc[0] + d / 2 + tol,
            ymax=rc[1] + d / 2 + tol,
            zmax=tol,
            dim=1,
        )
    points["holes"] = list(set([tag for _, tag in points["holes"]]))
    curves["holes"] = list(set([tag for _, tag in curves["holes"]]))

    # REMAINING ENTITIES
    ## curves, dim = 1
    currCurves = curves["holes"]
    allCurves = [tag for _, tag in gmsh.model.getEntities(dim=1)]
    remCurves = list(set(allCurves) - set(currCurves))
    remCurves_mass_com = [
        (
            gmsh.model.occ.getMass(1, tag),
            np.array(gmsh.model.occ.getCenterOfMass(1, tag)),
        )
        for tag in remCurves
    ]
    ### ZONE 1
    points["zone1"], curves["zone1"] = [], []
    for mass, rc in zip(arcMass["zone1"], arcCoM["zone1"]):
        for i, (other_mass, other_rc) in enumerate(remCurves_mass_com):
            if isclose(mass, other_mass) and all(np.isclose(rc, other_rc)):
                curves["zone1"].append(remCurves[i])
                _, downward = gmsh.model.getAdjacencies(1, remCurves[i])
                points["zone1"] += list(downward)
                continue
    points["zone1"] = list(set(points["zone1"]))
    ### ZONE 2
    points["zone2"], curves["zone2"] = [], []
    for mass, rc in zip(arcMass["zone2"], arcCoM["zone2"]):
        for i, (other_mass, other_rc) in enumerate(remCurves_mass_com):
            if isclose(mass, other_mass) and all(np.isclose(rc, other_rc)):
                curves["zone2"].append(remCurves[i])
                _, downward = gmsh.model.getAdjacencies(1, remCurves[i])
                points["zone2"] += list(downward)
                continue
    points["zone2"] = list(set(points["zone2"]))
    ### GAP
    points["gap"], curves["gap"] = [], []
    for mass, rc in zip(arcMass["gap"], arcCoM["gap"]):
        for i, (other_mass, other_rc) in enumerate(remCurves_mass_com):
            if isclose(mass, other_mass) and all(np.isclose(rc, other_rc)):
                curves["gap"].append(remCurves[i])
                _, downward = gmsh.model.getAdjacencies(1, remCurves[i])
                points["gap"] += list(downward)
                continue
    points["gap"] = list(set(points["gap"]))
    ### BARREL
    points["barrel"], curves["barrel"] = [], []
    for mass, rc in zip(arcMass["barrel"], arcCoM["barrel"]):
        for i, (other_mass, other_rc) in enumerate(remCurves_mass_com):
            if isclose(mass, other_mass) and all(np.isclose(rc, other_rc)):
                curves["barrel"].append(remCurves[i])
                _, downward = gmsh.model.getAdjacencies(1, remCurves[i])
                points["barrel"] += list(downward)
                continue
    points["barrel"] = list(set(points["barrel"]))
    ## surfaces, dim = 2
    remSurfs = [tag for _, tag in gmsh.model.getEntities(dim=2)]
    remSurfs_mass_com = [
        (
            gmsh.model.occ.getMass(2, tag),
            np.array(gmsh.model.occ.getCenterOfMass(2, tag)),
        )
        for tag in remSurfs
    ]
    ### ZONE 1
    surfs["zone1"], _ = gmsh.model.getAdjacencies(dim=1, tag=curves["zone1"][1])
    ### ZONE 2
    surfs["zone2"], _ = gmsh.model.getAdjacencies(dim=1, tag=curves["zone2"][0])
    ### GAP
    surfs["gap"], _ = gmsh.model.getAdjacencies(dim=1, tag=curves["gap"][0])
    ### BARREL
    surfs["barrel"], _ = gmsh.model.getAdjacencies(dim=1, tag=curves["barrel"][0])

    # ----------------------------- #
    # TRANSFINITE CURVES & SURFACES #
    # ----------------------------- #
    # Set structured grid parameters on hook, barrel and gap zones.
    ## Gap zone curves.
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["gap"][0], numNodes=Nr)  # l1
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["gap"][1], numNodes=Ntheta)  # l2
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["gap"][2], numNodes=Nr)  # l3
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["gap"][3], numNodes=Ntheta)  # l4
    ## Barrel zone curves.
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["barrel"][0], numNodes=Nr)  # l1
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["barrel"][1], numNodes=Ntheta)  # l2
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["barrel"][2], numNodes=Nr)  # l3
    gmsh.model.mesh.setTransfiniteCurve(tag=curves["barrel"][3], numNodes=Ntheta)  # l4

    ## Gap zone surface
    gmsh.model.mesh.setTransfiniteSurface(tag=surfs["gap"][0])
    ## Barrel zone surface
    gmsh.model.mesh.setTransfiniteSurface(tag=surfs["barrel"][0])

    # Synchronize model...
    gmsh.model.occ.synchronize()

    # Extrude all surfaces...
    entities = gmsh.model.occ.getEntities(2)
    heights = [1] if form == "uniform" else [0.25, 0.55, 1]
    print(f"INFO:  3D Mesh - nz={nz}\theights={heights}")
    gmsh.model.occ.extrude(entities, 0, 0, height, nz, heights)
    print(f"INFO:  3D Mesh has been extruded!")

    # Remove duplicate surfaces...
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    # ---------------------------------------- #
    # SEARCH ENTITIES: Physical groups purpose #
    # ---------------------------------------- #
    # HOLES ENTITIES
    holeSurfs = []
    for rc, d in holes:
        holeSurfs += gmsh.model.occ.getEntitiesInBoundingBox(
            xmin=rc[0] - d / 2 - tol,
            ymin=rc[1] - d / 2 - tol,
            zmin=-tol,
            xmax=rc[0] + d / 2 + tol,
            ymax=rc[1] + d / 2 + tol,
            zmax=height + tol,
            dim=2,
        )
    holeSurfs = list(set([tag for _, tag in holeSurfs]))
    # REMAINING ENTITIES
    ## surfaces, dim = 2
    currSurfs = holeSurfs
    allSurfs = [tag for _, tag in gmsh.model.getEntities(dim=2)]
    remSurfs = list(set(allSurfs) - set(currSurfs))
    remSurfs_mass_com = [
        (
            gmsh.model.occ.getMass(2, tag),
            np.array(gmsh.model.occ.getCenterOfMass(2, tag)),
        )
        for tag in remSurfs
    ]
    ### INNER BOUNDARY ENTITIES
    innerSurfs = []
    innerMass = surfMass["zone1"][1:] + [surfMass["zone2"][3], surfMass["zone2"][5]]
    innerCoM = surfCoM["zone1"][1:] + [surfCoM["zone2"][3], surfCoM["zone2"][5]]
    for mass, rc in zip(innerMass, innerCoM):
        for i, (other_mass, other_rc) in enumerate(remSurfs_mass_com):
            if isclose(mass, other_mass) and all(np.isclose(rc, other_rc)):
                innerSurfs.append(remSurfs[i])
                continue
    ### OUTER BOUNDARY ENTITIES
    outerSurfs = []
    outerMass = [surfMass["barrel"][1]]
    outerCoM = [surfCoM["barrel"][1]]
    for mass, rc in zip(outerMass, outerCoM):
        for i, (other_mass, other_rc) in enumerate(remSurfs_mass_com):
            if isclose(mass, other_mass) and all(np.isclose(rc, other_rc)):
                outerSurfs.append(remSurfs[i])
                continue
    ### SIDE BOUNDARY ENTITIES
    sideSurfs = []
    sideMass = [
        surfMass["zone2"][0],
        surfMass["zone2"][2],
        surfMass["gap"][0],
        surfMass["gap"][2],
        surfMass["barrel"][0],
        surfMass["barrel"][2],
    ]
    sideCoM = [
        surfCoM["zone2"][0],
        surfCoM["zone2"][2],
        surfCoM["gap"][0],
        surfCoM["gap"][2],
        surfCoM["barrel"][0],
        surfCoM["barrel"][2],
    ]
    for mass, rc in zip(sideMass, sideCoM):
        for i, (other_mass, other_rc) in enumerate(remSurfs_mass_com):
            if isclose(mass, other_mass) and all(np.isclose(rc, other_rc)):
                sideSurfs.append(remSurfs[i])
                continue
    ### BOTTOM BOUNDARY ENTITIES
    bottomSurfs = gmsh.model.occ.getEntitiesInBoundingBox(
        xmin=0,
        ymin=min(y["barrel"]) - tol,
        zmin=-tol,
        xmax=Re_bar + tol,
        ymax=max(y["barrel"]) + tol,
        zmax=tol,
        dim=2,
    )
    bottomSurfs = [tag for _, tag in bottomSurfs]
    ### TOP BOUNDARY ENTITIES
    topSurfs = gmsh.model.occ.getEntitiesInBoundingBox(
        xmin=0,
        ymin=min(y["barrel"]) - tol,
        zmin=height - tol,
        xmax=Re_bar + tol,
        ymax=max(y["barrel"]) + tol,
        zmax=height + tol,
        dim=2,
    )
    topSurfs = [tag for _, tag in topSurfs]

    ## volumes, dim = 3
    allVols = [tag for _, tag in gmsh.model.getEntities(dim=3)]
    vols_mass_com = [
        (
            gmsh.model.occ.getMass(3, tag),
            np.array(gmsh.model.occ.getCenterOfMass(3, tag)),
        )
        for tag in allVols
    ]
    ### BARREL ENTITY
    vols["barrel"] = []
    for mass, rc in zip(volMass["barrel"], volCoM["barrel"]):
        for i, (other_mass, other_rc) in enumerate(vols_mass_com):
            if isclose(mass, other_mass) and all(np.isclose(rc, other_rc)):
                vols["barrel"].append(allVols[i])
                continue
    ### GAP ENTITY
    vols["gap"] = []
    for mass, rc in zip(volMass["gap"], volCoM["gap"]):
        for i, (other_mass, other_rc) in enumerate(vols_mass_com):
            if isclose(mass, other_mass) and all(np.isclose(rc, other_rc)):
                vols["gap"].append(allVols[i])
                continue
    ### ZONE 2 ENTITY
    vols["zone2"], _ = gmsh.model.getAdjacencies(2, surfs["zone2"][0])
    ### ZONE 1 ENTITY
    currVols = [
        vols["barrel"][0],
        vols["gap"][0],
        vols["zone2"][0],
    ]
    vols["zone1"] = list(set(allVols) - set(currVols))

    # --------------- #
    # PHYSICAL GROUPS #
    # --------------- #
    # Inner BC -> Robin type.
    innerTag = gmsh.model.addPhysicalGroup(2, innerSurfs)
    gmsh.model.setPhysicalName(2, innerTag, "inner")

    # Outer BC -> Robin type.
    outerTag = gmsh.model.addPhysicalGroup(2, outerSurfs)
    gmsh.model.setPhysicalName(2, outerTag, "outer")

    # Side BC -> Neumann type.
    sideTag = gmsh.model.addPhysicalGroup(2, sideSurfs)
    gmsh.model.setPhysicalName(2, sideTag, "side")

    # Bottom BC -> Dirichlet type.
    bottomTag = gmsh.model.addPhysicalGroup(2, bottomSurfs)
    gmsh.model.setPhysicalName(2, bottomTag, "bottom")

    # Top BC -> Robin type.
    topTag = gmsh.model.addPhysicalGroup(2, topSurfs)
    gmsh.model.setPhysicalName(2, topTag, "top")

    # Channels BC -> Robin type.
    for i, channel in enumerate(holeSurfs):
        chTag = gmsh.model.addPhysicalGroup(2, [channel])
        gmsh.model.setPhysicalName(2, chTag, "channel" + str(int(i + 1)))

    # Domain part: Zone 1
    vols["zone1"] = [vols["zone1"][0]]
    zOneTag = gmsh.model.addPhysicalGroup(3, vols["zone1"])
    gmsh.model.setPhysicalName(3, zOneTag, "zoneOne")

    # Domain part: Zone 2
    zTwoTag = gmsh.model.addPhysicalGroup(3, vols["zone2"])
    gmsh.model.setPhysicalName(3, zTwoTag, "zoneTwo")

    # Domain part: Gap
    gapTag = gmsh.model.addPhysicalGroup(3, vols["gap"])
    gmsh.model.setPhysicalName(3, gapTag, "gap")

    # Domain part: Barrel
    barrelTag = gmsh.model.addPhysicalGroup(3, vols["barrel"])
    gmsh.model.setPhysicalName(3, barrelTag, "barrel")

    # Set colors for different BC:
    gmsh.model.setColor([(2, tag) for tag in innerSurfs], 255, 0, 0)  # Red - Inner
    gmsh.model.setColor([(2, tag) for tag in outerSurfs], 0, 255, 0)  # Green - Outer
    gmsh.model.setColor([(2, tag) for tag in topSurfs], 0, 0, 255)  # Blue - Top
    gmsh.model.setColor([(2, tag) for tag in bottomSurfs], 0, 0, 255)  # Blue - Bottom
    gmsh.model.setColor([(2, tag) for tag in sideSurfs], 255, 255, 0)  # Yellow - Side
    gmsh.model.setColor([(2, tag) for tag in holeSurfs], 0, 255, 0)  # Green - Channel

    # Assign a mesh size to hole points:
    gmsh.model.mesh.setSize([(0, tag) for tag in points["zone1"]], lc)
    gmsh.model.mesh.setSize([(0, tag) for tag in points["holes"]], lch)

    # Set the order of the elements in the mesh.
    gmsh.model.mesh.setOrder(1)

    # Generate a 2D mesh...
    gmsh.model.mesh.generate(3)

    # ... and save it to disk
    # gmsh.write(fileName=src_path + "test_3D.msh")

    # ... and save it to disk
    gmsh.write(fileName=root + fileName + ".msh")

    # Launch the GUI to see the results:
    # if "-nopopup" not in sys.argv:
    #    gmsh.fltk.run()

    gmsh.model.remove()

    gmsh.finalize()


class Reflector(object):
    hc = {"inner": 6640, "outer": 1420, "top": None, "bottom": None}
    Tref = {"inner": 595.95, "outer": 555.15, "top": 595.95, "bottom": 555.15}

    def __init__(
        self,
        root: str,  # root address
        folderName: str,  # folder name
        fileName: str,  # file name
        modelName: str = "model_0",  # gmsh model name
        height: float = 1,  # reflector's height in meters
        nz: int = 10,  # discretized on z axis
        holes: list = [],  # hole[i] = (rc, d)
        lc: float = 10e-3,  # characteristic length
        lch: float = 1e-3,  # channel characteristic length
        Ntheta: int = 100,  # number of elements on theta axis
        Nr: int = 5,  # number of elements on r axis
        embed: list = [],  # embed[i] = (rc, lc), embedded points
        form: str = "uniform",  # distributed z elements
    ) -> None:
        self.root = root  # root address of main folder
        self.folderName = folderName  # main folder name
        self.rootFolderAdr = root + folderName + "/"  # absolute main folder address
        self.mshFolderAdr = root + folderName + "/mesh/"  # mesh folder address
        self.solFolderAdr = root + folderName + "/solution/"  # solutions folder address
        self.asciiFolderAdr = root + folderName + "/ascii/"  # ascii folder address
        self.fileName = fileName  # mesh file name
        self.modelName = modelName  # gmsh model name
        self.height = height  # reflector's height in meters
        self.nz = nz  # mesh discretized on z axis
        self.holes = holes  # reflector channels
        self.lc = lc  # main characteristic length
        self.lch = lch  # channels characteristic length
        self.Ntheta = Ntheta  # discretized on theta axis
        self.Nr = Nr  # discretized on radial axis
        self.embed = embed  # local refine mesh

        # Build tree:
        #       folderName/
        #           |- mesh/
        #           |   |- base/
        #           |- solution/
        #           |   |- state/
        #           |   |- adjoint/
        #           |- ascii/
        # ----------------------------- #
        subprocess.run(["mkdir", self.rootFolderAdr])
        subprocess.run(["mkdir", "-p", self.mshFolderAdr + "base/"])
        subprocess.run(["mkdir", "-p", self.solFolderAdr + "state/"])
        subprocess.run(["mkdir", "-p", self.solFolderAdr + "adjoint/"])
        subprocess.run(["mkdir", "-p", self.asciiFolderAdr])

        # Build 2D base mesh (optimization purpose)
        baseMesh(
            root=self.mshFolderAdr + "base/",
            fileName=fileName,
            modelName=modelName,
            holes=holes,
            lc=lc,
            lch=lch,
            Ntheta=Ntheta,
            Nr=Nr,
            embed=embed,
        )

        # Build 3D mesh (solve FEM problem)
        extrudeMesh(
            root=self.mshFolderAdr,
            fileName=fileName,
            modelName=modelName,
            height=height,
            nz=nz,
            holes=holes,
            lc=lc,
            lch=lch,
            Ntheta=Ntheta,
            Nr=Nr,
            embed=embed,
            form=form,
        )

        self.baseMesh = mesh.Mesh(self.mshFolderAdr + f"base/{fileName}.msh")
        self.mesh = mesh.Mesh(self.mshFolderAdr + f"{fileName}.msh")

    def compHeatProfiles(
        self,
        centers: list,
        diameter: float = 10e-3,
        Nz: float = 20,
        Ntheta: float = 10,
    ) -> Tuple[list, np.ndarray]:
        # Main variables
        qn_z = []  # heat flux pofiles list
        Nch = len(centers)  # total channel's number
        R = diameter / 2  # hole's radius

        # Channel coordinates on prime reference system.
        z = np.linspace(0, self.height, Nz, endpoint=True)
        theta = np.linspace(0, 2 * pi, Ntheta, endpoint=False)
        x_prime = R * np.cos(theta)
        y_prime = R * np.sin(theta)
        z_prime = np.zeros(Ntheta)
        r_prime = np.c_[x_prime, y_prime, z_prime]
        n = -r_prime / R

        # Reshape coordinates.
        ## Normal vector.
        nn = np.repeat(np.array([n]), repeats=Nz, axis=0)
        nn = np.reshape(nn, (Ntheta * Nz, 3))
        nn = np.repeat(np.array([nn]), repeats=Nch, axis=0)
        nn = np.reshape(nn, (Ntheta * Nz * Nch, 3))
        ## Prime system coordinates.
        rr_prime = np.repeat(np.array([r_prime]), repeats=Nz, axis=0)
        rr_prime = np.reshape(rr_prime, (Ntheta * Nz, 3))
        rr_prime = np.repeat(np.array([rr_prime]), repeats=Nch, axis=0)
        rr_prime = np.reshape(rr_prime, (Ntheta * Nz * Nch, 3))
        ## z-coordinates (not prime system).
        zz = np.repeat(np.reshape(z, (1, Nz)), repeats=Ntheta, axis=0)
        zz = np.reshape(zz, (Ntheta * Nz, 1), order="F")
        zz = np.repeat(np.array([zz]), repeats=Nch, axis=0)
        zz = np.reshape(zz, (Ntheta * Nz * Nch, 1))
        ## Centers.
        rc = np.repeat(centers, Ntheta * Nz, axis=0)
        rc[:, 2] = zz[:, 0]  # refresh zc by its true z.

        # Points to interpolate values.
        points = rc + rr_prime

        # Open and extract heat flux q''(x,y,z).
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(self.mshFolderAdr + self.fileName + ".msh")
        gmsh.merge(self.solFolderAdr + "state/q.msh")
        viewTags = gmsh.view.getTags()
        data = gmsh.view.getModelData(viewTags[0], 1)
        gmsh.finalize()
        heatFlux = np.reshape(np.array(data[2]), (data[1].size, 3))
        nodeCoords = np.array([node.r for node in self.mesh.nodes.values()])

        # Interpolate heat flux.
        q_xyz = griddata(nodeCoords, heatFlux, points, method="linear")
        # Obtain normal heat flux.
        qn_xyz = np.sum(q_xyz * nn, axis=1)
        # Reshape normal heat flux --> by channel.
        qn_xyz_chs = np.reshape(qn_xyz, (Nch, Ntheta * Nz))
        # Integrate over theta --> qn'(z)
        for qn_xyz_ch in qn_xyz_chs:
            qn_xyz_ch = np.reshape(qn_xyz_ch, (Nz, Ntheta))
            qn_z.append(R * np.sum(qn_xyz_ch, axis=1))

        return qn_z, z

    def fieldToBase(
        self,
        field: np.ndarray,  # field to proyect on base
        method: str = "min",  # operation over z axis
        output: str = "dict",  # output format
    ) -> Tuple[dict, np.ndarray]:
        field = field[np.lexsort((field[:, 1], field[:, 0]))]
        _, start = np.unique(field[:, 0], return_index=True)
        end = np.roll((start - 1) % field.shape[0], -1)
        idxs = np.column_stack((start, end))
        points, values = [], []
        for idx in idxs:
            points.append([field[idx[0], 0], field[idx[0], 1]])
            if method == "max":
                values.append(np.amax(field[idx[0] : idx[1], 3]))
            else:
                values.append(np.amin(field[idx[0] : idx[1], 3]))
        points = np.array(points)
        values = np.array(values)
        ri = np.array([node.r for node in self.baseMesh.nodes.values()])
        ri = np.delete(ri, 2, axis=1)
        iValues = griddata(points, values, ri, method="nearest")  # linear

        numNodes = len(self.baseMesh.nodes)
        if output == "array":
            fieldBase = iValues
        else:
            fieldBase = {tag: dt for tag, dt in zip(range(1, numNodes + 1), iValues)}

        return fieldBase

    def filterField(
        self, field: dict, groupOff: list, base: bool = False, reverse: bool = False
    ) -> list:
        unwantedNodeTags = []
        myMesh = self.baseMesh if base else self.mesh
        for pgName in groupOff:
            unwantedNodeTags += myMesh.physicalGroups[pgName].getNodeTags()
        unwantedNodeTags = np.array(list(set(unwantedNodeTags)), dtype="int32") - 1

        numNodes = len(myMesh.nodes)
        wantedNodeTags = np.arange(1, numNodes + 1, dtype="int32")
        wantedNodeTags = list(np.delete(wantedNodeTags, unwantedNodeTags))
        wantedData = itemgetter(*wantedNodeTags)(field)

        fieldByNodeTag = [(tag, val) for tag, val in zip(wantedNodeTags, wantedData)]
        fieldByNodeTag.sort(key=lambda x: x[1], reverse=reverse)

        return fieldByNodeTag

    def fieldFilter(
        self,
        field: str = "T",  # weight field, 'q' or 'T'
        fraction: float = 0.5,  # node fraction to filter
        nodeTags: list = [],  # input nodes to filter
        base: bool = False,  # filter nodes mesh
        method: str = "max",  # if base is settled
    ) -> list:
        meshFile = self.mshFolderAdr + self.fileName + ".msh"
        solFile = self.solFolderAdr + "state/"
        # Extract field
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(meshFile)
        solFile = solFile + "T.msh" if field == "T" else solFile + "q.msh"
        gmsh.merge(solFile)
        viewTags = gmsh.view.getTags()
        data = gmsh.view.getModelData(viewTags[0], 1)
        gmsh.finalize()
        numC = 1 if field == "T" else 3
        fieldData = np.reshape(np.array(data[2]), (data[1].size, numC))
        # Extract node coords from mesh
        nodeCoords = np.array([node.r for node in self.mesh.nodes.values()])
        # Compute weights
        normFieldData = np.linalg.norm(fieldData, axis=1)
        if len(nodeTags) == 0:
            nodeTags = (
                list(self.baseMesh.nodes.keys())
                if base
                else list(self.mesh.nodes.keys())
            )
        if base:
            normFieldData = np.c_[nodeCoords, normFieldData]
            normFieldData = self.fieldToBase(normFieldData, method)
            normFieldData = np.array(itemgetter(*nodeTags)(normFieldData))
            normFieldData = normFieldData / np.amax(normFieldData)
        else:
            meshData = {
                tag: value for tag, value in zip(self.mesh.nodes.keys(), normFieldData)
            }
            normFieldData = np.array(itemgetter(*nodeTags)(meshData))
            normFieldData = normFieldData / np.amax(normFieldData)
        nodeTags = random.choices(
            nodeTags, normFieldData, k=int(fraction * len(nodeTags))
        )
        nodeTags = list(set(nodeTags))

        return nodeTags

    def geomFilter(
        self,
        tol: float = 20e-3,
        boundOff: str = [],
        domainOff: str = [],
        base: bool = True,
    ) -> list:
        groupOff = boundOff + domainOff
        myMesh = self.baseMesh if base else self.mesh
        entityOff = []
        for bound in boundOff:
            entityOff += list(myMesh.physicalGroups[bound].entityTags)
        entityOff = list(set(entityOff))
        allNodeTags = np.array(list(myMesh.nodes.keys()), dtype="int32")
        if len(groupOff) == 0:
            onNodeTags = allNodeTags
        else:
            offNodeTags = []
            for phyGrp in groupOff:
                offNodeTags += myMesh.physicalGroups[phyGrp].getNodeTags()
            offNodeTags = np.array(list(set(offNodeTags)), dtype="int32") - 1
            onNodeTags = list(np.delete(allNodeTags, offNodeTags))
        nodesCond = myMesh.checkCircleHole(tol, onNodeTags, entityOff)
        onNodeTags = list(compress(onNodeTags, nodesCond))
        return onNodeTags

    def getTemp(self):
        mshFile = self.mshFolderAdr + self.fileName + ".msh"
        tempFile = self.solFolderAdr + "state/T.msh"

        # Open and extract temperature T(x,y,z).
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(mshFile)
        gmsh.merge(tempFile)
        viewTags = gmsh.view.getTags()
        data = gmsh.view.getModelData(viewTags[0], 1)
        gmsh.finalize()
        tempByNodeTag = {int(tag): temp[0] for tag, temp in zip(list(data[1]), data[2])}

        # Get maximum temperature
        maxTempNodeTag = max(tempByNodeTag, key=tempByNodeTag.get)
        maxTempNodeCoord = self.mesh.nodes[maxTempNodeTag].r
        maxTemp = tempByNodeTag[maxTempNodeTag]

        # Get minimum temperature
        minTempNodeTag = min(tempByNodeTag, key=tempByNodeTag.get)
        minTempNodeCoord = self.mesh.nodes[minTempNodeTag].r
        minTemp = tempByNodeTag[minTempNodeTag]

        return (minTempNodeCoord, minTemp), (maxTempNodeCoord, maxTemp)

    def optimizeByTemp(
        self, tol: float, method: str = "max", boundOff: list = [], domainOff: list = []
    ) -> None:
        mshFile = self.mshFolderAdr + self.fileName + ".msh"
        tempFile = self.solFolderAdr + "state/T.msh"
        groupOff = boundOff + domainOff
        reverse = True if method == "max" else False

        # Extract temperature field.
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(mshFile)
        gmsh.merge(tempFile)
        viewTags = gmsh.view.getTags()
        data = gmsh.view.getModelData(viewTags[0], 1)
        gmsh.finalize()
        temp = np.reshape(np.array(data[2]), data[1].size)
        nodeCoords = np.array([node.r for node in self.mesh.nodes.values()])

        # T(x,y,z) ON 3D MESH (PLANE Z = 0) -> 2D BASE MESH #
        tempCoords = np.c_[nodeCoords, temp]
        tempByNodeTag = self.fieldToBase(tempCoords, method)

        # Filter temperature by groups
        tempByNodeTag = self.filterField(tempByNodeTag, groupOff, True, reverse)
        # PERFORM the i-th HOLE #
        tempTags = [node[0] for node in tempByNodeTag]
        holeTag, _ = self.baseMesh.computeCircHole(tempTags, tol, check=False)

        return holeTag, self.baseMesh.nodes[holeTag].r

    def optimize(
        self, tol: float, method: str = "max", boundOff: list = [], domainOff: list = []
    ) -> None:
        mshFile = self.mshFolderAdr + self.fileName + ".msh"
        topDervFile = self.solFolderAdr + "adjoint/DT.msh"
        groupOff = boundOff + domainOff
        reverse = True if method == "max" else False

        # Extract topological derivative field.
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(mshFile)
        gmsh.merge(topDervFile)
        viewTags = gmsh.view.getTags()
        data = gmsh.view.getModelData(viewTags[0], 1)
        gmsh.finalize()
        topDerv = np.reshape(np.array(data[2]), data[1].size)
        nodeCoords = np.array([node.r for node in self.mesh.nodes.values()])

        # DT(x,y,z) ON 3D MESH (PLANE Z = 0) -> 2D BASE MESH #
        topDervCoords = np.c_[nodeCoords, topDerv]
        topDervByNodeTag = self.fieldToBase(topDervCoords, method)
        ## Write down DT(x,y,z=0) in an ascii file.
        lines = []
        for tag, value in topDervByNodeTag.items():
            r = self.baseMesh.nodes[tag].r
            lines.append(f"{r[0]}\t{r[1]}\t{value}\n")
        open(self.asciiFolderAdr + f"baseDT.dat", "w").close()  # Erase content...
        f = open(self.asciiFolderAdr + f"baseDT.dat", "wt")
        f.writelines(lines)
        f.close()
        ## Run feenox and write it into vtk file
        self.writeBaseDTFeenox()
        feeFile = self.rootFolderAdr + "baseDT.fee"
        baseMeshFile = f"{self.mshFolderAdr}base/{self.fileName}.msh"
        subprocess.run(["feenox", feeFile, baseMeshFile])
        ## Filter topological derivative by groups
        topDervByNodeTag = self.filterField(topDervByNodeTag, groupOff, True, reverse)
        # PERFORM the i-th HOLE #
        topDervTags = [node[0] for node in topDervByNodeTag]
        holeTag, _ = self.baseMesh.computeCircHole(topDervTags, tol, check=False)

        return holeTag, self.baseMesh.nodes[holeTag].r

    def setRobinBC(
        self,
        hc: dict = {},  # hc = {group: value (scalar|None)}
        Tref: dict = {},  # Tref = {group: value (scalar|None)}
        hc_holes: list = [],  # hc_h = [hc_1, ..., hc_n] (scalar)
        Tref_holes: list = [],  # [Tref_h]_ij, i:hole, j:z
        Ntheta: int = 10,  # element's number on theta axis
        Nz: int = 20,  # element's number on z axis
    ) -> None:
        # Set boundary condition except on channels
        self.hc = Reflector.hc if len(hc) == 0 else hc
        self.Tref = Reflector.Tref if len(Tref) == 0 else Tref

        # Set boundary condition on each channel
        # Build hc and Tref ascii file for each channel.
        Nch = len(self.holes)
        if Nch > 0:
            R = self.holes[0][1] / 2
            centers = [hole[0] for hole in self.holes]

            # Channel coordinates on prime reference system.
            z = np.linspace(0, self.height, Nz, endpoint=True)
            theta = np.linspace(0, 2 * pi, Ntheta, endpoint=False)
            x_prime = R * np.cos(theta)
            y_prime = R * np.sin(theta)
            z_prime = np.zeros(Ntheta)
            r_prime = np.c_[x_prime, y_prime, z_prime]

            # Reshape coordinates.
            ## Prime system coordinates.
            rr_prime = np.repeat(np.array([r_prime]), repeats=Nz, axis=0)
            rr_prime = np.reshape(rr_prime, (Ntheta * Nz, 3))
            rr_prime = np.repeat(np.array([rr_prime]), repeats=Nch, axis=0)
            rr_prime = np.reshape(rr_prime, (Ntheta * Nz * Nch, 3))
            ## z-coordinates (not prime system).
            zz = np.repeat(np.reshape(z, (1, Nz)), repeats=Ntheta, axis=0)
            zz = np.reshape(zz, (Ntheta * Nz, 1), order="F")
            zz = np.repeat(np.array([zz]), repeats=Nch, axis=0)
            zz = np.reshape(zz, (Ntheta * Nz * Nch, 1))
            ## Centers.
            rc = np.repeat(centers, Ntheta * Nz, axis=0)
            rc[:, 2] = zz[:, 0]  # refresh zc by its true z.
            ## hc in hole's points.
            hc_holes = np.repeat(hc_holes, Ntheta * Nz, axis=0)
            hc_holes = np.reshape(hc_holes, (Nch, Ntheta * Nz))
            ## Tref in hole's points.
            Tref_chs = []
            for Tref_hole in Tref_holes:
                Tref_hole = np.repeat(
                    np.reshape(Tref_hole, (1, Nz)), repeats=Ntheta, axis=0
                )
                Tref_hole = np.reshape(Tref_hole, (Ntheta * Nz), order="F")
                Tref_chs.append(Tref_hole)

            points = rc + rr_prime  # hole's points
            points = np.reshape(points, (Nch, Ntheta * Nz, 3))  # reshape by hole

            # Write down each channel's hc ascii file.
            for i, (points_hole, hc_hole) in enumerate(zip(points, hc_holes)):
                lines, fileName = [], f"hc_{i+1}.dat"
                for (xi, yi, zi), hc_i in zip(points_hole, hc_hole):
                    lines.append(f"{xi}\t{yi}\t{zi}\t{hc_i}\n")
                open(self.asciiFolderAdr + fileName, "w").close()  # Erase content...
                f = open(self.asciiFolderAdr + fileName, "wt")
                f.writelines(lines)
                f.close()

            # Write down each channel's Tref ascii file.
            for i, (points_hole, Tref_ch) in enumerate(zip(points, Tref_chs)):
                lines, fileName = [], f"Tref_{i+1}.dat"
                for (xi, yi, zi), Tref_i in zip(points_hole, Tref_ch):
                    lines.append(f"{xi}\t{yi}\t{zi}\t{Tref_i}\n")
                open(self.asciiFolderAdr + fileName, "w").close()  # Erase content...
                f = open(self.asciiFolderAdr + fileName, "wt")
                f.writelines(lines)
                f.close()

    def solveAdjoint(self) -> None:
        # Solve adjoint equation and compute topological derivative.
        # Call Feenox and solve the adjoint problem (adjoint eq.).
        feeFile = self.rootFolderAdr + "adjoint.fee"
        mshFile = self.mshFolderAdr + self.fileName + ".msh"
        result = subprocess.run(
            ["feenox", feeFile, mshFile],
            stdout=subprocess.PIPE,
        )

    def solveState(self) -> list:
        # Solve state equation and return: cost function, current volume.
        # Call Feenox and solve the direct problem (state eq.).
        feeFile = self.rootFolderAdr + "state.fee"
        mshFile = self.mshFolderAdr + self.fileName + ".msh"
        result = subprocess.run(
            ["feenox", feeFile, mshFile],
            stdout=subprocess.PIPE,
        )
        result = result.stdout.decode("utf-8")
        result = result.split("|")
        result = [float(val) for val in result]

        return result

    def writeAdjointFeenox(self, costFunc: str = "d") -> None:
        # costFunc: select cost function type
        # "a"-> internal energy
        # "b"-> external source work
        # "c"-> total potential energy
        # "d" -> transient energy

        fileName = self.rootFolderAdr + "adjoint.fee"
        open(fileName, "w").close()  # erase content

        # Main file parameters...without channels.
        headers = ["PROBLEM thermal 3d \n", "READ_MESH $1 \n"]
        variables = [
            "kWater=0.574\t# in [W/mK] \n",
            "kSteal=16.3\t# in [W/mK] \n",
            "inTref=555.15\t# in [K] (282 °C) \n",
            "outTref=595.95\t# in [K] (322.8 °C) \n",
            "inWallhc=6640\t# in [W/m²K] \n",
            "outWallhc=1420\t# in [W/m²K] \n",
            "qGap=0.26e6\t# in [W/m³] \n",
            "qBarrel=0.27e6\t# in [W/m³] \n",
        ]
        if costFunc == "d":
            functions = [
                f"FUNCTION u(x,y,z) FILE {self.asciiFolderAdr}domain.dat \n",
                f"FUNCTION innerT(x,y,z) FILE {self.asciiFolderAdr}inner.dat \n",
                f"FUNCTION outerT(x,y,z) FILE {self.asciiFolderAdr}outer.dat \n",
                f"FUNCTION topT(x,y,z) FILE {self.asciiFolderAdr}top.dat \n",
                f"FUNCTION hc(x,y,z) FILE {self.asciiFolderAdr}hc.dat \n",
                f"FUNCTION Tref(x,y,z) FILE {self.asciiFolderAdr}Tref.dat \n",
                "FUNCTION inWallTref(x,y,z) = 552.43 + 48.338*z - 10.682*z*z - 1.604*z*z*z \t# in [K] \n",
                "FUNCTION qZoneOne(x,y,z) = 278448 + 4470118*z - 4351753*z*z + 1005712*z*z*z \t# in [W/m³] \n",
                "FUNCTION qZoneTwo(x,y,z) = 100268 + 1518109*z - 1431195*z*z + 313786*z*z*z \t# in [W/m³] \n",
            ]
        else:
            functions = [
                f"FUNCTION u(x,y,z) FILE {self.asciiFolderAdr}domain.dat \n",
                f"FUNCTION hc(x,y,z) FILE {self.asciiFolderAdr}hc.dat \n",
                f"FUNCTION Tref(x,y,z) FILE {self.asciiFolderAdr}Tref.dat \n",
            ]
        volumes = [
            "# 	      name    | conductivity | power density \n",
            "MATERIAL zoneOne   k=kSteal       q=-2*kSteal*qZoneOne(x,y,z) \n",
            "MATERIAL zoneTwo   k=kSteal       q=-2*kSteal*qZoneTwo(x,y,z) \n",
            "MATERIAL gap       k=kWater       q=-2*kWater*qGap \n",
            "MATERIAL barrel    k=kSteal       q=-2*kSteal*qBarrel \n",
        ]
        # old version
        # "MATERIAL barrel    k=kSteal       q=-2*qBarrel \n",
        # "BC inner      h=inWallhc   Tref=-2*kSteal*(innerT(x,y,z)-inWallTref(x,y,z)) \n",
        bc = [
            "#  name     | condition \n",
            "BC inner      h=inWallhc   Tref=2*kSteal*(innerT(x,y,z)-inWallTref(x,y,z)) \n",
            "BC outer      h=outWallhc  Tref=2*kSteal*(outerT(x,y,z)-outTref) \n",
            "BC side       q=0 \n",
            "BC top        h=inWallhc   Tref=2*kSteal*(topT(x,y,z)-outTref) \n",
            "BC bottom     T=0 \n",
        ]

        # Define extra functions depends on cost function
        if costFunc == "a":
            extraFunctions = [
                "FUNCTION DT(x,y,z) = hc(x,y,z)*T(x,y,z)*(u(x,y,z) - 2*Tref(x,y,z)) \n",
            ]
        elif costFunc == "b":
            extraFunctions = [
                "FUNCTION DT(x,y,z) = hc(x,y,z)*T(x,y,z)*(u(x,y,z) - 2*Tref(x,y,z)) \n",
            ]
        elif costFunc == "c":
            extraFunctions = [
                "FUNCTION DT(x,y,z) = -hc(x,y,z)*T(x,y,z)*(u(x,y,z) - 2*Tref(x,y,z)) \n",
            ]
        else:
            extraFunctions = [
                "FUNCTION DT(x,y,z) = -hc(x,y,z)*T(x,y,z)*(u(x,y,z) - Tref(x,y,z)) \n",
            ]

        # Channels file parameters...
        chFunctions = []
        for i in range(1, len(self.holes) + 1):
            chFunctions.append(
                f"FUNCTION ch{i}T(x,y,z) FILE {self.asciiFolderAdr}channel{i}.dat \n"
            )
            chFunctions.append(
                f"FUNCTION Tref{i}(x,y,z) FILE {self.asciiFolderAdr}Tref_{i}.dat \n"
            )
            chFunctions.append(
                f"FUNCTION hc{i}(x,y,z) FILE {self.asciiFolderAdr}hc_{i}.dat \n"
            )
        chBC = [
            f"BC channel{i} h=hc{i}(x,y,z)  Tref=2*kSteal*(ch{i}T(x,y,z)-Tref{i}(x,y,z)) \n"
            for i in range(1, len(self.holes) + 1)
        ]

        # Write feenox file from python...
        if costFunc == "d":
            f = open(fileName, "wt")
            f.writelines(headers)
            f.write("\n# Define some variables \n")
            f.writelines(variables)
            f.write("\n# Define some functions \n")
            f.writelines(functions)
            f.writelines(chFunctions)
            f.write("\n# Volume properties \n")
            f.writelines(volumes)
            f.write("\n# Boundary conditions \n")
            f.writelines(bc)
            f.writelines(chBC)
            f.write("\n# Run and solve the heat conduction problem \n")
            f.write("SOLVE_PROBLEM \n")
            f.write("\n# Define some extra functions after solving \n")
            f.writelines(extraFunctions)
            # ---------------------
            save = self.solFolderAdr + "adjoint/"
            f.write("\n# Write down the results in .msh format file \n")
            f.write(
                f"WRITE_MESH {save}lambda.msh MESH $1 FILE_FORMAT gmsh NODE T(x,y,z)\n"
            )
            f.write(
                f"WRITE_MESH {save}DT.msh MESH $1 FILE_FORMAT gmsh NODE DT(x,y,z)\n"
            )
            f.write(
                f"WRITE_MESH {save}hc.msh MESH $1 FILE_FORMAT gmsh NODE hc(x,y,z)\n"
            )
            f.write(
                f"WRITE_MESH {save}Tref.msh MESH $1 FILE_FORMAT gmsh NODE Tref(x,y,z)\n"
            )
            # ---------------------
            f.write("\n# Write down the results in .vtk format file \n")
            f.write(
                f"WRITE_MESH {save}lambda.vtk MESH $1 FILE_FORMAT vtk NODE T(x,y,z)\n"
            )
            f.write(f"WRITE_MESH {save}DT.vtk MESH $1 FILE_FORMAT vtk NODE DT(x,y,z)\n")
            f.write(f"WRITE_MESH {save}hc.vtk MESH $1 FILE_FORMAT vtk NODE hc(x,y,z)\n")
            f.write(
                f"WRITE_MESH {save}Tref.vtk MESH $1 FILE_FORMAT vtk NODE Tref(x,y,z)\n"
            )
            # ---------------------
            f.close()
        else:
            f = open(fileName, "wt")
            f.writelines(headers)
            f.write("\n# Define some functions \n")
            f.writelines(functions)
            f.write("\n# Define some extra functions \n")
            f.writelines(extraFunctions)
            # ---------------------
            save = self.solFolderAdr + "adjoint/"
            f.write("\n# Write down the results in .msh format file \n")
            f.write(
                f"WRITE_MESH {save}DT.msh MESH $1 FILE_FORMAT gmsh NODE DT(x,y,z)\n"
            )
            # ---------------------
            f.write("\n# Write down the results in .vtk format file \n")
            f.write(f"WRITE_MESH {save}DT.vtk MESH $1 FILE_FORMAT vtk NODE DT(x,y,z)\n")
            # ---------------------
            f.close()

    def writeBaseDTFeenox(self) -> None:
        fileName = self.rootFolderAdr + "baseDT.fee"
        open(fileName, "w").close()  # erase content

        headers = ["PROBLEM thermal 2d \n", "READ_MESH $1 \n"]
        functions = [f"FUNCTION baseDT(x,y) FILE {self.asciiFolderAdr}baseDT.dat \n"]

        # Write feenox file from python...
        f = open(fileName, "wt")
        f.writelines(headers)
        f.write("\n# Define some functions \n")
        f.writelines(functions)
        # ---------------------
        save = self.solFolderAdr + "adjoint/"
        f.write("\n# Write down the results in .msh format file \n")
        f.write(
            f"WRITE_MESH {save}baseDT.msh MESH $1 FILE_FORMAT gmsh NODE baseDT(x,y)\n"
        )
        # ---------------------
        f.write("\n# Write down the results in .vtk format file \n")
        f.write(
            f"WRITE_MESH {save}baseDT.vtk MESH $1 FILE_FORMAT vtk NODE baseDT(x,y)\n"
        )
        # ---------------------
        f.close()

    def writeBCMap(self, points: list, hc: list, Tref: list) -> None:
        hc_lines, Tref_lines = [], []
        Nz = Tref[0].size
        z = np.linspace(0, self.height, Nz, endpoint=True)
        for rc, hc_rc, Tref_rc in zip(points, hc, Tref):
            rc = np.repeat(np.array([rc]), Nz, axis=0)
            rc[:, 2] = z
            for rc_z, Tref_rc_z in zip(rc, Tref_rc):
                hc_lines.append(f"{rc_z[0]}\t{rc_z[1]}\t{rc_z[2]}\t{hc_rc}\n")
                Tref_lines.append(f"{rc_z[0]}\t{rc_z[1]}\t{rc_z[2]}\t{Tref_rc_z}\n")
        # Write hc file
        open(self.asciiFolderAdr + f"hc.dat", "w").close()  # Erase content...
        f = open(self.asciiFolderAdr + f"hc.dat", "wt")
        f.writelines(hc_lines)
        f.close()
        # Write Tref file
        open(self.asciiFolderAdr + f"Tref.dat", "w").close()  # Erase content...
        f = open(self.asciiFolderAdr + f"Tref.dat", "wt")
        f.writelines(Tref_lines)
        f.close()

    def writeScalarField(
        self,
        fieldByNodeTag: dict,  # field = {tag: value},
        mesh: Any,  # consistent with the field
        address: str,  # saved absolute path
        groups: list = [],  # physical groups
    ) -> None:
        groups = mesh.physicalGroups.keys() if len(groups) == 0 else groups
        for group in groups:
            # For each group ...
            ## Extract node tags from mesh.
            nodeTags = mesh.physicalGroups[group].getNodeTags()
            ## Write down the field over the group in an ascii file.
            lines = []
            for tag in nodeTags:
                x = mesh.nodes[tag].x
                y = mesh.nodes[tag].y
                z = mesh.nodes[tag].z
                field = fieldByNodeTag[tag]
                lines.append(f"{x}\t{y}\t{z}\t{field}\n")
            open(address + f"{group}.dat", "w").close()  # Erase content...
            f = open(address + f"{group}.dat", "wt")
            f.writelines(lines)
            f.close()

    def writeStateFeenox(self, costFunc: str = "d") -> None:
        # costFunc: select cost function type
        # "a"-> internal energy
        # "b"-> external source work
        # "c"-> total potential energy
        # "d" -> transient energy

        fileName = self.rootFolderAdr + "state.fee"
        open(fileName, "w").close()  # erase content

        # Main file parameters...without channels.
        headers = ["PROBLEM thermal 3d \n", "READ_MESH $1 \n"]
        variables = [
            "kWater=0.676\t# in [W/mK] \n",
            "kSteal=16.27\t# in [W/mK] \n",
            "inTref=558.35\t# in [K] (285.2 °C) \n",
            "outTref=595.25\t# in [K] (322.1 °C) \n",
            "inWallhc=6640\t# in [W/m²K] \n",
            "outWallhc=1420\t# in [W/m²K] \n",
            "topWallhc=1795\t# in [W/m²K] \n",
            "qGap=0.26e6\t# in [W/m³] \n",
            "qBarrel=0.27e6\t# in [W/m³] \n",
        ]
        functions = [
            "FUNCTION inWallTref(x,y,z) = 552.43 + 48.338*z - 10.682*z*z - 1.604*z*z*z \t# in [K] \n",
            "FUNCTION qZoneOne(x,y,z) = 278448 + 4470118*z - 4351753*z*z + 1005712*z*z*z \t# in [W/m³] \n",
            "FUNCTION qZoneTwo(x,y,z) = 100268 + 1518109*z - 1431195*z*z + 313786*z*z*z \t# in [W/m³] \n",
        ]
        volumes = [
            "# 	      name    | conductivity | power density \n",
            "MATERIAL zoneOne   k=kSteal       q=qZoneOne(x,y,z) \n",
            "MATERIAL zoneTwo   k=kSteal       q=qZoneTwo(x,y,z) \n",
            "MATERIAL gap       k=kWater       q=qGap \n",
            "MATERIAL barrel    k=kSteal       q=qBarrel \n",
        ]
        bc = [
            "#  name     | condition \n",
            "BC inner      h=inWallhc   Tref=inWallTref(x,y,z) \n",
            "BC outer      h=outWallhc  Tref=inTref \n",
            "BC side       q=0 \n",
            "BC top        h=inWallhc   Tref=outTref \n",
            "BC bottom     T=inTref \n",
        ]
        extraFunctions = [
            "FUNCTION sqrGradT(x,y,z) = dTdx(x,y,z)^2 + dTdy(x,y,z)^2 + dTdz(x,y,z)^2 \n",
            "FUNCTION qInner(x,y,z) = inWallhc*(T(x,y,z)-inWallTref(x,y,z)) \n",
            "FUNCTION qOuter(x,y,z) = outWallhc*(T(x,y,z)-inTref) \n",
            "FUNCTION qTop(x,y,z) = inWallhc*(T(x,y,z)-outTref) \n",
        ]
        integrals = [
            "INTEGRATE 1 OVER zoneOne MESH $1 CELLS RESULT vol1 \n",
            "INTEGRATE 1 OVER zoneTwo MESH $1 CELLS RESULT vol2 \n",
            "INTEGRATE q(x,y,z) MESH $1 NODES RESULT Q \n",
            "INTEGRATE qInner(x,y,z) OVER inner MESH $1 NODES RESULT innerQ \n",
            "INTEGRATE qOuter(x,y,z) OVER outer MESH $1 NODES RESULT outerQ \n",
            "INTEGRATE qTop(x,y,z) OVER top MESH $1 NODES RESULT topQ \n",
        ]

        # Add extra functions and integrals depends on cost function type
        if costFunc == "a":
            extraFunctions += [
                "FUNCTION aDom(x,y,z)=k(x,y,z)*sqrGradT(x,y,z) \n",
                "FUNCTION aInner(x,y,z)=inWallhc*T(x,y,z)*T(x,y,z) \n",
                "FUNCTION aOuter(x,y,z)=outWallhc*T(x,y,z)*T(x,y,z) \n",
                "FUNCTION aTop(x,y,z)=inWallhc*T(x,y,z)*T(x,y,z) \n",
            ]
            integrals += [
                "INTEGRATE aDom(x,y,z) OVER zoneOne MESH $1 CELLS RESULT aDomIntOne \n",
                "INTEGRATE aDom(x,y,z) OVER zoneTwo MESH $1 CELLS RESULT aDomIntTwo \n",
                "INTEGRATE aInner(x,y,z) OVER inner MESH $1 CELLS RESULT aInnerInt \n",
                "INTEGRATE aOuter(x,y,z) OVER outer MESH $1 CELLS RESULT aOuterInt \n",
                "INTEGRATE aTop(x,y,z) OVER top MESH $1 CELLS RESULT aTopInt \n",
            ]
        elif costFunc == "b":
            extraFunctions += [
                "FUNCTION lDom(x,y,z)=q(x,y,z)*T(x,y,z) \n",
                "FUNCTION lInner(x,y,z)=inWallhc*inWallTref(x,y,z)*T(x,y,z) \n",
                "FUNCTION lOuter(x,y,z)=outWallhc*inTref*T(x,y,z) \n",
                "FUNCTION lTop(x,y,z)=inWallhc*inWallTref(x,y,z)*T(x,y,z) \n",
            ]
            integrals += [
                "INTEGRATE lDom(x,y,z) OVER zoneOne MESH $1 CELLS RESULT lDomIntOne \n",
                "INTEGRATE lDom(x,y,z) OVER zoneTwo MESH $1 CELLS RESULT lDomIntTwo \n",
                "INTEGRATE lInner(x,y,z) OVER inner MESH $1 CELLS RESULT lInnerInt \n",
                "INTEGRATE lOuter(x,y,z) OVER outer MESH $1 CELLS RESULT lOuterInt \n",
                "INTEGRATE lTop(x,y,z) OVER top MESH $1 CELLS RESULT lTopInt \n",
            ]
        elif costFunc == "c":
            extraFunctions += [
                "FUNCTION aDom(x,y,z)=k(x,y,z)*sqrGradT(x,y,z) \n",
                "FUNCTION aInner(x,y,z)=inWallhc*T(x,y,z)*T(x,y,z) \n",
                "FUNCTION aOuter(x,y,z)=outWallhc*T(x,y,z)*T(x,y,z) \n",
                "FUNCTION aTop(x,y,z)=inWallhc*T(x,y,z)*T(x,y,z) \n",
                "FUNCTION lDom(x,y,z)=q(x,y,z)*T(x,y,z) \n",
                "FUNCTION lInner(x,y,z)=inWallhc*inWallTref(x,y,z)*T(x,y,z) \n",
                "FUNCTION lOuter(x,y,z)=outWallhc*inTref*T(x,y,z) \n",
                "FUNCTION lTop(x,y,z)=inWallhc*inWallTref(x,y,z)*T(x,y,z) \n",
            ]
            integrals += [
                "INTEGRATE aDom(x,y,z) OVER zoneOne MESH $1 CELLS RESULT aDomIntOne \n",
                "INTEGRATE aDom(x,y,z) OVER zoneTwo MESH $1 CELLS RESULT aDomIntTwo \n",
                "INTEGRATE aInner(x,y,z) OVER inner MESH $1 CELLS RESULT aInnerInt \n",
                "INTEGRATE aOuter(x,y,z) OVER outer MESH $1 CELLS RESULT aOuterInt \n",
                "INTEGRATE aTop(x,y,z) OVER top MESH $1 CELLS RESULT aTopInt \n",
                "INTEGRATE lDom(x,y,z) OVER zoneOne MESH $1 CELLS RESULT lDomIntOne \n",
                "INTEGRATE lDom(x,y,z) OVER zoneTwo MESH $1 CELLS RESULT lDomIntTwo \n",
                "INTEGRATE lInner(x,y,z) OVER inner MESH $1 CELLS RESULT lInnerInt \n",
                "INTEGRATE lOuter(x,y,z) OVER outer MESH $1 CELLS RESULT lOuterInt \n",
                "INTEGRATE lTop(x,y,z) OVER top MESH $1 CELLS RESULT lTopInt \n",
            ]
        else:
            integrals += [
                "INTEGRATE k(x,y,z)*k(x,y,z)*sqrGradT(x,y,z) OVER zoneOne MESH $1 NODES RESULT cf1 \n",
                "INTEGRATE k(x,y,z)*k(x,y,z)*sqrGradT(x,y,z) OVER zoneTwo MESH $1 NODES RESULT cf2 \n",
            ]

        # Channels file parameters...
        chFunctions = []
        for i in range(1, len(self.holes) + 1):
            chFunctions.append(
                f"FUNCTION Tref{i}(x,y,z) FILE {self.asciiFolderAdr}Tref_{i}.dat \n"
            )
            chFunctions.append(
                f"FUNCTION hc{i}(x,y,z) FILE {self.asciiFolderAdr}hc_{i}.dat \n"
            )
        chBC = [
            f"BC channel{i} h=hc{i}(x,y,z)  Tref=Tref{i}(x,y,z) \n"
            for i in range(1, len(self.holes) + 1)
        ]
        extraChFunctions = [
            f"FUNCTION qCh{i}(x,y,z) = hc{i}(x,y,z)*(T(x,y,z)-Tref{i}(x,y,z)) \n"
            for i in range(1, len(self.holes) + 1)
        ]
        chIntegrals = [
            f"INTEGRATE qCh{i}(x,y,z) OVER channel{i} MESH $1 NODES RESULT ch{i}Q \n"
            for i in range(1, len(self.holes) + 1)
        ]

        # Add functions and integrals on channels
        # depends on cost function type
        if costFunc == "a":
            extraChFunctions += [
                f"FUNCTION aCh{i}(x,y,z) = hc{i}(x,y,z)*T(x,y,z)*T(x,y,z) \n"
                for i in range(1, len(self.holes) + 1)
            ]
            chIntegrals += [
                f"INTEGRATE aCh{i}(x,y,z) OVER channel{i} MESH $1 NODES RESULT aCh{i}Int \n"
                for i in range(1, len(self.holes) + 1)
            ]
        elif costFunc == "b":
            extraChFunctions += [
                f"FUNCTION lCh{i}(x,y,z) = hc{i}(x,y,z)*Tref{i}(x,y,z)*T(x,y,z) \n"
                for i in range(1, len(self.holes) + 1)
            ]
            chIntegrals += [
                f"INTEGRATE lCh{i}(x,y,z) OVER channel{i} MESH $1 NODES RESULT lCh{i}Int \n"
                for i in range(1, len(self.holes) + 1)
            ]
        elif costFunc == "c":
            extraChFunctions += [
                f"FUNCTION aCh{i}(x,y,z) = hc{i}(x,y,z)*T(x,y,z)*T(x,y,z) \n"
                for i in range(1, len(self.holes) + 1)
            ]
            extraChFunctions += [
                f"FUNCTION lCh{i}(x,y,z) = hc{i}(x,y,z)*Tref{i}(x,y,z)*T(x,y,z) \n"
                for i in range(1, len(self.holes) + 1)
            ]
            chIntegrals += [
                f"INTEGRATE aCh{i}(x,y,z) OVER channel{i} MESH $1 NODES RESULT aCh{i}Int \n"
                for i in range(1, len(self.holes) + 1)
            ]
            chIntegrals += [
                f"INTEGRATE lCh{i}(x,y,z) OVER channel{i} MESH $1 NODES RESULT lCh{i}Int \n"
                for i in range(1, len(self.holes) + 1)
            ]
        else:
            pass

        # Define cost function variable
        if costFunc == "a":
            aChannel = [f"+ aCh{i}Int" for i in range(1, len(self.holes) + 1)]
            cf = "0.5*(aDomIntOne + aDomIntTwo + aInnerInt + aOuterInt + aTopInt "
            cf += " ".join(aChannel) + ")"
        elif costFunc == "b":
            lChannel = [f"+ lCh{i}Int" for i in range(1, len(self.holes) + 1)]
            cf = "lDomIntOne + lDomIntTwo + lInnerInt + lOuterInt + lTopInt "
            cf += " ".join(lChannel)
        elif costFunc == "c":
            aChannel = [f"+ aCh{i}Int" for i in range(1, len(self.holes) + 1)]
            lChannel = [f"- lCh{i}Int" for i in range(1, len(self.holes) + 1)]
            cf = "0.5*(aDomIntOne + aDomIntTwo + aInnerInt + aOuterInt + aTopInt "
            cf += " ".join(aChannel) + ") - "
            cf += "lDomIntOne - lDomIntTwo - lInnerInt - lOuterInt - lTopInt "
            cf += " ".join(lChannel)
        else:
            cf = "cf1 + cf2"

        # Write feenox file from python...
        f = open(fileName, "wt")
        f.writelines(headers)
        f.write("\n# Define some variables \n")
        f.writelines(variables)
        f.write("\n# Define some functions \n")
        f.writelines(functions)
        f.writelines(chFunctions)
        f.write("\n# Volume properties \n")
        f.writelines(volumes)
        f.write("\n# Boundary conditions \n")
        f.writelines(bc)
        f.writelines(chBC)
        f.write("\n# Run and solve the heat conduction problem \n")
        f.write("SOLVE_PROBLEM \n")
        f.write("\n# Define some extra functions after solving \n")
        f.writelines(extraFunctions)
        f.writelines(extraChFunctions)
        f.write("\n# Perform some integrals \n")
        f.writelines(integrals)
        f.writelines(chIntegrals)
        # ---------------------
        f.write("\n# Compute cost function cf depends on type \n")
        f.write(f"cf = {cf} \n")
        # ---------------------
        save = self.solFolderAdr + "state/"
        f.write("\n# Write down the results in .msh format file \n")
        f.write(f"WRITE_MESH {save}T.msh MESH $1 FILE_FORMAT gmsh NODE T(x,y,z)\n")
        f.write(
            f"WRITE_MESH {save}q.msh MESH $1 FILE_FORMAT gmsh NODE VECTOR qx qy qz\n"
        )
        # ---------------------
        f.write("\n# Write down the results in .vtk format file \n")
        f.write(f"WRITE_MESH {save}T.vtk MESH $1 FILE_FORMAT vtk NODE T(x,y,z)\n")
        f.write(
            f"WRITE_MESH {save}q.vtk MESH $1 FILE_FORMAT vtk NODE VECTOR qx qy qz\n"
        )
        # ---------------------
        f.write("\n# Write back the volume and cost function value to stdout... \n")
        f.write("vol = vol1 + vol2 \n")
        if len(self.holes) == 0:
            val_print = 'PRINT vol cf Q innerQ outerQ topQ SEP "|" \n'
        else:
            q_chs = " ".join([f"ch{i}Q" for i in range(1, len(self.holes) + 1)])
            val_print = f'PRINT vol cf Q innerQ outerQ topQ {q_chs} SEP "|" \n'
        f.write(val_print)
        # ---------------------
        f.close()

    def writeTemperature(self, groups: list = [], domain: bool = False) -> None:
        root = self.asciiFolderAdr
        mshFile = self.mshFolderAdr + self.fileName + ".msh"
        tempFile = self.solFolderAdr + "state/T.msh"
        groups = self.mesh.physicalGroups.keys() if len(groups) == 0 else groups

        # Open and extract temperature T(x,y,z).
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(mshFile)
        gmsh.merge(tempFile)
        viewTags = gmsh.view.getTags()
        data = gmsh.view.getModelData(viewTags[0], 1)
        gmsh.finalize()
        tempByNodeTag = {int(tag): temp[0] for tag, temp in zip(list(data[1]), data[2])}

        # Write down the Temperature over each group.
        for group in groups:
            # For each group ...
            ## Extract node tags from mesh.
            nodeTags = self.mesh.physicalGroups[group].getNodeTags()
            ## Write down the field over the group in an ascii file.
            lines = []
            for tag in nodeTags:
                x = self.mesh.nodes[tag].x
                y = self.mesh.nodes[tag].y
                z = self.mesh.nodes[tag].z
                temp = tempByNodeTag[tag]
                lines.append(f"{x}\t{y}\t{z}\t{temp}\n")
            open(root + f"{group}.dat", "w").close()  # Erase content...
            f = open(root + f"{group}.dat", "wt")
            f.writelines(lines)
            f.close()

        # Write down Temperature over the whole domain.
        if domain:
            lines = []
            for tag, temp in tempByNodeTag.items():
                x = self.mesh.nodes[tag].x
                y = self.mesh.nodes[tag].y
                z = self.mesh.nodes[tag].z
                lines.append(f"{x}\t{y}\t{z}\t{temp}\n")
            open(root + "domain.dat", "w").close()  # Erase content...
            f = open(root + "domain.dat", "wt")
            f.writelines(lines)
            f.close()
