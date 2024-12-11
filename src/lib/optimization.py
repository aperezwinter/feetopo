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


# ------------------ #
# OPTIMIZATION CLASS #
# ------------------ #

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
