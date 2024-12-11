import numpy as np
import collections, math, gmsh, sys
from typing import Tuple, Any, List, Union

from .elements.node import Node
from .elements.line2n import Line2n
from .elements.tri3n import Tri3n
from .elements.quad4n import Quad4n
from .elements.tetra4n import Tetra4n
from .elements.prism6n import Prism6n

from .physicalgroup import PhysicalGroup
from .entity import Entity

# key = type of element (see Gmsh's manual as a reference)
clsElement = {1: Line2n, 2: Tri3n, 3: Quad4n, 4: Tetra4n, 6: Prism6n, 15: Node}

class Mesh:
    def __init__(self, fileName: str):
        gmsh.initialize() 
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(fileName) # open the current model

        self.fileName = fileName
        self.modelName = gmsh.model.getCurrent() # get current model name
        self.modelDim = gmsh.model.getDimension() # get current model dimension

        # Get and save all the nodes.
        nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
        nodeCoords = nodeCoords.reshape(-1, 3)
        self.nodeTags = nodeTags
        self.nodes = {tag: Node(tag, r) for tag, r in zip(nodeTags, nodeCoords)}

        # Get and save all the elements.
        self.elementTypes = gmsh.model.mesh.getElementTypes()
        self.elementTagsByTypes = {}
        self.elementTags = []
        self.elements = {}

        for elemType in self.elementTypes:
            elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)
            elemNodeTags = elemNodeTags.reshape(-1, clsElement[elemType].numNodes)

            self.elementTagsByTypes[elemType] = elemTags
            self.elementTags += list(elemTags)
            if elemType == 15:
                for eTag, eNodeTags in zip(elemTags, elemNodeTags):
                    self.elements[eTag] = Node(eTag, self.nodes[eNodeTags[0]].r)
            else:
                for eTag, eNodeTags in zip(elemTags, elemNodeTags):
                    nodesByElement = [self.nodes[nodeTag] for nodeTag in eNodeTags]
                    self.elements[eTag] = clsElement[elemType](eTag, nodesByElement)
        self.elementTags.sort()

        # Create map (dict): nodeTag -> elementTags (elements sharing node)
        self.elemTagsByNodeTag = {nodeTag: [] for nodeTag in self.nodeTags}
        for elemTag, element in self.elements.items():
            if isinstance(element, Node):
                self.elemTagsByNodeTag[elemTag].append(elemTag)
            else:
                for nodeTag in element.nodeTags:
                    self.elemTagsByNodeTag[nodeTag].append(elemTag)

        # Get and save all entities. 
        self.entities = {dim: {} for dim in range(self.modelDim + 1)}
        dimTags = gmsh.model.getEntities()
        for dim, tag in dimTags:
            elemTypes, elemTags, _ = gmsh.model.mesh.getElements(dim, tag)
            self.entities[dim] = {tag: Entity(dim, tag, elemTypes, elemTags)}

        # Get and save all physical groups.
        dimTags = gmsh.model.getPhysicalGroups()
        names = [gmsh.model.getPhysicalName(dim, tag) for dim, tag in dimTags]
        self.physicalGroupNames = names
        self.physicalGroupsByName = {}
        self.physicalGroups = {dim: {} for dim in range(self.modelDim + 1)}
        for dimTag, name in zip(dimTags, names):
            dim = dimTag[0]; tag = dimTag[1]
            entityTags = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            self.physicalGroups[dim][tag] = PhysicalGroup(dim, tag, name, entityTags)
            self.physicalGroupsByName[name] = PhysicalGroup(dim, tag, name, entityTags)

        # Remove model and close Gmsh.
        gmsh.model.remove()
        gmsh.finalize()

    def __str__(self):
        s = f"{self.modelDim}D-Mesh\n"
        s += f"Model: {self.modelName}\nFile: {self.fileName}\n"
        s += f"#Nodes={len(self.nodes)} | #Elements={len(self.elements)}\n"
        n_entities = sum([len(entitiesByType) for entitiesByType in self.entities.values()])
        n_physicalGroups = len(self.physicalGroupsByName)
        s += f"#Entities={n_entities} | #PhysicalGroups={n_physicalGroups}\n"
        s += "-" * len(max(s.split("\n"), key=len))
        return s

    def __repr__(self):
        return self.__str__()

    def isElementInEntity(self, elemTag, entityDim, entityTag):
        return elemTag in self.entities[entityDim][entityTag].elementTags
    
    def isElementInPhysicalGroup(self, elemTag, name):
        phyGroup = self.physicalGroupsByName[name]
        for entityTag in phyGroup.entityTags:
            if self.isElementInEntity(elemTag, phyGroup.dim, entityTag):
                return True
        return False
    
    def isElementInPhysicalGroups(self, elemTag, names):
        if len(names) == 0:
            return False, None
        else:
            n=0; flag=False
            while (not flag) and (n < len(names)):
                flag = self.isElementInPhysicalGroup(elemTag, names[n])
                n += 1
            return True, names[n-1] if flag else False, None

    def checkExistanceElementByGroups(
        self, element: Any, groups: List[str]
    ) -> Union[str, None]:
        # Check if element, given by its tag, is on groups.
        # Return physical group found or None.
        # Devolver true/false + los grupos donde está o none
        Ng = len(groups)
        if Ng == 0:
            return None
        else:
            g, groupCond, name = 0, False, None
            while not groupCond and (g < Ng):
                elemCond = self.physicalGroups[groups[g]].find(element)
                groupCond = False if elemCond == -1 else True
                g += 1
            return groups[g - 1] if groupCond else None

    
    def isNodeInEntity(self, nodeTag, entityDim, entityTag):
        elementTagsWithNode = sorted(self.elemTagsByNodeTag[nodeTag])
        elementTagsInEntity = self.entities[entityDim][entityTag].elementTags
        ptr1=0; ptr2=0

        while ptr1 < len(elementTagsWithNode) and ptr2 < len(elementTagsInEntity):
            if elementTagsWithNode[ptr1] == elementTagsInEntity[ptr2]:
                return True
            elif elementTagsWithNode[ptr1] < elementTagsInEntity[ptr2]:
                ptr1 += 1
            else:
                ptr2 += 1
        return False
    
    def isNodeInPhysicalGroup(self, nodeTag, name):
        phyGroup = self.physicalGroupsByName[name]
        for entityTag in phyGroup.entityTags:
            if self.isNodeInEntity(nodeTag, phyGroup.dim, entityTag):
                return True
        return False
    
    def getMayorEntityForNode(self, nodeTag):
        nodeInEntities = []
        for entityDim, entitiesByTag in self.entities.items():
            for entityTag in entitiesByTag.keys():
                if self.isNodeInEntity(nodeTag, entityDim, entityTag):
                    nodeInEntities.append((entityDim, entityTag))
        if len(nodeInEntities) > 1:
            nodeInEntities.sort(reverse=True, key=lambda x: x[0])
            return nodeInEntities[0]
        elif len(nodeInEntities) == 0:
            return None
        else:
            return nodeInEntities[0]

    def getCommonElementTags(self, tags: list, otherTags: list) -> list:
        tags = set(tags)
        otherTags = set(otherTags)
        return list(tags.intersection(otherTags))

    def checkExistanceElementByGroups(
        self, element: Any, groups: List[str]
    ) -> Union[str, None]:
        # Check if element, given by its tag, is on groups.
        # Return physical group found or None.
        # Devolver true/false + los grupos donde está o none
        Ng = len(groups)
        if Ng == 0:
            return None
        else:
            g, groupCond, name = 0, False, None
            while not groupCond and (g < Ng):
                elemCond = self.physicalGroups[groups[g]].find(element)
                groupCond = False if elemCond == -1 else True
                g += 1
            return groups[g - 1] if groupCond else None

    def checkExistanceNodesByGroups(
        self, nodeTags: List[int], bounds: List[str] = [], domains: List[str] = []
    ) -> bool:
        # Check if nodes (given by its tags) are on non available
        # physical groups. For boundaries see on
        # 'boundOff'. For domains see on 'domainOff'.
        # If there is at least one of them in a single group,
        # the condition is considered satisfied.
        if len(bounds) > 0:
            bounds = [bound for bound in bounds if bound in self.physicalGroups.keys()]
        if len(domains) > 0:
            domains = [
                domain for domain in domains if domain in self.physicalGroups.keys()
            ]
        numBounds = len(bounds)
        numDomains = len(domains)
        # Exceptional condition: If both list groups are
        # empty, then return False.
        if (numBounds == 0) and (numDomains == 0):
            return False
        # At least one list group isn't empty.
        n, flag = 0, True
        boundCond, domainCond = False, False
        while flag and (n < len(nodeTags)):
            b, d = 0, 0
            nodeTag = nodeTags[n]
            if numBounds > 0:
                while not boundCond and (b < numBounds):
                    boundCond |= self.physicalGroups[bounds[b]].isNode(nodeTag)
                    b += 1
                if not boundCond:
                    if numDomains > 0:
                        while not domainCond and (d < numDomains):
                            domainCond |= self.physicalGroups[domains[d]].isNode(
                                nodeTag
                            )
                            d += 1
            else:
                if numDomains > 0:
                    while not domainCond and (d < numDomains):
                        domainCond |= self.physicalGroups[domains[d]].isNode(nodeTag)
                        d += 1
            flag = False if boundCond or domainCond else True
            n += 1
        return boundCond | domainCond

    def checkAloneNodeByElements(self, edges: list, others: list) -> bool:
        # Extract unique node tags from edges.
        nodeTags = list(
            dict.fromkeys([nodeTag for edge in edges for nodeTag in edge.nodeTags])
        )
        # Extract unique node tags from other edges.
        otherNodeTags = list(
            dict.fromkeys([nodeTag for edge in others for nodeTag in edge.nodeTags])
        )
        # Extract common node tags.
        commonNodeTags = list(set(nodeTags).intersection(set(otherNodeTags)))
        # Extract common edges and its node tags.
        commonEdges = self.getCommonElements(edges, others)
        nodeTagsCommEdges = list(
            dict.fromkeys(
                [nodeTag for edge in commonEdges for nodeTag in edge.nodeTags]
            )
        )

        return sorted(commonNodeTags) != sorted(nodeTagsCommEdges)

    def checkCircleHole(
        self, minDist: float, nodeTags: list = [], entityTags: list = []
    ) -> list:
        solution = []
        for nodeTag in nodeTags:
            # Compute minimum distance to each 1D entity (boundary).
            offCond, distByEntity = [], []
            if len(entityTags) == 0:
                # Calcula la distancia con respecto a todas las entidades 1D
                for entity in self.entities[1].values():
                    if entity.numElements() > 0:
                        r_ref = self.nodes[nodeTag].r
                        entityDist = entity.minDistance(r_ref)
                        offCond.append(entityDist > minDist)
            else:
                # Calcula la distancia respecto a las entidades dadas (< que todas)
                for entityTag in entityTags:
                    entity = self.entities[1][entityTag]
                    if entity.numElements() > 0:
                        r_ref = self.nodes[nodeTag].r
                        entityDist = entity.minDistance(r_ref)
                        offCond.append(entityDist > minDist)
            solution.append(all(offCond))

        return solution

    def computeCircHole(
        self,
        nodeTags: List[int],
        minDist: float,
        boundOff: List[str] = [],
        domainOff: List[str] = [],
        check: bool = True,
    ) -> int:
        nodeIdx = 0

        while nodeIdx < len(nodeTags):
            # STEP 0:
            # Check if node to kill is on non available physical groups.
            # For boundaries see on 'boundOff'. For domains see on 'domainOff'.
            if check:
                offCond = self.checkExistanceNodesByGroups(
                    [nodeTags[nodeIdx]], boundOff, domainOff
                )
                if offCond:
                    nodeIdx += 1
                    continue

            # STEP 1:
            # Compute minimum distance to each 1D entity (boundary).
            offCond, distByEntity = [], []
            for entity in self.entities[1].values():
                if entity.numElements() > 0:
                    r_ref = self.nodes[nodeTags[nodeIdx]].r
                    entityDist = entity.minDistance(r_ref)
                    distByEntity.append(
                        f"Entity {entity.tag}: rc={r_ref} \t min = {entityDist} \t d = {minDist}"
                    )
                    offCond.append(entityDist > minDist)
            if all(offCond):
                return nodeTags[nodeIdx], nodeIdx
            else:
                nodeIdx += 1
                continue

        return 0, -1
