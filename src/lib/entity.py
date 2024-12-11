import numpy as np
from typing_extensions import Self
from typing import Union, Any, Tuple, List

class Entity:
    def __init__(self, dim, tag, elementTypes, elementTags):
        self.dim = dim
        self.tag = tag
        self.elementTypes = elementTypes
        self.elementTagsByType = {eType: eTags for eType, eTags in zip(elementTypes, elementTags)}
        self.elementTags = np.reshape(np.array(elementTags), (-1))
        self.elementTags = sorted(list(self.elementTags))

    def __str__(self):
        n_elements = len(self.elementTags)
        minTag = min(self.elementTags)
        maxTag = max(self.elementTags)
        s = f"Entity({self.tag}): dim={self.dim} #Elements={n_elements} | {minTag} to {maxTag}\n"
        for eType in self.elementTypes:
            n_elements = np.size(self.elementTagsByType[eType])
            minTag = np.amin(self.elementTagsByType[eType])
            maxTag = np.amax(self.elementTagsByType[eType])
            s += f"Elements of type {eType}: #Elements={n_elements} | {minTag} to {maxTag}\n"
        s += "-" * len(max(s.split("\n"), key=len))
        return s

    def __repr__(self):
        return self.__str__()

    def isElement(self, tag: int) -> bool:
        return tag in list(self.elementTags)
    
    def isElementType(self, _type: int) -> bool:
        return _type in list(self.elementTypes)
    
    def findElementType(self, tag: int) -> int:
        for elemType, elementTags in self.elementTagsByType.items():
            if tag in elementTags:
                return elemType
        return -1

    def getCommonNodeTags(self, other: Self) -> list:
        selfNodeTags = set(self.getNodeTags())
        otherNodeTags = other.getNodeTags()
        return list(selfNodeTags.intersection(otherNodeTags))

    def getNumCommonNodes(self, other: Self) -> int:
        return len(self.getCommonNodeTags(other))

    def getElement(self, type: int, tag: int) -> Union[Any, None]:
        if type in self.elementTypes:
            return self.elements.get(tag)
        else:
            return None

    def getElements(self, type: int) -> list:
        if type in self.elementTypes:
            elemTags = self.elementTagsByType.get(type)
            elements = [self.elements.get(tag) for tag in elemTags]
            return elements
        else:
            return []

    def getElementTags(self, type: int) -> list:
        if type in self.elementTypes:
            return self.elementTagsByType.get(type)
        else:
            return []

    def getElementTagsByNode(self, tag: int) -> list:
        elemTags = []
        for elemTag in self.elementTags:
            if tag in self.elements[elemTag].nodeTags:
                elemTags.append(elemTag)
        return elemTags

    def getElementsByNode(self, tag: int) -> list:
        elements = []
        for elemTag in self.elementTags:
            if tag in self.elements[elemTag].nodeTags:
                elements.append(self.elements.get(elemTag))
        return elements

    def getNodeTags(self) -> List[Any]:
        nodeTags = [element.nodeTags for element in self.elements.values()]
        nodeTags = [
            nodeTag for nodeTagsByElem in nodeTags for nodeTag in nodeTagsByElem
        ]
        return nodeTags

    def minDistance(self, ref: np.ndarray = np.zeros(3)) -> float:
        d = []
        for element in self.elements.values():
            d.append(element.minDistance(ref))

        return min(d)

    def isElement(self, type: int, tag: int) -> bool:
        if type in self.elementTypes:
            if tag in self.elementTagsByType[type]:
                return True
            else:
                return False
        else:
            return False

    def isNode(self, tag: int) -> bool:
        flag, condition = True, False
        type_idx, tag_idx = 0, 0
        types_len = len(self.elementTypes)
        tags_len = [len(tags) for tags in self.elementTagsByType.values()]
        if types_len > 0:
            while flag and (type_idx < types_len):
                type = self.elementTypes[type_idx]
                elemTag = self.elementTagsByType[type][tag_idx]
                if tag in self.elements[elemTag].nodeTags:
                    condition = True
                    flag = False
                if tag_idx == tags_len[type_idx] - 1:
                    type_idx += 1
                    tag_idx = 0
                else:
                    tag_idx += 1
        return condition

    def numElements(self, type: int = -1) -> int:
        if type == -1:
            return len(self.elements)
        elif (type <= 0) or (type not in self.elementTypes):
            return 0
        else:
            return len(self.elementTagsByType[type])

    def numNodes(self) -> int:
        # Suma de todos los nodos de cada entidad
        # Restar todas las intersecciones unificadas
        # Modificar lo programado
        nodeTags = []
        for tag in self.elementTags:
            nodeTags += self.elements[tag].nodeTags
        return len(list(dict.fromkeys(nodeTags)))
