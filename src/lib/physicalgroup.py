from typing import Union, Any, List

class PhysicalGroup:
    def __init__(self, dim: int, tag: int, name: str, entityTags: list):
        self.dim = dim
        self.tag = tag
        self.name = name
        self.entityTags = entityTags

    def __str__(self):
        s = f"Physical group{self.tag}: dim={self.dim} | name={self.name}\n"
        s += f"Entities: {self.entityTags}\n"
        s += "-" * len(max(s.split("\n"), key=len))
        return s

    def __repr__(self):
        return self.__str__()

    # Methods.
    def find(self, element: Any) -> int:
        """Return 1, 0 or -1 if the element is in the Physical group
        with equal condition, or it's in the Physical group but with
        similar condition, or it's not.
        """
        i, flag = 0, True
        while flag and (i < len(self.entities)):
            tag = self.entityTags[i]
            condition = self.entities[tag].find(element)
            flag = False if condition in [0, 1] else True
            i += 1
        return condition

    def getElement(self, type: int, tag: int) -> Union[Any, None]:
        i, flag = 0, True
        element = None
        numEntities = len(self.entityTags)
        while flag and (i < numEntities):
            entityTag = self.entityTags[i]
            element = self.entities[entityTag].getElement(type, tag)
            if element is not None:
                flag = False
            i += 1
        return element

    def getElements(self, type: int) -> List[Any]:
        elements = []
        for entityTag in self.entityTags:
            elements += self.entities[entityTag].getElements(type)
        return elements

    def getElementTags(self, type: int) -> List[int]:
        elementTags = []
        for entityTag in self.entityTags:
            elementTags += self.entities[entityTag].getElementTags(type)
        return elementTags

    def getElementTagsByNode(self, tag: int) -> List[int]:
        elemTags = []
        for entityTag in self.entityTags:
            elemTags += self.entities[entityTag].getElementTagsByNode(tag)
        return elemTags

    def getElementsByNode(self, tag: int) -> List[Any]:
        elements = []
        for entityTag in self.entityTags:
            elements += self.entities[entityTag].getElementsByNode(tag)
        return elements

    def getNodeTags(self) -> List[int]:
        nodeTags = []
        for entityTag in self.entityTags:
            nodeTags += self.entities[entityTag].getNodeTags()
        nodeTags = list(set(nodeTags))
        return nodeTags

    def isElement(self, type: int, tag: int) -> bool:
        i, flag, condition = 0, True, False
        numEntities = len(self.entityTags)
        while flag and (i < numEntities):
            entityTag = self.entityTags[i]
            condition = self.entities[entityTag].isElement(type, tag)
            if condition:
                flag = False
            i += 1
        return condition

    def isNode(self, tag: int) -> bool:
        i, flag, condition = 0, True, False
        numEntities = len(self.entityTags)
        while flag and (i < numEntities):
            entityTag = self.entityTags[i]
            condition = self.entities[entityTag].isNode(tag)
            if condition:
                flag = False
            i += 1
        return condition

    def numElements(self, type: int = -1) -> int:
        num = 0
        for entityTag in self.entityTags:
            num += self.entities[entityTag].numElements(type)
        return num

    def numNodes(self) -> int:
        nodeTags = []
        for entityTag in self.entityTags:
            nodeTags += self.entities[entityTag].getNodeTags()
        return len(list(dict.fromkeys(nodeTags)))

    # Decorators: @Getters
    @property
    def vol(self):
        return sum(entity.vol for entity in self.entities.values())
