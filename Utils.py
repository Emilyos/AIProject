import enum
class FeatureType(enum.Enum):
    Discrete = 0
    Continuous = 1


class Feature:
    def __init__(self, index, type, domain=[], used=False):
        self.index = index
        self.type = type
        self.domain = domain
        self.used = False
