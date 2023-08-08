from abc import ABC
from ..data_types import BuildStep

class AbstractGenerativeModel(ABC):
    def gen_brick(self, graph) -> BuildStep:
        raise NotImplementedError()