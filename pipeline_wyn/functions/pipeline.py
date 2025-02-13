from functions.data_entity import DataEntity
from typing import Callable, List

class Pipeline:
    def __init__(self, steps: List[Callable]):
        self.steps = steps

    def run(self, d: DataEntity) -> DataEntity:
        """Run all processing steps sequentially"""
        for step in self.steps:
            d = step.apply(d)
        return d
    