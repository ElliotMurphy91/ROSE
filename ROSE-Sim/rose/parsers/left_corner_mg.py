"""
Left-Corner Minimalist Grammar parser -- *stub version*.
Implements .parse_incremental(word) -> list[Action]
"""
from dataclasses import dataclass
from typing import List

@dataclass
class Action:
    name: str
    stack_size: int = 0

class LeftCornerMGParser:
    def __init__(self):
        self.stack: list[str] = []

    def parse_incremental(self, word: str) -> List[Action]:
        # ***Replace with real logic later***
        self.stack.append(word)
        return [Action("SHIFT", len(self.stack))]
