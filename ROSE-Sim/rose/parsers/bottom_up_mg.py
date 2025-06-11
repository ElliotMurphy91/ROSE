"""
Bottom-Up Minimalist Grammar parser – stub version
Implements .parse_incremental(word) -> list[Action]

Strategy: SHIFT every word, then naively REDUCE whenever two
items sit on top of the stack.  Replace with proper feature-
checking and projection logic when ready.
"""
from dataclasses import dataclass
from typing import List

@dataclass
class Action:
    name: str
    stack_size: int = 0

class BottomUpMGParser:
    def __init__(self):
        self.stack: list[str] = []

    def parse_incremental(self, word: str) -> List[Action]:
        events: List[Action] = []

        # SHIFT: push current word
        self.stack.append(word)
        events.append(Action("SHIFT", len(self.stack)))

        # naïve REDUCE: whenever ≥2 items, merge top-2 -> ‘X’
        if len(self.stack) >= 2:
            self.stack[-2:] = ["X"]
            events.append(Action("REDUCE", len(self.stack)))

        return events
