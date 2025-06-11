"""
Top-Down (predictive) Minimalist Grammar parser – stub version
Implements .parse_incremental(word) -> list[Action]

Strategy: keep an `expect` list initialised with ['S'].
For each incoming word: open a non-terminal (NT), SHIFT
the word, and (for the toy version) close the phrase with
REDUCE when the expect list empties.
"""
from dataclasses import dataclass
from typing import List

@dataclass
class Action:
    name: str
    stack_size: int = 0

class TopDownMGParser:
    def __init__(self):
        self.expect: list[str] = ["S"]  # what we still need to see

    def parse_incremental(self, word: str) -> List[Action]:
        events: List[Action] = []

        # Open the predicted phrase
        events.append(Action("NT", len(self.expect)))

        # Consume the prediction and SHIFT the word
        self.expect.pop(0)
        events.append(Action("SHIFT", len(self.expect)))

        # When expectations exhausted, close constituent
        if not self.expect:
            events.append(Action("REDUCE", len(self.expect)))

        return events
