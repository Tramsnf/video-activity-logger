from __future__ import annotations
from typing import Dict, Any, Set
import yaml, json

class Taxonomy:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self._index()

    def _index(self):
        self.states: Set[str] = set([s['name'] for s in self.data.get('states', [])])
        self.events: Set[str] = set([s['name'] for s in self.data.get('events', [])])
        self.markers: Set[str] = set([s['name'] for s in self.data.get('markers', [])])
        self.all: Set[str] = self.states | self.events | self.markers

    def is_valid(self, activity: str) -> bool:
        return activity in self.all

def load_taxonomy(path: str) -> Taxonomy:
    with open(path, "r") as f:
        return Taxonomy(yaml.safe_load(f))
