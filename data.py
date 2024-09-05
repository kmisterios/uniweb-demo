from dataclasses import dataclass
from typing import List


@dataclass
class Knowledge:
    name: str
    child: List[str]


@dataclass
class Competence:
    name: str
    child: List[Knowledge]


@dataclass
class MainCompetence:
    name: str
    child: List[Competence]
