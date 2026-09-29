from simhash import Simhash
from core.domain.article import Article
from typing import List

class LexicalFilter:
    """Filtro Lexical (SimHash) para barrar redundâncias mecânicas em tempo O(1)."""
    def __init__(self, tolerance: int = 3):
        self.tolerance = tolerance

    def compute_hash(self, text: str) -> int:
        return Simhash(text).value

    def is_duplicate(self, new_hash: int, existing_hashes: List[int]) -> bool:
        for h in existing_hashes:
            x = new_hash ^ h
            dist = bin(x).count('1')
            if dist <= self.tolerance:
                return True
        return False
