import json
from pathlib import Path
from typing import Dict

class BayesianFilter:
    """Filtro Bayesiano Naive Bayes para crivo de relevância com Cold Start."""
    def __init__(self, seed_file: str = "vocabulary_seed.json"):
        self.seed_file = Path(seed_file)
        self.vocabulary: Dict[str, float] = {}
        self.load_seed()

    def load_seed(self):
        if not self.seed_file.exists():
            default_seed = {
                "positive": {"rust": 2.0, "go": 2.0, "arquitetura": 2.0, "startups": 1.5, "tecnologia": 1.5, "ai": 2.0},
                "negative": {"fofoca": -2.0, "celebridade": -2.0, "horóscopo": -2.0, "astrologia": -2.0, "crime": -1.5}
            }
            with open(self.seed_file, 'w') as f:
                json.dump(default_seed, f, indent=4)

        with open(self.seed_file, 'r') as f:
            data = json.load(f)
            for word, weight in data.get("positive", {}).items():
                self.vocabulary[word.lower()] = weight
            for word, weight in data.get("negative", {}).items():
                self.vocabulary[word.lower()] = weight

    def score_article(self, text: str) -> float:
        words = text.lower().split()
        score = 0.0
        for word in words:
            clean_word = "".join(c for c in word if c.isalnum())
            if clean_word in self.vocabulary:
                score += self.vocabulary[clean_word]
        return score

    def is_relevant(self, text: str, threshold: float = 0.0) -> bool:
        return self.score_article(text) >= threshold
