import pytest
from core.filters.lexical import LexicalFilter
from core.filters.bayesian import BayesianFilter


class TestFunnelFilters:
    """Testes unitários para os filtros O(1) do funil de ingestão."""

    def test_lexical_filter_exact_and_near_duplicates(self):
        f = LexicalFilter(tolerance=3)
        h1 = f.compute_hash("Notícia urgente sobre lançamento de foguete espacial.")
        h2 = f.compute_hash("Notícia urgente sobre lançamento de foguete espacial!")
        h3 = f.compute_hash("Uma receita culinária completamente diferente de pudim de leite.")

        hashes = [h1]
        assert f.is_duplicate(h2, hashes) is True  # Quase idêntico (distância Hamming <= 3)
        assert f.is_duplicate(h3, hashes) is False  # Texto totalmente diferente

    def test_bayesian_filter_scoring(self, tmp_path):
        seed_path = tmp_path / "vocab_test.json"
        seed_path.write_text('{"positive": {"ai": 3.0, "python": 2.0}, "negative": {"fofoca": -3.0}}')

        bf = BayesianFilter(str(seed_path))

        pos_score = bf.score_article("Novo modelo de AI em Python lançado hoje")
        assert pos_score == 5.0
        assert bf.is_relevant("Novo modelo de AI em Python lançado hoje", threshold=1.0) is True

        neg_score = bf.score_article("Muita fofoca sobre celebridades")
        assert neg_score == -3.0
        assert bf.is_relevant("Muita fofoca sobre celebridades", threshold=1.0) is False

