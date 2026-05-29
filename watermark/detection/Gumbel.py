import numpy as np
import torch
from scipy.stats import gamma

from watermark.utils.seed import compute_ngram_seeds
from watermark_cuda import uniform_seeded_indexed

from .base import BaseZerobitDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
one_minus32 = torch.nextafter(
    torch.tensor(1.0, dtype=torch.float32, device=device),
    torch.tensor(0.0, dtype=torch.float32, device=device),
)


def h_ars(u: torch.Tensor) -> torch.Tensor:
    return -torch.log1p(-u.clamp(max=one_minus32))


class GumbelDetector(BaseZerobitDetector):
    def __init__(
        self,
        vocab_size: int,
        context_width: int = 4,
        salt_key: int = 35317,
        seed: int = 0,
        seeding: str = "hash",
        aggregation: str = "sum",
        alpha: float = 0.01,
        h_func: str = "ars",
    ):
        super().__init__(
            vocab_size=vocab_size,
            context_width=context_width,
            salt_key=salt_key,
            seed=seed,
            seeding=seeding,
            aggregation=aggregation,
            alpha=alpha,
        )
        self.h_func = h_func

    def compute_scores(
        self,
        ngram_tokens: torch.LongTensor,  # (N, context_width)
        token_ids: torch.LongTensor,  # (N,)
    ) -> torch.FloatTensor:
        # (n_positions,)
        salt_keys = torch.full(
            (ngram_tokens.shape[0],),
            self.salt_key,
            device=device,
            dtype=torch.long,
        )
        seeds = compute_ngram_seeds(ngram_tokens, salt_keys, self.seed, self.seeding)
        return uniform_seeded_indexed(seeds, token_ids, dtype=torch.float32)

    def transform(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.h_func == "ars":
            return h_ars(scores)
        if self.h_func == "id":
            return scores
        raise ValueError(f"Unknown h_func: {self.h_func}")

    def compute_pvalue(
        self,
        aggregated: np.ndarray,
        n_tokens: np.ndarray,
    ) -> np.ndarray:
        return self.compute_bit_pvalues(
            aggregated.view(-1, 1, 1),
            n_tokens.view(-1, 1),
        ).view(-1)

    def compute_bit_pvalues(
        self,
        bit_level_scores: np.ndarray,
        bit_n_tokens: np.ndarray,
    ) -> np.ndarray:
        _, _, base = bit_level_scores.shape
        k_obs = bit_level_scores.max(axis=-1)
        cdf = gamma.cdf(k_obs, bit_n_tokens, scale=1.0)
        return np.clip(1 - np.power(cdf, base), 0.0, 1.0)
