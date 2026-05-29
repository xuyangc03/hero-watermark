import torch
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

from .base import BaseMultibitDetector, BaseZerobitDetector
from watermark.utils.seed import compute_ngram_seeds
from watermark_cuda import uniform_seeded_indexed
from watermark.generation.HeRo import VocabChunker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HeRoDetector(BaseMultibitDetector):
    def __init__(
        self,
        base_detector: BaseZerobitDetector,
        bits: int,
        n_layers: int,
        layer_base: List[int],
        layer_salts: List[List[int]],
        split_schedule: List[int],
        combine: str = "fisher",
        position_salt: int = 35323,
        message_only: bool = False,
    ):
        if len(layer_base) != n_layers:
            raise ValueError(f"len(layer_base) must equal n_layers")
        if len(layer_salts) != n_layers:
            raise ValueError(f"len(layer_salts) must equal n_layers")
        if len(split_schedule) != n_layers - 1:
            raise ValueError(f"len(split_schedule) must be n_layers - 1")

        super().__init__(base_detector, bits, combine, position_salt, message_only)
        self.transform = self.base_detector.transform
        self.n_layers = n_layers
        self.layer_base = layer_base
        self.layer_salts = [
            torch.tensor(salts, dtype=torch.int64, device=device)
            for salts in layer_salts
        ]
        self.split_schedule = split_schedule
        self.chunkers = [VocabChunker(K=k) for k in split_schedule]

    def compute_pvalue(
        self, aggregated: np.ndarray, n_tokens: np.ndarray
    ) -> np.ndarray:
        return self.base_detector.compute_pvalue(aggregated, n_tokens)

    def decode(self, bit_level_scores: np.ndarray) -> np.ndarray:
        return bit_level_scores.argmax(axis=2)

    def compute_scores(
        self,
        ngram_tokens: torch.LongTensor,  # (batch_size, context_width)
        token_ids: torch.LongTensor,  # (batch_size, )
    ) -> List[torch.Tensor]:

        batch_size = token_ids.shape[0]
        device = token_ids.device
        current_token_offset = torch.zeros(batch_size, dtype=torch.long, device=device)
        current_vocab_size = torch.full(
            (batch_size,), self.vocab_size, dtype=torch.long, device=device
        )

        scores_list = []
        for i, chunker in enumerate(self.chunkers):
            all_boundary, chunk_size = chunker(current_vocab_size)

            rel_token_pos = token_ids - current_token_offset  # (B, )

            boundary_mask = rel_token_pos.unsqueeze(1) >= all_boundary
            sampled_chunk_idx = boundary_mask.sum(dim=1) - 1

            layer_salts = self.layer_salts[i]
            base = len(layer_salts)

            ngram_expanded = ngram_tokens.unsqueeze(1).expand(-1, base, -1)

            seeds = compute_ngram_seeds(
                ngram_expanded, layer_salts, self.seed, self.seeding
            )  # (N, base)
            seeds_flat = seeds.reshape(-1)

            sampled_chunk_idx_flat = (
                sampled_chunk_idx.unsqueeze(1).expand(-1, base).reshape(-1)
            )  # (N*base,)
            scores = uniform_seeded_indexed(
                seeds_flat, sampled_chunk_idx_flat, torch.float32
            )  # (N*base,)

            scores_list.append(scores.view(-1, base))

            chosen_start_rel = torch.gather(
                all_boundary, 1, sampled_chunk_idx.unsqueeze(1)
            ).squeeze(1)

            chosen_size = torch.gather(
                chunk_size, 1, sampled_chunk_idx.unsqueeze(1)
            ).squeeze(1)

            current_token_offset += chosen_start_rel
            current_vocab_size = chosen_size

        layer_salts = self.layer_salts[-1]
        base = len(layer_salts)

        ngram_expanded = ngram_tokens.unsqueeze(1).expand(-1, base, -1)
        seeds = compute_ngram_seeds(
            ngram_expanded,
            layer_salts,
            self.seed,
            self.seeding,
        )
        seeds_flat = seeds.reshape(-1)

        final_rel_pos = token_ids - current_token_offset  # (B, )
        final_rel_pos_flat = final_rel_pos.unsqueeze(1).expand(-1, base).reshape(-1)

        final_layer_scores = uniform_seeded_indexed(
            seeds_flat, final_rel_pos_flat, torch.float32
        )

        scores_list.append(final_layer_scores.view(-1, base))

        return scores_list
