from typing import List, Optional
import torch
import torch.nn.functional as F
from watermark.utils.seed import compute_ngram_seeds, get_current_positions
from .base import (
    BaseMultibitLogitsProcessor,
    BaseWatermarkLogitsProcessor,
    int_to_digits,
)

import logging

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VocabChunker:
    def __init__(self, K: int):
        self.K = K

    def __call__(
        self, current_vocab_size: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure input tensor is 1D for this logic.
        assert current_vocab_size.dim() == 1, "Input tensor must be 1D."
        assert torch.all(
            current_vocab_size >= self.K
        ), f"All vocab sizes must be >= K={self.K}."

        a = current_vocab_size // self.K
        b = current_vocab_size % self.K

        k_range = torch.arange(self.K, device=current_vocab_size.device)
        a = a.unsqueeze(1)
        b = b.unsqueeze(1)

        mask = k_range < b
        chunk_size = torch.where(mask, a + 1, a)

        padded_chunks = torch.cat([torch.zeros_like(a), chunk_size], dim=1)
        all_boundary = torch.cumsum(padded_chunks, dim=1)

        return all_boundary, chunk_size


def parallel_slice_and_pad_gather(
    probs: torch.Tensor,
    current_token: torch.Tensor,
    current_vocab_size: torch.Tensor,
) -> torch.Tensor:
    vocab_size = probs.shape[1]

    max_current_vs = current_vocab_size.max().item()
    column_indices = torch.arange(
        max_current_vs, device=probs.device, dtype=torch.long
    ).unsqueeze(0)

    gather_indices = current_token.unsqueeze(1) + column_indices
    gather_indices.clamp_(max=vocab_size - 1)

    gathered_chunk = torch.gather(probs, 1, gather_indices)

    mask = column_indices < current_vocab_size.unsqueeze(1)
    gathered_chunk.masked_fill_(~mask, 0.0)

    return gathered_chunk


class HeRoLogitsProcessor(BaseMultibitLogitsProcessor):
    def __init__(
        self,
        base_processor: BaseWatermarkLogitsProcessor,
        bits: int,
        n_layers: int,
        layer_base: List[int] = [],
        layer_salts: List[List[int]] = [],
        split_schedule: List[int] = [],
        position_salt: int = 35323,
    ):
        if len(layer_base) != n_layers:
            raise ValueError(
                f"layer_base length {len(layer_base)} != n_layers {n_layers}"
            )
        if len(layer_salts) != n_layers:
            raise ValueError(
                f"layer_salts length {len(layer_salts)} != n_layers {n_layers}"
            )
        if len(split_schedule) != n_layers - 1:
            raise ValueError(f"split_schedule length must be {n_layers - 1}")
        if not hasattr(base_processor, "_sample"):
            raise RuntimeError("Base processor must have _sample method")
        super().__init__(
            base_processor=base_processor,
            bits=bits,
            position_salt=position_salt,
        )
        self.n_layers = n_layers
        self.layer_base = layer_base
        self.layer_salts = [
            torch.tensor(layer_salts[i], dtype=torch.int64, device=device)
            for i in range(n_layers)
        ]
        self.split_schedule = split_schedule

        self.chunkers = [VocabChunker(K=k) for k in split_schedule]

    def set_messages(self, messages: List[int]) -> torch.LongTensor:
        self.message_digits = torch.tensor(
            [
                int_to_digits(message, self.layer_base[i], self.bits)
                for i, message in enumerate(messages)
            ],
            dtype=torch.int64,
            device=device,
        )

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.message_digits is None:
            raise RuntimeError("Must call set_messages() before using the processor")

        batch_size, seq_len = input_ids.shape

        if not self._should_apply_watermark(seq_len):
            return scores

        ngram_tokens = input_ids[:, -self.context_width :]
        positions = get_current_positions(
            ngram_tokens, self.position_salt, self.seed, self.seeding, self.bits
        )
        # all_salts shape: (n_layers * batch_size,)
        all_salts = torch.stack(
            [
                self.layer_salts[i][self.message_digits[i][positions]]
                for i in range(self.n_layers)
            ]
        ).view(-1)
        repeated_ngram_tokens = ngram_tokens.repeat(self.n_layers, 1)

        all_seeds = compute_ngram_seeds(
            repeated_ngram_tokens,
            all_salts,
            self.seed,
            self.seeding,
        )
        layer_seeds = all_seeds.view(self.n_layers, batch_size)

        current_token_offset = torch.zeros(batch_size, dtype=torch.long, device=device)
        current_vocab_size = torch.full(
            (batch_size,), self.vocab_size, dtype=torch.long, device=device
        )

        probs = F.softmax(scores.to(torch.float32), dim=-1)
        cdf = torch.cumsum(probs, dim=1)
        cdf = F.pad(cdf, (1, 0))

        for i, chunker in enumerate(self.chunkers):
            all_boundary, chunk_size = chunker(current_vocab_size)

            global_boundaries = current_token_offset.unsqueeze(1) + all_boundary
            cdf_chunk = torch.gather(cdf, 1, global_boundaries)  # (bsz, K+1)
            probs_chunk = cdf_chunk[:, 1:] - cdf_chunk[:, :-1]  # (bsz, K)
            probs_chunk.clamp_(min=0.0)

            sampled_chunk_idx = self.base_processor._sample(probs_chunk, layer_seeds[i])

            chosen_start_rel = torch.gather(
                all_boundary, 1, sampled_chunk_idx.unsqueeze(1)
            ).squeeze(1)
            chosen_size = torch.gather(
                chunk_size, 1, sampled_chunk_idx.unsqueeze(1)
            ).squeeze(1)

            current_token_offset += chosen_start_rel
            current_vocab_size = chosen_size

        probs_final = parallel_slice_and_pad_gather(
            probs, current_token_offset, current_vocab_size
        )
        sampled_rel_token = self.base_processor._sample(probs_final, layer_seeds[-1])

        final_token_ids = current_token_offset + sampled_rel_token

        output_scores = torch.full_like(scores, float("-inf"))
        output_scores.scatter_(1, final_token_ids.unsqueeze(1), 0.0)

        context_seeds = compute_ngram_seeds(
            ngram_tokens, [self.salt_key] * batch_size, self.seed, self.seeding
        )
        output_scores = self.context_masking.apply_repeated_context_masking(
            scores_original=scores,
            scores_watermarked=output_scores,
            seeds=context_seeds,
        )

        return output_scores
