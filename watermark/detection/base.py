from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import stats
from transformers import AutoTokenizer

from watermark.utils.seed import get_current_positions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DetectionResult:
    n_tokens: np.ndarray | None = None
    p_values: np.ndarray | None = None
    decisions: np.ndarray | None = None
    scores: np.ndarray | None = None
    valid_mask: torch.BoolTensor | None = None


@dataclass
class DecoderResult:
    decoded_messages: Union[torch.Tensor, List[torch.Tensor], None] = None
    n_tokens: np.ndarray | None = None
    p_values: np.ndarray | None = None
    decisions: np.ndarray | None = None
    positions: np.ndarray | None = None
    bit_level_scores: Union[np.ndarray, List[np.ndarray], None] = None
    bit_pvalues: Union[np.ndarray, List[np.ndarray], None] = None
    valid_mask: torch.BoolTensor | None = None


class BaseZerobitDetector(ABC):
    def __init__(
        self,
        vocab_size: int,
        context_width: int = 4,
        salt_key: int = 35317,
        seed: int = 0,
        seeding: str = "hash",
        aggregation: str = "sum",
        alpha: float = 0.01,
    ):
        self.vocab_size = vocab_size
        self.context_width = context_width
        self.salt_key = salt_key
        self.seed = seed
        self.seeding = seeding
        self.aggregation = aggregation
        self.alpha = alpha

    def prepare_inputs(
        self,
        tokens: torch.LongTensor,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        ngram_tokens = tokens.unfold(dimension=1, size=self.context_width, step=1)[
            :, :-1, :
        ]

        ngram_flat = ngram_tokens.reshape(-1, self.context_width)

        token_ids = tokens[:, self.context_width :].reshape(-1)

        return ngram_flat, token_ids

    def get_valid_mask(self, mask: torch.BoolTensor | None) -> torch.BoolTensor | None:
        if mask is None:
            return None
        return mask[:, self.context_width :]

    @abstractmethod
    def compute_scores(
        self,
        ngram_tokens: torch.LongTensor,  # (N, context_width)
        token_ids: torch.LongTensor,  # (N,)
    ) -> torch.FloatTensor:
        pass

    def transform(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores

    def aggregate(
        self,
        scores: torch.FloatTensor,  # (batch_size, T-context_width)
        mask: torch.BoolTensor,  # (batch_size, T-context_width)
    ) -> torch.FloatTensor:  # (batch_size,)
        masked_scores = scores * mask.float()
        if self.aggregation == "sum":
            return masked_scores.sum(dim=1)
        if self.aggregation == "mean":
            n = mask.sum(dim=1).clamp(min=1)
            return masked_scores.sum(dim=1) / n
        if self.aggregation == "max":
            scores_copy = scores.clone()
            scores_copy[~mask] = -float("inf")
            return scores_copy.max(dim=1).values
        raise ValueError(f"Unknown aggregation method: {self.aggregation}")

    @abstractmethod
    def compute_pvalue(
        self,
        aggregated: np.ndarray,
        n_tokens: np.ndarray,
    ) -> np.ndarray:
        pass

    def is_watermarked(self, p_values: np.ndarray) -> np.ndarray:
        return p_values < self.alpha

    def detect(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
    ) -> DetectionResult:
        encoded = tokenizer(
            texts, return_tensors="pt", add_special_tokens=False, padding=True
        )
        tokens = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device).bool()
        batch_size, seq_len = tokens.shape
        if seq_len <= self.context_width:
            return DetectionResult()

        ngram_flat, token_ids = self.prepare_inputs(tokens)

        raw_scores = self.compute_scores(ngram_flat, token_ids).view(batch_size, -1)
        valid_mask = self.get_valid_mask(mask)
        aggregated = (
            self.aggregate(self.transform(raw_scores), valid_mask).cpu().numpy()
        )
        n_tokens = valid_mask.sum(dim=1).cpu().numpy()
        p_values = self.compute_pvalue(aggregated, n_tokens)
        return DetectionResult(
            n_tokens=n_tokens,
            p_values=p_values,
            decisions=self.is_watermarked(p_values),
            scores=aggregated,
            valid_mask=valid_mask,
        )


class BaseMultibitDetector(BaseZerobitDetector, ABC):
    def __init__(
        self,
        base_detector: BaseZerobitDetector,
        bits: int,
        combine: str = "fisher",
        position_salt: int = 35323,
        message_only: bool = False,
    ):
        super().__init__(
            base_detector.vocab_size,
            base_detector.context_width,
            base_detector.salt_key,
            base_detector.seed,
            base_detector.seeding,
            base_detector.aggregation,
            base_detector.alpha,
        )
        self.base_detector = base_detector
        self.bits = bits
        self.combine = combine
        self.position_salt = position_salt
        self.message_only = message_only

    @abstractmethod
    def decode(
        self,
        bit_level_scores: np.ndarray,  # (batch_size, bits, base)
    ) -> np.ndarray:  # (batch_size, bits)
        pass

    def combine_pvalues(self, pvals: np.ndarray) -> np.ndarray:
        batch_size, k = pvals.shape
        pvals = np.clip(pvals, 1e-100, 1.0)
        if self.combine == "fisher":
            X = -2.0 * np.sum(np.log(pvals), axis=1)
            return stats.chi2.sf(X, df=2 * k)
        if self.combine == "sidak":
            return 1.0 - np.prod(1.0 - pvals, axis=1)
        if self.combine == "holm":
            combined = np.ones(batch_size, dtype=np.float64)
            for i in range(batch_size):
                p_sorted = np.sort(pvals[i, :])
                combined[i] = min(1.0, np.min((k - np.arange(k)) * p_sorted))
            return combined
        if self.combine == "bonferroni":
            return np.minimum(1.0, k * np.min(pvals, axis=1))
        raise ValueError(f"Unknown combine method: {self.combine}")

    def aggregate_bit_level_scores(
        self,
        positions: torch.LongTensor,  # (batch_size, seq_len - context_width)
        scores: torch.FloatTensor,  # (batch_size, seq_len - context_width, base)
        valid_mask: torch.BoolTensor,  # (batch_size, seq_len - context_width)
    ) -> np.ndarray:
        batch_size, _, base = scores.shape
        expanded_index = positions.unsqueeze(-1).expand(-1, -1, base)
        masked_scores = scores * valid_mask.unsqueeze(-1).float()

        bit_level_scores = torch.zeros(
            (batch_size, self.bits, base), device=scores.device, dtype=torch.float
        )
        if self.aggregation == "sum":
            bit_level_scores.scatter_add_(
                dim=1, index=expanded_index, src=masked_scores
            )
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        return bit_level_scores.cpu().numpy()

    def compute_bit_pvalues(
        self,
        bit_level_scores: np.ndarray,  # (batch_size, bits, base)
        bit_n_tokens: np.ndarray,  # (batch_size, bits)
    ) -> np.ndarray:  # (batch_size, bits)
        return self.base_detector.compute_bit_pvalues(bit_level_scores, bit_n_tokens)

    def _process_scores(
        self,
        raw_scores: torch.FloatTensor,
        positions: torch.LongTensor,
        valid_mask: torch.BoolTensor,
        bit_n_tokens: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        transformed = self.transform(raw_scores)
        bit_level_scores = self.aggregate_bit_level_scores(
            positions,
            transformed,
            valid_mask,
        )
        decoded_messages = self.decode(bit_level_scores)
        if self.message_only:
            return decoded_messages, None, bit_level_scores
        bit_pvalues = self.compute_bit_pvalues(bit_level_scores, bit_n_tokens)
        return decoded_messages, bit_pvalues, bit_level_scores

    def detect(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        n_tr: Optional[int] = None,
    ) -> DecoderResult:
        encoded = tokenizer(
            texts, return_tensors="pt", add_special_tokens=False, padding=True
        )
        tokens = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device).bool()
        if n_tr is not None:
            tokens = tokens[:, : n_tr + self.context_width]
            mask = mask[:, : n_tr + self.context_width]
        batch_size, seq_len = tokens.shape
        if seq_len <= self.context_width:
            return DecoderResult()

        ngram_flat, token_ids = self.prepare_inputs(tokens)

        positions = get_current_positions(
            ngram_flat, self.position_salt, self.seed, self.seeding, self.bits
        ).view(
            batch_size, -1
        )  # (batch_size, T-context_width)

        valid_mask = self.get_valid_mask(mask)

        if self.message_only:
            n_tokens = None
            bit_n_tokens = None
        else:
            n_tokens = valid_mask.sum(dim=1).cpu().numpy()
            bit_n_tokens = torch.zeros(
                (batch_size, self.bits), device=device, dtype=torch.int64
            )
            bit_n_tokens.scatter_add_(dim=1, index=positions, src=valid_mask.long())
            bit_n_tokens = bit_n_tokens.cpu().numpy()

        scores = self.compute_scores(ngram_flat, token_ids)
        scores_list = [scores] if isinstance(scores, torch.Tensor) else scores

        decoded_messages_list = []
        bit_pvalues_list = []
        bit_level_scores_list = []

        for scores in scores_list:
            decoded_messages, bit_pvalues, bit_level_scores = self._process_scores(
                scores.view(batch_size, -1, scores.shape[-1]),
                positions,
                valid_mask,
                bit_n_tokens,
            )
            decoded_messages_list.append(decoded_messages)
            if not self.message_only:
                bit_pvalues_list.append(bit_pvalues)
                bit_level_scores_list.append(bit_level_scores)

        if self.message_only:
            return DecoderResult(decoded_messages=decoded_messages_list)

        bit_pvalues = np.concatenate(bit_pvalues_list, axis=1)
        document_pvalues = self.combine_pvalues(bit_pvalues)
        decisions = self.is_watermarked(document_pvalues)
        return DecoderResult(
            decoded_messages=decoded_messages_list,
            n_tokens=n_tokens,
            p_values=document_pvalues,
            decisions=decisions,
            positions=positions.cpu().numpy(),
            bit_level_scores=bit_level_scores_list,
            bit_pvalues=bit_pvalues_list,
            valid_mask=valid_mask,
        )
