from abc import ABC, abstractmethod
import logging
from typing import Optional

import torch
from transformers import LogitsProcessor

logger = logging.getLogger(__name__)

from watermark.utils.seed import (
    compute_ngram_seeds,
    get_position_from_seeds,
    int_to_digits,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MAX_INT64 = torch.iinfo(torch.int64).max


class ContextMasking:
    def __init__(
        self,
        repeated_context_masking: bool = False,
        context_history_size: int = -1,
    ):
        self.repeated_context_masking = repeated_context_masking
        self.context_history_size = context_history_size
        self.context_history: Optional[torch.LongTensor] = None
        self._initialized_batch_size = None

    def _initialize_history(self, batch_size: int, device: torch.device):
        if self.context_history is None or self._initialized_batch_size != batch_size:
            if self.context_history_size == -1:
                self.context_history = torch.empty(
                    (batch_size, 0),
                    dtype=torch.int64,
                    device=device,
                )
            else:
                self.context_history = torch.zeros(
                    (batch_size, self.context_history_size),
                    dtype=torch.int64,
                    device=device,
                )
            self._initialized_batch_size = batch_size

    def check_and_update_repeated_context(
        self,
        seeds: torch.LongTensor,
    ) -> torch.BoolTensor:
        batch_size = seeds.shape[0]
        device = seeds.device
        if not self.repeated_context_masking:
            # (batch_size,)
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        self._initialize_history(batch_size, device)
        if self.context_history.shape[1] > 0:
            # (batch_size,)
            is_repeated_context = (self.context_history == seeds.unsqueeze(-1)).any(
                dim=1
            )
        else:
            is_repeated_context = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=device,
            )

        current_seeds = seeds.unsqueeze(-1)  # (batch_size, 1)
        if self.context_history_size == -1:
            self.context_history = torch.cat(
                (self.context_history, current_seeds), dim=1
            )
        else:
            self.context_history = torch.cat(
                (self.context_history[:, 1:], current_seeds),
                dim=1,
            )

        return is_repeated_context

    def apply_repeated_context_masking(
        self,
        scores_original: torch.FloatTensor,
        scores_watermarked: torch.FloatTensor,
        seeds: torch.LongTensor,
    ) -> torch.FloatTensor:
        if not self.repeated_context_masking:
            return scores_watermarked
        # (batch_size, 1)
        mask = self.check_and_update_repeated_context(seeds).unsqueeze(1)
        return torch.where(mask, scores_original, scores_watermarked)


class BaseWatermarkLogitsProcessor(LogitsProcessor, ABC):
    def __init__(
        self,
        vocab_size: int,
        context_width: int = 4,
        seed: int = 0,
        salt_key: int = 35317,
        seeding: str = "hash",
        repeated_context_masking: bool = True,
        context_history_size: int = -1,
        min_new_tokens: int | None = None,
    ):
        self.vocab_size = vocab_size
        self.context_width = context_width
        self.seed = seed
        self.salt_key = salt_key
        self.seeding = seeding
        self.device = device
        self.min_new_tokens = (
            min_new_tokens if min_new_tokens is not None else context_width
        )
        self.prompt_len = None
        self.context_masking = ContextMasking(
            repeated_context_masking=repeated_context_masking,
            context_history_size=context_history_size,
        )

    def set_prompt_len(self, prompt_len: int):
        self.prompt_len = prompt_len

    def _should_apply_watermark(self, seq_len: int) -> bool:
        if self.prompt_len is None:
            logger.warning("Prompt length not set. Watermark skipped.")
            return False
        new_tokens = seq_len - self.prompt_len + 1
        if new_tokens <= self.min_new_tokens:
            logger.info(f"Skipping watermark at new token {new_tokens}.")
            return False
        return True

    @abstractmethod
    def __call__(
        self,
        input_ids: torch.LongTensor,  # (batch_size, seq_len)
        scores: torch.FloatTensor,  # (batch_size, vocab_size)
    ) -> torch.FloatTensor:
        return scores


class BaseMultibitLogitsProcessor(BaseWatermarkLogitsProcessor, ABC):
    def __init__(
        self,
        base_processor: BaseWatermarkLogitsProcessor,
        bits: int,
        position_salt: int = 35323,
    ):
        self.base_processor = base_processor
        self.bits = bits
        self.position_salt = position_salt
        self.message_digits = None
        super().__init__(
            vocab_size=base_processor.vocab_size,
            context_width=base_processor.context_width,
            seed=base_processor.seed,
            salt_key=base_processor.salt_key,
            seeding=base_processor.seeding,
        )

    def set_prompt_len(self, prompt_len: int):
        self.prompt_len = prompt_len
        if self.base_processor:
            self.base_processor.set_prompt_len(prompt_len)

    def set_message(self, message: int) -> torch.LongTensor:
        self.message_digits = torch.tensor(
            int_to_digits(message, self.base, self.bits),
            dtype=torch.int64,
            device=device,
        )
