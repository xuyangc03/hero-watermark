import logging

import torch
import torch.nn.functional as F

from watermark.utils.seed import compute_ngram_seeds

try:
    from watermark_cuda import uniform_seeded

    CUSTOM_UNIFORM = True
except ImportError:
    CUSTOM_UNIFORM = False

from .base import BaseWatermarkLogitsProcessor

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
one_minus32 = torch.nextafter(
    torch.tensor(1.0, dtype=torch.float32, device=device),
    torch.tensor(0.0, dtype=torch.float32, device=device),
)


class GumbelLogitsProcessor(BaseWatermarkLogitsProcessor):
    def _sample(
        self,
        probs: torch.FloatTensor,  # (batch_size, vocab_size)
        seeds: torch.LongTensor,  # (batch_size,)
    ) -> torch.LongTensor:
        batch_size, vocab_size = probs.shape
        dtype = torch.float32

        if CUSTOM_UNIFORM and seeds.is_cuda:
            uniform = uniform_seeded(seeds, vocab_size, dtype)
        else:
            uniform = torch.empty_like(probs)
            for i in range(batch_size):
                generator = torch.Generator(device=device)
                generator.manual_seed(seeds[i].item())
                uniform[i] = torch.rand(vocab_size, generator=generator, device=device)

        uniform = uniform.clamp(max=one_minus32)
        gumbel = -torch.log(-torch.log(uniform))
        log_probs = torch.where(
            probs > 0,
            torch.log(probs.to(dtype)),
            -float("inf"),
        )
        return torch.argmax(log_probs + gumbel, dim=-1)

    def _apply_gumbel_transform(
        self,
        scores: torch.FloatTensor,  # (batch_size, vocab_size)
        seeds: torch.LongTensor,  # (batch_size,)
    ) -> torch.FloatTensor:
        probs = F.softmax(scores, dim=-1)
        sampled_indices = self._sample(probs, seeds)

        output_scores = torch.full_like(scores, float("-inf"))
        output_scores.scatter_(1, sampled_indices.unsqueeze(1), 0.0)
        return output_scores

    def __call__(
        self,
        input_ids: torch.LongTensor,  # (batch_size, seq_len)
        scores: torch.FloatTensor,  # (batch_size, vocab_size)
    ) -> torch.FloatTensor:
        batch_size, seq_len = input_ids.shape

        if not self._should_apply_watermark(seq_len):
            return scores

        ngram_tokens = input_ids[:, -self.context_width :]

        seeds = compute_ngram_seeds(
            ngram_tokens, [self.salt_key] * batch_size, self.seed, self.seeding
        )
        output_scores = self._apply_gumbel_transform(scores, seeds)
        return self.context_masking.apply_repeated_context_masking(
            scores_original=scores,
            scores_watermarked=output_scores,
            seeds=seeds,
        )
