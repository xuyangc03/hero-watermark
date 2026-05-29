import random
from typing import List

import torch

MAX_INT64 = torch.iinfo(torch.int64).max

try:
    from watermark_cuda import uniform_seeded

    CUSTOM_UNIFORM = True
except ImportError:
    CUSTOM_UNIFORM = False


def int_to_digits(message: int, base: int, bits: int = -1) -> List[int]:
    if message < 0:
        raise ValueError("message must be non-negative")
    if message == 0:
        digits = [0]
    else:
        digits = []
        remaining = message
        while remaining:
            digits.append(remaining % base)
            remaining //= base
        digits.reverse()
    if bits > 0:
        if len(digits) < bits:
            digits = [0] * (bits - len(digits)) + digits
        elif len(digits) > bits:
            digits = digits[-bits:]
    return digits


def digits_to_int(digits: List[int], base: int) -> int:
    value = 0
    for digit in digits:
        value = value * base + int(digit)
    return value


def compute_ngram_seeds(
    ngram_tokens: torch.LongTensor,
    salt_keys: torch.LongTensor,
    base_seed: int = 0,
    seeding: str = "hash",
) -> torch.LongTensor:
    device = ngram_tokens.device
    context_width = ngram_tokens.shape[-1]

    assert salt_keys is not None, "salt keys must be provided when hashing ngram tokens"
    if isinstance(salt_keys, list):
        salt_keys_t = torch.tensor(salt_keys, dtype=torch.int64, device=device)
    else:
        salt_keys_t = salt_keys
    assert salt_keys_t.ndim == 1, f"salt_keys must be 1D, got {salt_keys_t.ndim}D"
    assert (
        salt_keys_t.shape[0] == ngram_tokens.shape[-2]
    ), f"salt_keys length {salt_keys_t.shape[0]} must match ngram_tokens.shape[-2] ({ngram_tokens.shape[-2]})"

    if seeding == "hash":
        # Initialize seeds
        seeds = torch.full(
            ngram_tokens.shape[:-1], base_seed, dtype=torch.int64, device=device
        )

        # Rolling hash
        for i in range(context_width):
            seeds = (seeds * salt_keys_t) % MAX_INT64
            seeds = (seeds + ngram_tokens[..., i]) % MAX_INT64
    else:
        raise ValueError(f"Unknown seeding method: {seeding}")

    return seeds


def get_position_from_seeds(
    seeds: torch.LongTensor,
    bits: int,
) -> torch.LongTensor:
    device = seeds.device
    batch_size = seeds.shape[0]

    positions = torch.empty(batch_size, dtype=torch.long, device=device)

    if CUSTOM_UNIFORM and seeds.is_cuda:
        u = uniform_seeded(seeds, 1, torch.float32).squeeze(1)  # (batch_size,)
        positions = (u * bits).long().clamp(0, bits - 1)
    else:
        for i in range(batch_size):
            generator = torch.Generator(device=device)
            generator.manual_seed(seeds[i].item())
            pos = torch.randint(0, bits, (1,), generator=generator, device=device)
            positions[i] = pos

    return positions


def get_current_positions(
    ngram_tokens: torch.LongTensor,  # (batch_size, context_width)
    position_salt: int,
    seed: int,
    seeding: str,
    bits: int,
) -> torch.LongTensor:
    batch_size = ngram_tokens.shape[0]
    seeds = compute_ngram_seeds(
        ngram_tokens, [position_salt] * batch_size, seed, seeding
    )
    return get_position_from_seeds(seeds, bits)
