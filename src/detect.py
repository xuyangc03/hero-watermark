import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watermark.detection import (
    BaseMultibitDetector,
    GumbelDetector,
    HeRoDetector,
)
from watermark.utils import load_data, load_tokenizer
from watermark.utils.data import count_existing_samples, load_config

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to detection_config.json"
    )
    return parser.parse_args()


def create_detector(config: dict, vocab_size: int):
    base_wm = config["base_watermark"]
    multibit_wm = config["multibit_watermark"]
    context_width = config["context_width"]
    salt_key = config["salt_key"]
    seeding = config["seeding"]
    seed = config["seed"]
    alpha = config["alpha"]
    aggregation = config.get("aggregation", "sum")

    if base_wm == "Gumbel":
        base_detector = GumbelDetector(
            vocab_size=vocab_size,
            context_width=context_width,
            salt_key=salt_key,
            seed=seed,
            seeding=seeding,
            alpha=alpha,
            aggregation=aggregation,
        )

    if multibit_wm == "HeRo":
        return HeRoDetector(
            base_detector=base_detector,
            bits=config["bits"],
            n_layers=len(config["layer_base"]),
            layer_base=config["layer_base"],
            split_schedule=config["split_schedule"],
            layer_salts=config["layer_salts"],
            combine=config["combine"],
            position_salt=config["position_salt"],
        )


def convert_format(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_format(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_format(item) for item in obj]
    return obj


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    args = get_args()
    exp_config = load_config(args.config_path)
    logger.info("Config loaded: %s", exp_config)

    # Load data
    input_path = exp_config["input_path"]
    input_key = exp_config.get("input_key", "result")
    texts = load_data(input_path, n_samples=None, key=input_key)
    logger.info("Loaded %s texts from %s", len(texts), input_path)

    tokenizer = load_tokenizer(exp_config["model_name"])
    detector = create_detector(exp_config, exp_config["vocab_size"])
    logger.info("Created detector: %s", type(detector).__name__)

    # Output setup
    output_path = exp_config["output_path"]
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Resume check
    start_point = count_existing_samples(output_path)
    if start_point > 0:
        logger.info("Resuming from sample %s", start_point)

    batch_size = exp_config["batch_size"]

    with open(output_path, "a", encoding="utf-8") as f:
        for ii in tqdm(range(start_point, len(texts), batch_size)):
            chunk_texts = texts[ii : ii + batch_size]

            # Run detection
            with torch.no_grad():
                start_time = time.time()
                results = detector.detect(
                    chunk_texts, tokenizer, n_tr=exp_config.get("n_tr", None)
                )
                duration = time.time() - start_time

            for j, text in enumerate(chunk_texts):
                output_results = {
                    "index": ii + j,
                    "text": text,
                    "n_tokens": int(results.n_tokens[j]),
                    "p_value": float(results.p_values[j]),
                    "decision": bool(results.decisions[j]),
                }
                if isinstance(detector, BaseMultibitDetector):
                    output_results.update(
                        {
                            "decoded_message": [
                                decoded_messages[j]
                                for decoded_messages in results.decoded_messages
                            ],
                            "positions": results.positions[j],
                            "bit_level_scores": [
                                bit_level_scores[j]
                                for bit_level_scores in results.bit_level_scores
                            ],
                            "bit_pvalues": [
                                bit_pvalues[j] for bit_pvalues in results.bit_pvalues
                            ],
                        }
                    )
                else:
                    output_results["scores"] = results.scores[j]
                f.write(
                    json.dumps(convert_format(output_results), ensure_ascii=False)
                    + "\n"
                )
                f.flush()

            logger.info(
                "Batch %s: Speed %.2f samples/s",
                ii // batch_size,
                len(chunk_texts) / duration,
            )

    # Log peak memory usage
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
        logger.info(f"Peak GPU memory usage: {peak_memory:.2f} GB")

    logger.info("Detection completed. Results saved to %s", output_path)


if __name__ == "__main__":
    main()
