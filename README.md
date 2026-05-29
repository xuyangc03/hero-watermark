# [ICML 2026] Official Code for [Selective Disclosure Watermarking for Large Language Models](https://openreview.net/pdf?id=Oi1AISOxzt)

## File Structure

```
hero-watermark/
├── src/
│   ├── generate.py              # generation entrypoint
│   └── detect.py                # detection entrypoint
├── configs/
│   ├── generation_config.json   # example generation config
│   └── detection_config.json    # example detection config
└── watermark/
    ├── generation/              # HeRo generation code
    │   ├── base.py
    │   ├── HeRo.py
    │   └── Gumbel.py
    ├── detection/               # HeRo detection code
    │   ├── base.py
    │   ├── HeRo.py
    │   └── Gumbel.py
    └── utils/
        ├── data.py
        ├── models.py
        ├── seed.py
        └── watermark_cuda/      # CUDA extension
            ├── setup.py
            ├── watermark_cuda.cpp
            └── watermark_cuda_kernel.cu
```

## Install

```bash
uv venv .venv --python 3.12
uv pip install torch transformers accelerate scipy tqdm sentencepiece
cd watermark/utils/watermark_cuda
uv pip install --no-build-isolation -e .
cd ../../..
```

## Example Configs

- `configs/generation_config.json`
- `configs/detection_config.json`

Edit `model_name`, `input_path`, and `output_path` before running.

### Models

Built-in aliases are defined in `watermark/utils/models.py`.

Current alias map:

```python
{
    "Llama2_7B": "meta-llama/Llama-2-7b-hf",
}
```

If `model_name` is not in this map, it is used directly as the Hugging Face model path.

## Run Generation

```bash
python src/generate.py --config_path configs/generation_config.json
```

## Run Detection

```bash
python src/detect.py --config_path configs/detection_config.json
```
