# Model Training Workflow

This directory contains scripts for preparing training data and training an E5-based model for use in the tool2vec pipeline.

## 1. Format Training Data

First, generate triplet data from your raw training JSON:

```sh
PYTHONPATH=../.. python format_train_data.py \
  --model intfloat/e5-base-v2 \
  --train_data_path ../data/train.json \
  --output_path ../data/train_triplets.pt
```
- `--model`: The HuggingFace model name to use for embeddings (default: `intfloat/e5-base-v2`).
- `--train_data_path`: Path to your raw training data (JSON format).
- `--output_path`: Where to save the generated triplet dataset (PyTorch .pt file).

## 2. Train the Model

Train the E5 model using the generated triplet data:

```sh
PYTHONPATH=../.. python train.py \
  --train_data_path ../data/train_triplets.pt \
  --model_name intfloat/e5-base-v2 \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-5 \
  --wd 0.01 \
  --margin 1 \
  --num_linear_warmup_steps 1600 \
  --checkpoint_dir checkpoints
```
- `--train_data_path`: Path to the triplet dataset generated above.
- `--model_name`: The HuggingFace model name to use for training.
- `--checkpoint_dir`: Directory to save model checkpoints (e.g., `checkpoints/`).

## 3. Using the Trained Model in tool2vec_tool_rag_algorithm.py

After training, you will have checkpoint files (e.g., `checkpoints/model_epoch_1.pt`).

To use a specific checkpoint in `tool2vec_tool_rag_algorithm.py`, set the environment variable before running:

```sh
export MODEL_CHECKPOINT_PATH=model/checkpoints/model_epoch_1.pt
PYTHONPATH=../.. python ../tool2vec_tool_rag_algorithm.py
```

This will load your trained E5 model weights for embedding and retrieval in the tool2vec pipeline.
