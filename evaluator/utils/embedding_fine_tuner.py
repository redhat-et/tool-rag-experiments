import argparse
import importlib
import os
import random
from typing import Callable, Dict, Any, List
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses

from evaluator.components.data_provider import get_queries
from evaluator.eval_spec import EvaluationEnvSpec


# -------------------- utils --------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_callable(dotted: str) -> Callable:
    """
    Load a callable specified as 'package.module:function_name'.
    """
    if ":" not in dotted:
        raise ValueError("reader must be in the form 'package.module:function_name'")
    mod_name, fn_name = dotted.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise TypeError(f"{dotted} is not callable.")
    return fn

# -------------------- simple preprocessing & composing --------------------


def preprocess_text(s: str) -> str:
    # keep it minimal; mirror your index-time preprocessing here if you have one
    return " ".join(s.strip().lower().split())


def render_args_schema(schema: Dict[str, Any]) -> str:
    """
    Expecting a JSONSchema-like dict with keys 'properties' and optional 'required'.
    """
    if not schema:
        return ""
    props = schema.get("properties") or {}
    required = list(schema.get("required") or [])
    ordered = required + [p for p in props.keys() if p not in required]
    parts = []
    for name in ordered:
        spec = props.get(name) or {}
        t = spec.get("type")
        if isinstance(t, str):
            parts.append(f"{name}:{t}")
        else:
            parts.append(name)
    return " ".join(parts)


def truncate_head_tail(s: str, max_chars: int, sep: str = " ... ") -> str:
    if len(s) <= max_chars:
        return s
    half = (max_chars - len(sep)) // 2
    return s[:half] + sep + s[-half:]


def compose_tool_text(tool_spec: Dict[str, Any],
                      parts: List[str],
                      max_chars: int = 1000,
                      do_preprocess: bool = True) -> str:
    """
    parts is a list like ["name","description","args"]
    tool_spec has at least: name, description, parameter schema at tool_spec.get("parameters") or "args_schema"
    """
    segments = []
    for p in parts:
        if p == "name":
            v = tool_spec.get("name") or ""
            if v:
                segments.append(f"name: {v}")
        elif p == "description":
            v = tool_spec.get("description") or ""
            if v:
                segments.append(f"desc: {v}")
        elif p == "args":
            schema = tool_spec.get("parameters") or tool_spec.get("required_parameters") or tool_spec.get("args_schema") or {}
            v = render_args_schema(schema)
            if v:
                segments.append(f"args: {v}")
        # ignore unknown
    text = " | ".join(segments)
    if do_preprocess:
        text = preprocess_text(text)
    text = truncate_head_tail(text, max_chars=max_chars)
    return text

# -------------------- training --------------------


@dataclass
class TrainConfig:
    base_model: str
    output_dir: str
    parts: List[str]
    max_chars: int = 1000
    epochs: int = 1
    batch_size: int = 64
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_seq_len: int = 512
    seed: int = 42
    normalize: bool = True  # for inference later


def build_examples_from_reader(parts: List[str], max_chars: int) -> List[InputExample]:
    """
    For each (query, tool_spec), create one positive pair.
    """
    query_specs = get_queries(EvaluationEnvSpec(irrelevant_tools_ratio=0.0), fine_tuning_mode=True)
    examples: List[InputExample] = []
    count_pairs = 0
    for query_spec in query_specs:
        q = preprocess_text(query_spec.query)
        for spec in query_spec.golden_tools.values():
            tool_txt = compose_tool_text(spec, parts=parts, max_chars=max_chars)
            if tool_txt:
                examples.append(InputExample(texts=[q, tool_txt]))
                count_pairs += 1
    if count_pairs == 0:
        raise SystemExit("No training pairs constructed; check your reader output.")
    print(f"[info] built {count_pairs} positive pairs from reader")
    return examples


def main():
    ap = argparse.ArgumentParser("Tool RAG embedding fine-tuning")
    ap.add_argument("--base-model", required=True, help="e.g., intfloat/e5-base-v2")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--parts", default="name,description,args", help="Comma-separated fields to include")
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = TrainConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        parts=[p.strip() for p in args.parts.split(",") if p.strip()],
        max_chars=args.max_chars,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")
    print(f"[info] loading base model: {cfg.base_model}")
    model = SentenceTransformer(cfg.base_model, device=device)
    model.max_seq_length = cfg.max_seq_len

    # Build training pairs using your reader only
    train_examples = build_examples_from_reader(cfg.parts, cfg.max_chars)

    # Dataloader & loss
    class STExampleDataset(Dataset):
        def __init__(self, examples): self.examples = examples

        def __len__(self): return len(self.examples)

        def __getitem__(self, i): return self.examples[i]

    train_dataset = STExampleDataset(train_examples)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    # Warmup
    steps_per_epoch = len(train_loader)
    warmup_steps = int(steps_per_epoch * cfg.epochs * cfg.warmup_ratio)
    print(f"[info] steps/epoch={steps_per_epoch} warmup_steps={warmup_steps}")

    # Train
    model.fit(
        train_objectives=[(train_loader, loss_fn)],
        epochs=cfg.epochs,
        warmup_steps=warmup_steps,
        weight_decay=cfg.weight_decay,
        scheduler="linear",
        optimizer_params={"lr": cfg.lr},
        output_path=cfg.output_dir,
        show_progress_bar=True,
        save_best_model=False,
    )

    # Ensure saved
    if not os.listdir(cfg.output_dir):
        model.save(cfg.output_dir)

    print(f"[done] model saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
