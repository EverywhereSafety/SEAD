"""Assistant-only SFT for saved agentic tool-defender teacher traces."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from ..agentic_response import final_action_json
from .validation import rationale_contradicts_decision
from .checkpointing import _git_commit, _git_dirty
from .objectives import balanced_class_weights

AGENTIC_SFT_SCHEMA_VERSION = "sead-agentic-sft-v1"


@dataclass(frozen=True)
class AgenticSFTConfig:
    data_dir: str
    base_model: str
    output_dir: str
    model_revision: str | None = None
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_length: int = 65536
    mixed_precision: str = "bf16"
    parameter_dtype: str = "bfloat16"
    seed: int = 42
    gradient_checkpointing: bool = True
    max_grad_norm: float | None = 1.0
    balance_labels: bool = True
    length_bucketing: bool = True
    fsdp: bool = False
    trust_remote_code: bool = False
    disable_thinking: bool = True
    report_to: str = "none"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    save_each_epoch: bool = True
    resume_from_checkpoint: str | None = None

    def validate(self) -> None:
        if self.report_to not in {"none", "wandb"}:
            raise ValueError("report_to must be none or wandb")
        if self.epochs < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("epochs and gradient accumulation must be positive")
        # V1 normalizes token loss inside each trace. Keeping one trace per
        # microbatch makes the subsequent class weight trace-level as well.
        if self.batch_size != 1:
            raise ValueError("agentic SFT v1 requires batch_size=1")
        if self.learning_rate <= 0 or self.max_length < 2:
            raise ValueError("learning_rate must be positive and max_length >= 2")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be no, fp16, or bf16")
        if self.parameter_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("parameter_dtype must be float32, bfloat16, or float16")
        if self.max_grad_norm is not None and (
            not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0
        ):
            raise ValueError("max_grad_norm must be finite and positive")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"blank line at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError(f"expected object at {path}:{line_number}")
            rows.append(dict(value))
    return rows


def _normalized_action_content(raw: str) -> tuple[str, bool]:
    """Return one canonical runtime-parseable action and whether junk was removed."""

    candidate = final_action_json(raw)
    if candidate is None:
        raise ValueError("assistant output contains no complete JSON object")
    value = json.loads(candidate)
    if not isinstance(value, Mapping) or not value.get("action"):
        raise ValueError("assistant action must be a JSON object with action")
    normalized = json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
    return normalized, raw.strip() != candidate.strip()


def _messages(record: Mapping[str, Any]) -> list[dict[str, str]]:
    details = record.get("defender_details")
    if not isinstance(details, Mapping):
        raise TypeError("missing defender_details")
    raw = details.get("agentic_messages")
    if not isinstance(raw, list) or not raw:
        raise ValueError("missing agentic_messages")
    messages: list[dict[str, str]] = []
    for message in raw:
        if not isinstance(message, Mapping):
            raise TypeError("agentic message must be an object")
        role = str(message.get("role") or "")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("agentic messages require system/user/assistant text")
        if role == "assistant":
            content, _ = _normalized_action_content(content)
        messages.append({"role": role, "content": content})
    return messages


def validate_agentic_records(
    records: Sequence[Mapping[str, Any]], *, split: str
) -> dict[str, Any]:
    """Validate saved protocol, labels, split, and exact trace uniqueness."""

    labels: Counter[str] = Counter()
    samples: Counter[str] = Counter()
    sample_labels: dict[str, str] = {}
    identities: set[tuple[str, str]] = set()
    exact_traces: set[tuple[str, str]] = set()
    tool_using = 0
    protocol_recovered_messages = 0
    assistant_turns: list[int] = []
    for row_index, record in enumerate(records):
        sample_id = str(record.get("sample_id") or "")
        call_id = str(record.get("call_id") or "")
        if not sample_id:
            raise ValueError("every training record requires sample_id")
        identity = (sample_id, call_id or f"row-{row_index}")
        if identity in identities:
            raise ValueError(f"duplicate rollout identity: {identity}")
        identities.add(identity)
        if record.get("split") != split:
            raise ValueError(f"split mismatch for {identity}")
        if "status" in record and record.get("status") != "success":
            raise ValueError(f"non-success rollout in {split}: {identity}")
        if (
            "candidate_executed" in record
            and record.get("candidate_executed") is not False
        ):
            raise ValueError(f"candidate execution was not disabled: {identity}")
        if "rejection_reasons" in record and record.get("rejection_reasons"):
            raise ValueError(f"rejected rollout in training data: {identity}")
        label = str(record.get("label") or "").lower()
        if label not in {"pass", "block"}:
            raise ValueError(f"invalid label: {identity}")
        if "decision" in record and record.get("decision") != label:
            raise ValueError(f"decision/label mismatch: {identity}")
        if sample_id in sample_labels and sample_labels[sample_id] != label:
            raise ValueError(f"conflicting labels for sample: {sample_id}")
        sample_labels[sample_id] = label

        messages = _messages(record)
        raw_messages = record["defender_details"]["agentic_messages"]
        protocol_recovered_messages += sum(
            _normalized_action_content(str(message.get("content") or ""))[1]
            for message in raw_messages
            if isinstance(message, Mapping) and message.get("role") == "assistant"
        )
        roles = [message["role"] for message in messages]
        expected = ["system", "user"] + [
            "assistant" if index % 2 == 0 else "user"
            for index in range(len(messages) - 2)
        ]
        if roles != expected or roles[-1] != "assistant":
            raise ValueError(f"invalid agentic message ordering: {identity}")
        parsed_assistant = []
        for message in messages:
            if message["role"] != "assistant":
                continue
            value = json.loads(message["content"])
            if not isinstance(value, Mapping) or not value.get("action"):
                raise ValueError(f"invalid assistant action JSON: {identity}")
            parsed_assistant.append(value)
        final = parsed_assistant[-1]
        if final.get("action") != "submit" or final.get("decision") != label:
            raise ValueError(f"final submit does not match label: {identity}")
        if rationale_contradicts_decision(str(final.get("rationale") or ""), label):
            raise ValueError(f"rationale_decision_contradiction: {identity}")
        if any(value.get("action") == "submit" for value in parsed_assistant[:-1]):
            raise ValueError(f"submit appears before final turn: {identity}")

        trace_key = (
            sample_id,
            json.dumps(messages, ensure_ascii=False, sort_keys=True),
        )
        if trace_key in exact_traces:
            raise ValueError(f"duplicate exact trace for sample: {sample_id}")
        exact_traces.add(trace_key)
        labels[label] += 1
        samples[sample_id] += 1
        assistant_turns.append(len(parsed_assistant))
        tool_using += len(parsed_assistant) > 1

    return {
        "split": split,
        "records": len(records),
        "labels": dict(labels),
        "tool_using_records": tool_using,
        "unique_samples": len(samples),
        "rollouts_per_sample": {
            "min": min(samples.values(), default=0),
            "max": max(samples.values(), default=0),
        },
        "protocol_recovered_messages": protocol_recovered_messages,
        "assistant_turns": {
            "min": min(assistant_turns, default=0),
            "max": max(assistant_turns, default=0),
        },
    }


def load_agentic_dataset(
    data_dir: Path | str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = Path(data_dir).expanduser().resolve()
    train = _read_jsonl(root / "train.jsonl")
    validation = _read_jsonl(root / "validation.jsonl")
    train_audit = validate_agentic_records(train, split="train")
    validation_audit = validate_agentic_records(validation, split="validation")
    if not train_audit["labels"].get("pass") or not train_audit["labels"].get("block"):
        raise ValueError("agentic SFT training data must contain PASS and BLOCK")
    train_samples = {str(record["sample_id"]) for record in train}
    validation_samples = {str(record["sample_id"]) for record in validation}
    overlap = train_samples & validation_samples
    if overlap:
        raise ValueError(
            f"train/validation candidate overlap: {len(overlap)} sample_ids"
        )
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        contract = json.loads(manifest_path.read_text()).get("context_contract")
        if contract is not None:
            _validate_context_contract(train + validation, contract)
    return (
        train,
        validation,
        {
            "schema_version": AGENTIC_SFT_SCHEMA_VERSION,
            "splits": {"train": train_audit, "validation": validation_audit},
            "source": {
                name: {
                    "path": str((root / name).resolve()),
                    "bytes": (root / name).stat().st_size,
                    "mtime_ns": (root / name).stat().st_mtime_ns,
                }
                for name in ("train.jsonl", "validation.jsonl")
            },
        },
    )


def _validate_context_contract(records, contract):
    """Fail closed if repaired data drifts from its full-history runtime contract."""
    from ..sage import SAGEDefender, AGENTIC_CLOSURE_PROMPT_VERSION
    from ..inputs import normalize_tool_defense_input

    expected_prompt = SAGEDefender._default_system_prompt()
    if (
        contract.get("version") != "agentic-full-history-v2"
        or contract.get("prompt_version") != AGENTIC_CLOSURE_PROMPT_VERSION
        or contract.get("system_prompt_sha256")
        != hashlib.sha256(expected_prompt.encode()).hexdigest()
        or contract.get("full_history") is not True
        or contract.get("max_tokens_per_step") != 1024
    ):
        raise ValueError("agentic dataset context contract differs from runtime")
    for record in records:
        messages = _messages(record)
        if messages[0]["content"] != expected_prompt:
            raise ValueError("training system prompt differs from defender runtime")
        case = json.loads(messages[1]["content"].split("DEFENSE CASE\n", 1)[1])
        history, candidate, view = (
            case["prior_history"],
            case["candidate_tool_action"],
            case["history_view"],
        )
        if (
            view["omitted_events"] != 0
            or view["total_events"] != len(history)
            or view["visible_events"] != len(history)
        ):
            raise ValueError("agentic SFT context must contain full history")
        if candidate["action_id"] != record.get("action_id"):
            raise ValueError("candidate ID differs from source action ID")
        normalize_tool_defense_input(history, candidate)
        known_ids = {str(e["id"]) for e in history if e.get("id") is not None} | {
            candidate["action_id"]
        }
        for message in messages:
            if message["role"] == "assistant":
                action = json.loads(message["content"])
                if any(
                    str(value) not in known_ids
                    for value in action.get("evidence_event_ids", [])
                ):
                    raise ValueError("unknown evidence ID in repaired agentic data")


def _as_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("expected one chat-template sequence")
        value = value[0]
    return [int(token_id) for token_id in value]


def _chat_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
    disable_thinking: bool,
) -> list[int]:
    return _as_token_ids(
        tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=not disable_thinking,
        )
    )


def encode_agentic_messages(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    max_length: int,
    disable_thinking: bool = True,
) -> dict[str, list[int]]:
    """Encode an exact transcript and supervise assistant content plus EOT only."""

    full = _chat_ids(
        tokenizer,
        messages,
        add_generation_prompt=False,
        disable_thinking=disable_thinking,
    )
    if len(full) > max_length:
        raise ValueError(
            f"agentic trace has {len(full)} tokens, exceeding max_length={max_length}"
        )
    labels = [-100] * len(full)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        header = _chat_ids(
            tokenizer,
            messages[:index],
            add_generation_prompt=True,
            disable_thinking=disable_thinking,
        )
        through = _chat_ids(
            tokenizer,
            messages[: index + 1],
            add_generation_prompt=False,
            disable_thinking=disable_thinking,
        )
        if full[: len(through)] != through or through[: len(header)] != header:
            raise ValueError("chat template is not prefix-stable for assistant masking")
        if len(through) <= len(header):
            raise ValueError("assistant turn produced no supervised tokens")
        labels[len(header) : len(through)] = through[len(header) :]
    supervised = sum(label != -100 for label in labels)
    if supervised < 1:
        raise ValueError("agentic trace has no supervised assistant tokens")
    return {
        "input_ids": full,
        "attention_mask": [1] * len(full),
        "labels": labels,
    }


def _label_weights(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    counts = Counter(str(record["label"]) for record in records)
    pass_weight, block_weight = balanced_class_weights(counts["pass"], counts["block"])
    return {"pass": pass_weight, "block": block_weight}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _resume_signature(config: AgenticSFTConfig, world_size: int) -> dict[str, Any]:
    # Output/logging settings and target epoch count may change when continuing.
    settings = asdict(config)
    for key in (
        "data_dir",
        "output_dir",
        "epochs",
        "report_to",
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
        "save_each_epoch",
        "resume_from_checkpoint",
    ):
        settings.pop(key)
    hashes = {}
    for name in ("train.jsonl", "validation.jsonl"):
        digest = hashlib.sha256()
        with (Path(config.data_dir) / name).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    return {"settings": settings, "data_sha256": hashes, "world_size": world_size}


def _read_resume_state(path: Path, signature: dict, target_epochs: int) -> dict:
    state = json.loads((path / "trainer_state.json").read_text())
    if state.get(
        "schema_version"
    ) != "sead-agentic-epoch-checkpoint-v1" or not state.get("complete"):
        raise ValueError("checkpoint is incomplete or has an unsupported schema")
    if not (path / "state").is_dir() or not (path / "model").is_dir():
        raise ValueError("checkpoint is missing model or recovery state")
    if state.get("signature") != signature:
        raise ValueError(
            "resume requires identical data, training settings, and process count"
        )
    if not 1 <= state["completed_epochs"] <= target_epochs:
        raise ValueError(
            "target epochs is below the saved epoch or checkpoint epoch is invalid"
        )
    return state


def _save_epoch_checkpoint(
    accelerator: Any,
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    *,
    completed_epochs: int,
    optimizer_steps: int,
    history: list,
    signature: dict,
) -> Path:
    destination = output_dir / "checkpoints" / f"epoch-{completed_epochs:04d}"
    staging = destination.with_name(destination.name + ".incomplete")
    if accelerator.is_main_process:
        if destination.exists():
            raise FileExistsError(destination)
        staging.mkdir(parents=True, exist_ok=False)
    accelerator.wait_for_everyone()
    # All ranks participate: Accelerate stores per-rank RNG, model, optimizer,
    # scaler (when present), and its accumulation step counter.
    accelerator.save_state(str(staging / "state"), safe_serialization=True)
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(
            staging / "model",
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(staging / "model")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _write_json(
            staging / "trainer_state.json",
            {
                "schema_version": "sead-agentic-epoch-checkpoint-v1",
                "complete": True,
                "completed_epochs": completed_epochs,
                "optimizer_steps": optimizer_steps,
                "history": history,
                "signature": signature,
            },
        )
        staging.rename(destination)
        print(
            json.dumps(
                {
                    "event": "agentic_sft_checkpoint",
                    "path": str(destination),
                    "completed_epochs": completed_epochs,
                }
            ),
            flush=True,
        )
    accelerator.wait_for_everyone()
    return destination


def _init_tracking(
    accelerator: Any, config: AgenticSFTConfig, run_config: dict
) -> None:
    if config.report_to != "wandb" or not accelerator.is_main_process:
        return
    options = {"dir": str(Path(config.output_dir).expanduser().resolve())}
    if config.wandb_entity:
        options["entity"] = config.wandb_entity
    if config.wandb_run_name:
        options["name"] = config.wandb_run_name
    accelerator.init_trackers(
        config.wandb_project or os.environ.get("WANDB_PROJECT") or "sead",
        config=run_config,
        init_kwargs={"wandb": options},
    )


def _track_progress(accelerator: Any, config: AgenticSFTConfig, progress: dict) -> None:
    if config.report_to != "wandb" or not accelerator.is_main_process:
        return
    # Sampled progress is rank 0 only; epoch losses below are reduced across ranks.
    metrics = {
        "train/epoch": progress["epoch"],
        "train/optimizer_steps": progress["optimizer_steps"],
        "train/rank0_trace_loss": progress["trace_loss"],
        "train/rank0_weighted_trace_loss": progress["weighted_trace_loss"],
        "train/rank0_sequence_tokens": progress["sequence_tokens"],
        "train/rank0_supervised_tokens": progress["supervised_tokens"],
        "train/elapsed_seconds": progress["elapsed_seconds"],
    }
    if progress["peak_memory_gib"] is not None:
        metrics["train/rank0_peak_memory_gib"] = progress["peak_memory_gib"]
    accelerator.log(metrics)


def _selected_token_loss(model: Any, batch: Mapping[str, Any]) -> tuple[Any, int]:
    """Compute CE only at supervised positions, avoiding full-sequence vocab logits."""

    import torch

    labels = batch["labels"]
    if labels.shape[0] != 1:
        raise ValueError("agentic SFT v1 requires one trace per microbatch")
    positions = torch.nonzero(labels[0, 1:] != -100, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("batch has no next-token supervision")
    targets = labels[0, positions + 1]
    base_model = model
    while hasattr(base_model, "module"):
        base_model = base_model.module
    if hasattr(base_model, "get_base_model"):
        base_model = base_model.get_base_model()
    if "logits_to_keep" not in inspect.signature(base_model.forward).parameters:
        raise RuntimeError(
            "model must support tensor logits_to_keep for memory-safe agentic SFT"
        )
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        logits_to_keep=positions,
    )
    logits = outputs.logits
    if logits.shape[:2] != (1, positions.numel()):
        raise RuntimeError(
            "model did not preserve tensor logits_to_keep positions: "
            f"got {tuple(logits.shape)} for {positions.numel()} positions"
        )
    losses = torch.nn.functional.cross_entropy(
        logits[0].float(), targets, reduction="none"
    )
    return losses.mean(), int(positions.numel())


def _token_audit(
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    max_length: int,
    disable_thinking: bool,
    lengths_out: list[int] | None = None,
) -> dict[str, Any]:
    lengths = []
    supervised = []
    for record in records:
        encoded = encode_agentic_messages(
            tokenizer,
            _messages(record),
            max_length=max_length,
            disable_thinking=disable_thinking,
        )
        lengths.append(len(encoded["input_ids"]))
        supervised.append(sum(value != -100 for value in encoded["labels"]))
    if lengths_out is not None:
        lengths_out.extend(lengths)
    ordered = sorted(lengths)

    def percentile(fraction: float) -> int:
        if not ordered:
            return 0
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]

    return {
        "records": len(records),
        "total_tokens": sum(lengths),
        "total_supervised_tokens": sum(supervised),
        "max_tokens": max(lengths, default=0),
        "p50_tokens": percentile(0.50),
        "p95_tokens": percentile(0.95),
        "p99_tokens": percentile(0.99),
        "max_supervised_tokens": max(supervised, default=0),
        "truncation": False,
    }


def _length_grouped_indices(
    lengths: Sequence[int], *, world_size: int, seed: int
) -> list[int]:
    """Shuffle groups of similar lengths assigned to synchronous ranks."""

    if world_size < 1:
        raise ValueError("world_size must be positive")
    rng = random.Random(seed)
    ordered = sorted(range(len(lengths)), key=lambda index: lengths[index])
    complete = len(ordered) - (len(ordered) % world_size)
    groups = [
        ordered[start : start + world_size] for start in range(0, complete, world_size)
    ]
    remainder = ordered[complete:]
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(group)
    # Keep the incomplete group last so it cannot straddle a synchronous group
    # boundary after the full groups are shuffled.
    rng.shuffle(remainder)
    return [index for group in groups for index in group] + remainder


def train_agentic_sft(config: AgenticSFTConfig) -> dict[str, Any]:
    """Train one generative agentic defender checkpoint."""

    config.validate()
    if config.report_to == "wandb":
        try:
            import wandb  # noqa: F401 -- fail before expensive model/data work
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging requires wandb; install sead[wandb]"
            ) from exc
    train_records, validation_records, data_audit = load_agentic_dataset(
        config.data_dir
    )
    try:
        import accelerate
        import torch
        import transformers
        from accelerate import Accelerator, FullyShardedDataParallelPlugin
        from accelerate.utils import (
            DistributedDataParallelKwargs,
            InitProcessGroupKwargs,
        )
        from torch.utils.data import DataLoader
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "training dependencies are missing; install sead[training]"
        ) from exc

    output_dir = Path(config.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    fsdp_plugin = (
        FullyShardedDataParallelPlugin(
            sharding_strategy="FULL_SHARD",
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=["Qwen3DecoderLayer"],
            state_dict_type="FULL_STATE_DICT",
            use_orig_params=True,
            limit_all_gathers=True,
            sync_module_states=True,
        )
        if config.fsdp
        else None
    )
    accelerator = Accelerator(
        log_with="wandb" if config.report_to == "wandb" else None,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        kwargs_handlers=[
            DistributedDataParallelKwargs(gradient_as_bucket_view=True),
            InitProcessGroupKwargs(timeout=timedelta(minutes=10)),
        ],
        fsdp_plugin=fsdp_plugin,
    )
    signature = _resume_signature(config, accelerator.num_processes)
    resume_path = (
        Path(config.resume_from_checkpoint).expanduser().resolve()
        if config.resume_from_checkpoint
        else None
    )
    resume_state = (
        _read_resume_state(resume_path, signature, config.epochs)
        if resume_path
        else None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_token_lengths: list[int] = []
    data_audit["tokenization"] = {
        "train": _token_audit(
            tokenizer,
            train_records,
            max_length=config.max_length,
            disable_thinking=config.disable_thinking,
            lengths_out=train_token_lengths,
        ),
        "validation": _token_audit(
            tokenizer,
            validation_records,
            max_length=config.max_length,
            disable_thinking=config.disable_thinking,
        ),
    }
    parameter_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config.parameter_dtype]
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
        # Keep this explicit: BF16 parameters fit long agentic
        # traces on 80GB GPUs, while FP32 remains available for reproducibility.
        torch_dtype=parameter_dtype,
        attn_implementation="sdpa",
    )
    resolved_model_revision = getattr(model.config, "_commit_hash", None)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False

    class Records(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(train_records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return train_records[index]

    class EpochSampler(torch.utils.data.Sampler):
        def __init__(self) -> None:
            self.epoch = 0

        def __len__(self) -> int:
            return len(train_records)

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

        def __iter__(self):
            if config.length_bucketing:
                indices = _length_grouped_indices(
                    train_token_lengths,
                    world_size=accelerator.num_processes,
                    seed=config.seed + self.epoch,
                )
            else:
                generator = torch.Generator().manual_seed(config.seed + self.epoch)
                indices = torch.randperm(
                    len(train_records), generator=generator
                ).tolist()
            return iter(indices)

    loader = DataLoader(
        Records(),
        batch_size=1,
        sampler=EpochSampler(),
        collate_fn=list,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        foreach=False,
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    weights = _label_weights(train_records)
    run_config = {
        **asdict(config),
        "objective": "assistant_only_trace_normalized_causal_lm",
        "label_weights": weights if config.balance_labels else {"pass": 1, "block": 1},
        "chat_template_kwargs": (
            {"enable_thinking": False} if config.disable_thinking else {}
        ),
        "dataset_audit": data_audit,
        "provenance": {
            "code_commit": _git_commit(),
            "code_dirty": _git_dirty(),
            "resolved_model_revision": resolved_model_revision,
            "versions": {
                "accelerate": accelerate.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "distributed_processes": accelerator.num_processes,
        },
    }
    if accelerator.is_main_process:
        _write_json(output_dir / "run_config.json", run_config)
        _write_json(output_dir / "data_audit.json", data_audit)
    _init_tracking(accelerator, config, run_config)

    history = []
    started = time.monotonic()
    optimizer_steps = 0
    start_epoch = 0
    if resume_state is not None:
        accelerator.load_state(str(resume_path / "state"))
        start_epoch = resume_state["completed_epochs"]
        optimizer_steps = resume_state["optimizer_steps"]
        history = list(resume_state["history"])
        if accelerator.is_main_process:
            print(
                json.dumps(
                    {
                        "event": "agentic_sft_resumed",
                        "path": str(resume_path),
                        "completed_epochs": start_epoch,
                        "optimizer_steps": optimizer_steps,
                    }
                ),
                flush=True,
            )
    model.train()
    for epoch in range(start_epoch, config.epochs):
        loader.set_epoch(epoch)
        epoch_loss = torch.zeros((), device=accelerator.device)
        epoch_weighted_loss = torch.zeros((), device=accelerator.device)
        batches = torch.zeros((), device=accelerator.device)
        epoch_started = time.monotonic()
        for batch_index, rows in enumerate(loader, 1):
            record = rows[0]
            encoded = encode_agentic_messages(
                tokenizer,
                _messages(record),
                max_length=config.max_length,
                disable_thinking=config.disable_thinking,
            )
            batch = {
                key: torch.tensor([value], dtype=torch.long, device=accelerator.device)
                for key, value in encoded.items()
            }
            with accelerator.accumulate(model):
                trace_loss, supervised_tokens = _selected_token_loss(model, batch)
                weight = weights[str(record["label"])] if config.balance_labels else 1.0
                loss = trace_loss * weight
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("non-finite agentic SFT loss")
                window_start = (
                    (batch_index - 1) // config.gradient_accumulation_steps
                ) * config.gradient_accumulation_steps
                window_size = min(
                    config.gradient_accumulation_steps,
                    len(loader) - window_start,
                )
                accelerator.backward(
                    loss * (config.gradient_accumulation_steps / window_size)
                )
                if accelerator.sync_gradients and config.max_grad_norm is not None:
                    norm = accelerator.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                    if not torch.isfinite(norm).item():
                        raise FloatingPointError("non-finite gradient norm")
                optimizer.step()
                if accelerator.sync_gradients:
                    optimizer_steps += 1
                optimizer.zero_grad()
            epoch_loss += trace_loss.detach()
            epoch_weighted_loss += loss.detach()
            batches += 1
            if batch_index == 1 or batch_index % 25 == 0:
                progress = {
                    "event": "agentic_sft_progress",
                    "rank": accelerator.process_index,
                    "epoch": epoch + 1,
                    "batch": batch_index,
                    "batches_per_epoch": len(loader),
                    "optimizer_steps": optimizer_steps,
                    "trace_loss": trace_loss.detach().item(),
                    "weighted_trace_loss": loss.detach().item(),
                    "label": record["label"],
                    "sequence_tokens": len(encoded["input_ids"]),
                    "supervised_tokens": supervised_tokens,
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30
                    if torch.cuda.is_available()
                    else None,
                }
                print(json.dumps(progress), flush=True)
                _track_progress(accelerator, config, progress)
                if accelerator.is_main_process:
                    with (output_dir / "progress.jsonl").open("a") as sink:
                        sink.write(json.dumps(progress) + "\n")
        gathered_loss = accelerator.gather(epoch_loss)
        gathered_weighted_loss = accelerator.gather(epoch_weighted_loss)
        gathered_batches = accelerator.gather(batches)
        epoch_record = {
            "epoch": epoch + 1,
            "seconds": time.monotonic() - epoch_started,
            "optimizer_steps": optimizer_steps,
            "mean_unweighted_trace_loss": (
                gathered_loss.sum() / gathered_batches.sum().clamp_min(1)
            ).item(),
            "mean_weighted_trace_loss": (
                gathered_weighted_loss.sum() / gathered_batches.sum().clamp_min(1)
            ).item(),
        }
        history.append(epoch_record)
        if accelerator.is_main_process:
            if config.report_to == "wandb":
                accelerator.log(
                    {"epoch/" + key: value for key, value in epoch_record.items()}
                    | {"train/optimizer_steps": optimizer_steps}
                )
            with (output_dir / "training_metrics.jsonl").open("a") as sink:
                sink.write(json.dumps(epoch_record) + "\n")
        if config.save_each_epoch:
            _save_epoch_checkpoint(
                accelerator,
                model,
                tokenizer,
                output_dir,
                completed_epochs=epoch + 1,
                optimizer_steps=optimizer_steps,
                history=history,
                signature=signature,
            )

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    checkpoint = output_dir / "model"
    if accelerator.is_main_process:
        unwrapped.save_pretrained(
            checkpoint,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(checkpoint)
        metrics = {
            "training": history,
            "validation": {
                "status": "requires_agentic_generation_evaluation",
                "records": len(validation_records),
            },
        }
        _write_json(output_dir / "metrics.json", metrics)
        _write_json(
            output_dir / "serving_config.json",
            {
                "type": "sage",
                "model": str(checkpoint),
                "max_steps": 6,
                "max_tokens_per_step": 1024,
                "history_event_limit": None,
                "initial_history_max_chars": None,
                "temperature": 0.0,
                "chat_template_kwargs": run_config["chat_template_kwargs"],
                "training_input": {
                    "max_length": config.max_length,
                    "truncation": False,
                    "assistant_only": True,
                },
            },
        )
    else:
        metrics = {}
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=65536)
    parser.add_argument(
        "--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument(
        "--parameter-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
        help="dtype for model parameters and optimizer state",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--balance-labels", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--length-bucketing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="group similar-length traces across synchronous distributed ranks",
    )
    parser.add_argument(
        "--fsdp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fully shard parameters, gradients, and optimizer state across ranks",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--disable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--report-to", choices=("none", "wandb"), default="none")
    parser.add_argument(
        "--wandb-project", help="W&B project; defaults to WANDB_PROJECT or sead"
    )
    parser.add_argument(
        "--wandb-entity", help="W&B team/user; otherwise use SDK environment/default"
    )
    parser.add_argument(
        "--wandb-run-name", help="W&B run name; otherwise use WANDB_NAME/SDK default"
    )
    parser.add_argument(
        "--save-each-epoch", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Completed epoch checkpoint; use a new output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AgenticSFTConfig(
        data_dir=str(args.data_dir.expanduser().resolve()),
        base_model=args.base_model,
        output_dir=str(args.output_dir.expanduser().resolve()),
        model_revision=args.model_revision,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        mixed_precision=args.mixed_precision,
        parameter_dtype=args.parameter_dtype,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        balance_labels=args.balance_labels,
        length_bucketing=args.length_bucketing,
        fsdp=args.fsdp,
        trust_remote_code=args.trust_remote_code,
        disable_thinking=args.disable_thinking,
        report_to=args.report_to,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        save_each_epoch=args.save_each_epoch,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    if args.validate_only:
        train, validation, audit = load_agentic_dataset(config.data_dir)
        print(
            json.dumps(
                {
                    **audit,
                    "loaded": {"train": len(train), "validation": len(validation)},
                },
                indent=2,
            )
        )
        return 0
    metrics = train_agentic_sft(config)
    print(json.dumps(metrics, indent=2))
    return 0


__all__ = [
    "AGENTIC_SFT_SCHEMA_VERSION",
    "AgenticSFTConfig",
    "encode_agentic_messages",
    "load_agentic_dataset",
    "main",
    "train_agentic_sft",
    "validate_agentic_records",
]


if __name__ == "__main__":
    raise SystemExit(main())
