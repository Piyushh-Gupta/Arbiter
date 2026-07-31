"""Stateless model execution and preprocessing layers for Natural Language Inference (NLI)."""

from typing import Any, cast

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.core.verification.base import BaseNLIModel
from src.core.verification.verification_models import (
    NLILabelSchema,
    NLIModelDefinition,
    PassageVerificationInput,
    PassageVerificationScore,
    VerificationVerdict,
)


class TokenizerAdapter:
    """Centralized adapter for text tokenization, padding, truncation, and preprocessing."""

    def __init__(self, tokenizer_id: str, max_length: int = 512) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)  # type: ignore[no-untyped-call]
        self.max_length = max_length

    def tokenize(self, pairs: list[tuple[str, str]]) -> dict[str, Any]:
        """Preprocesses pair sequences into tensor attention masks and input IDs."""
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return dict(encoded)


class InferenceEngine:
    """Manages the raw model execution, device placement, precision, and state."""

    def __init__(
        self, model_id: str, device: str = "cpu", precision: str = "fp32"
    ) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.precision = precision

        if precision == "fp16":
            self.model.half()

    def infer(self, inputs: dict[str, Any]) -> torch.Tensor:
        """Executes model forward pass under torch.no_grad."""
        device_inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**device_inputs)
            return cast(torch.Tensor, outputs.logits)


class OutputNormalizer:
    """Normalizes output logits using softmax and translates indices via the NLILabelSchema."""

    def __init__(self, label_schema: NLILabelSchema) -> None:
        self.label_schema = label_schema

    def normalize(self, logits: torch.Tensor) -> tuple[PassageVerificationScore, ...]:
        """Applies softmax normalization and maps outputs to PassageVerificationScore objects."""
        probs = torch.softmax(logits, dim=-1)
        probs_list = probs.cpu().tolist()

        scores: list[PassageVerificationScore] = []
        for p in probs_list:
            entailment = 0.0
            contradiction = 0.0
            neutral = 0.0

            for idx, prob in enumerate(p):
                label_str = self.label_schema.id_mapping.get(idx, "INSUFFICIENT")
                # Normalize label string to standard keys
                if label_str in ("SUPPORTED", "SUPPORTS", "ENTAILMENT"):
                    entailment += prob
                elif label_str in ("CONTRADICTED", "REFUTES", "CONTRADICTION"):
                    contradiction += prob
                else:
                    neutral += prob

            total = entailment + contradiction + neutral
            if total > 0.0:
                entailment /= total
                contradiction /= total
                neutral /= total
            else:
                entailment, contradiction, neutral = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0

            scores.append(
                PassageVerificationScore(
                    entailment_probability=entailment,
                    contradiction_probability=contradiction,
                    neutral_probability=neutral,
                )
            )
        return tuple(scores)


class TransformerNLIModel(BaseNLIModel):
    """Production-grade NLI model wrapper implementing stateless execution and deterministic batching."""

    def __init__(
        self,
        config: NLIModelDefinition,
        label_schema: NLILabelSchema,
    ) -> None:
        self.config = config
        self.label_schema = label_schema

        self.tokenizer_adapter = TokenizerAdapter(
            tokenizer_id=config.tokenizer_id,
            max_length=config.max_sequence_length,
        )
        self.inference_engine = InferenceEngine(
            model_id=config.model_id,
            device=str(config.execution_device.value).lower()
            if hasattr(config.execution_device, "value")
            else str(config.execution_device).lower(),
            precision=config.inference_precision,
        )
        self.output_normalizer = OutputNormalizer(label_schema=label_schema)

    @property
    def label_map(self) -> dict[int, Any]:
        """Expose label map for backward compatibility."""
        return {
            idx: self.label_schema.verdict_mapping.get(
                label_str, VerificationVerdict.INSUFFICIENT
            )
            for idx, label_str in self.label_schema.id_mapping.items()
        }

    def predict(
        self, batch: tuple[PassageVerificationInput, ...]
    ) -> tuple[PassageVerificationScore, ...]:
        """Scores a batch of claim-passage pairs deterministically in batches."""
        if not batch:
            return ()

        all_scores: list[PassageVerificationScore] = []
        batch_size = self.config.batch_size

        for i in range(0, len(batch), batch_size):
            sub_batch = batch[i : i + batch_size]
            pairs = [(item.passage.text, item.claim) for item in sub_batch]

            encoded = self.tokenizer_adapter.tokenize(pairs)
            logits = self.inference_engine.infer(encoded)
            scores = self.output_normalizer.normalize(logits)
            all_scores.extend(scores)

        return tuple(all_scores)
