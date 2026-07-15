# Project Context

## Purpose
Arbiter is a Trustworthy AI Decision Support System for Evidence-Based Claim Verification. Its primary research contribution is failure-mode-aware selective prediction.

## Core Philosophy
Arbiter optimizes for **decision quality** rather than mere answer generation. Instead of relying on a single scalar confidence score, Arbiter decomposes verification uncertainty into three distinct diagnostic signals:
1. Retrieval insufficiency
2. Cross-evidence contradiction
3. Model epistemic uncertainty

## Pipeline Overview
The system operates a 5-stage deterministic pipeline:
1. **ASR**: Whisper-based transcription of spoken claims.
2. **Claim Extraction**: DistilBERT filtering conversational filler for verifiable assertions.
3. **Retrieval**: FAISS + BM25 hybrid search over a static FEVER Wikipedia corpus.
4. **Verification**: DeBERTa-v3 NLI entailment with MC-Dropout uncertainty quantification.
5. **Triage**: XGBoost routing policy directing the claim to ANSWER, RE-RETRIEVE, or ESCALATE.
