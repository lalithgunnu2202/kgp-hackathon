# Track A – Character Consistency Verification

## Problem
Given a character backstory and the original novel, determine whether the backstory is consistent with the novel.

## Key Idea
We decompose each backstory into atomic claims and verify each claim against the novel using evidence-based semantic retrieval.

## Pipeline
Backstory → Claim Extraction → Pathway-based Retrieval → Evidence Scoring → Final Decision

## Use of Pathway
Pathway is used as the semantic dataflow backbone to structure and retrieve relevant novel passages efficiently before reasoning.

## Decision Logic
Each claim is independently evaluated using similarity thresholds.
The final consistency label is obtained by aggregating claim-level decisions.

## Dataset Usage
Only the provided dataset is used.
Pretrained models are employed solely for semantic representation and do not introduce external knowledge.

## Explainability
For every prediction, the system provides:
- Retrieved evidence
- Similarity scores
- Claim-level support decisions
