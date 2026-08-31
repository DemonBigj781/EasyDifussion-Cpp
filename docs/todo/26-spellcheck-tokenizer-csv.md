# 26 — Spellcheck / Tokenizer Suggestions from `merged_2024-12-22_pt2-ia-dd-ed.csv`

## Objective
Parse the merged CSV dataset into a fast local suggestion source for prompt spellcheck and tokenizer-aware corrections.

## Implementation
1. Document the CSV columns, encoding, delimiters, duplicate semantics, and provenance before coding against it.
2. Build an offline preprocessing step that parses rows, normalizes Unicode/case where appropriate, removes invalid entries, and generates a compact indexed artifact.
3. Separate ordinary spell correction from tokenizer suggestions; a token that looks unusual to a dictionary may be valid model syntax.
4. Store frequency/score/source fields when available so ranking is data-driven.
5. Implement lookup using a prefix index/trie, BK-tree/edit-distance index, or compact hash structures depending on dataset size.
6. Add model/tokenizer awareness: suggestions can prefer vocabulary tokens or preserve known prompt syntax, embeddings, LoRA tags, weights, and artist/model-specific terms.
7. Keep processing local and asynchronous in the UI.
8. Add user option to disable suggestions and never auto-rewrite prompts without confirmation.

## Validation
Malformed rows, Unicode, punctuation, known prompt tags, common typos, model vocabulary matches, performance on full dataset, and deterministic preprocessing.

## Complete when
The full CSV is reproducibly converted into a fast local index and the prompt UI can produce useful non-destructive spell/token suggestions.
