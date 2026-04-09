---
title: Consistency Fix Log — Neural Router (FGCS)
date: 2026-04-06
manuscript: Manuscripts/Neural Router (Elsevier FGCS)/
---

# Consistency Fix Log

## Summary

| Category | Found | Fixed | Remaining |
|---|---|---|---|
| FORWARD_REF | 0 | 0 | 0 |
| TERMINOLOGY | 5 | 4 | 1 (SBERT variants in tables, acceptable) |
| STALE | 1 | 1 | 0 |
| CROSSREF | 2 | 1 | 1 (label prefix inconsistency, cosmetic) |
| FLOW | 1 | 1 | 0 |
| COUNT | 2 | 1 | 1 (contribution count, requires author judgment) |
| NOTATION | 1 | 0 | 1 (D symbol informality, acceptable per scoping note) |
| SECTION_PAIR | 0 | 0 | 0 |
| **Total** | **12** | **8** | **4** |

## Issue Table

| # | File:Line(s) | Category | Description | Severity | Status |
|---|---|---|---|---|---|
| 1 | Experiment.tex:200 vs :64 | TERMINOLOGY | "Claude 3 Haiku" vs "Claude 3.5 Haiku" — model ID is claude-3-haiku, not 3.5 | CRITICAL | FIXED: standardized to "Claude 3 Haiku" everywhere |
| 2 | Discussion/Abstract/Conclusion | COUNT | F1 0.69/0.689 (old single-seed) vs 0.656 (5-seed mean); SBERT 0.394 vs 0.423 | CRITICAL | FIXED: all instances updated to 5-seed values (0.656 Haiku, 0.66 rounded, 0.423 SBERT) |
| 3 | Discussion.tex:5,7 | TERMINOLOGY | $N$ used for subscription count instead of $\|\mathcal{S}\|$ | MAJOR | FIXED: replaced with $\|\mathcal{S}\|$ |
| 4 | Discussion.tex:42 | TERMINOLOGY | "discrimination load" variant | MAJOR | FIXED: changed to "discrimination capacity" |
| 5 | Design.tex:318 | TERMINOLOGY | CoverAndMerge vs Cover-AndMerge hyphenation | MINOR | Not fixed (line-break artifact, cosmetic) |
| 6 | Experiment.tex:167 | CROSSREF | \|S\|_cross notation overload (context-window vs discrimination) | MAJOR | FIXED: disambiguated with explicit note distinguishing the two crossover concepts |
| 7 | Results.tex:123 | CROSSREF | Crossover figure caption references W_cross on |S| axis | MAJOR | FIXED: caption now describes subscription volume exhausting context window |
| 8 | Introduction.tex:14-17 | COUNT | Contribution 4 (QoE) overlaps with contribution 2 | MINOR | Not fixed (requires author judgment on contribution structure) |
| 9 | main.tex:90 | FLOW | "Below the context-window crossover" should be "Above" | CRITICAL | FIXED: "Above the context-window crossover, where the full subscription set fits" |
| 10 | Experiment.tex:93-94 | STALE | A0 described as "accuracy ceiling" unconditionally | MAJOR | FIXED: qualified with "when |S| is below the discrimination-capacity crossover" |
| 11 | Design.tex:545 | NOTATION | D(model) introduced but not formalized | MINOR | Acceptable: the scoping note explicitly states "Full characterisation...is the subject of ongoing work" |
| 12 | Results.tex:115 | CROSSREF | sec:crossover vs ssec:res-* label prefix inconsistency | MINOR | Not fixed (compiles correctly, cosmetic) |
| 13 | Results.tex/Discussion.tex | TERMINOLOGY | SBERT vs S-BERT vs Sentence-BERT | MINOR | Acceptable: full form at first use, abbreviation thereafter, table uses short form |
| 14 | Discussion.tex:5 | COUNT | SBERT baseline value 0.394 vs 0.423 | MAJOR | FIXED (as part of Issue 2) |
| 17 | main.tex/Introduction/Conclusion | TERMINOLOGY | "tradeoff" vs "trade-off" | MINOR | FIXED: standardized to "trade-off" (hyphenated) |

## Files Modified

- `main.tex`: abstract crossover direction, F1 value, model name, spelling
- `txt/Introduction.tex`: spelling
- `txt/Design.tex`: model name
- `txt/Experiment.tex`: model name, A0 qualification, crossover disambiguation
- `txt/Results.tex`: crossover caption, tau plateau note
- `txt/Discussion.tex`: F1 values, $N$ → $|\mathcal{S}|$, terminology, model name
- `txt/Conclusion.tex`: F1 value, model name, spelling
- `bibtex/new-ref.bib`: added liu2024lost
