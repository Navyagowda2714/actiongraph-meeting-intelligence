# Evaluation

All numbers on this page come from `make eval` (`actiongraph eval --provider offline`) run in
this repository. The raw output is committed in
[`evals/results/latest.json`](../evals/results/latest.json) and
[`latest.md`](../evals/results/latest.md).

> **Read this first.** These are numbers for the **offline rule engine**, not for an LLM. No LLM
> API was reachable where this was built, so there are no model numbers here. The same harness
> runs against a real model with `actiongraph eval --provider anthropic` (or `openai`/`ollama`).

## Dataset

Six meeting transcripts, all written for this project (no third-party text), with hand-made gold
annotations in `data/gold/`.

| Split | Meetings | Format | Gold action items | Explicit / implicit | Decisions | Risks | Gold completions |
|---|---|---|---:|---|---:|---:|---:|
| **dev**: "Atlas checkout redesign" weekly series | 3 | VTT, JSON, TXT | 23 | 14 / 9 | 5 | 8 | 14 |
| **held-out**: incident review, launch planning, support ops | 3 | SRT with diarisation labels, TXT, VTT | 20 | 15 / 5 | 5 | 6 | - |

Each gold action item records the supporting turn ids, the owner (or `null` if unassigned), the
deadline as an ISO date (resolved relative to the meeting date) and whether the commitment is
explicit or implicit. The dev series also records which earlier item each later meeting closes.

**How the splits were used.** The rule engine was developed against the dev series, so dev
numbers are training-set numbers and are optimistic by construction. The held-out meetings were
written in different domains and styles (an SRT with anonymous `SPEAKER_0x` labels plus a speaker
map, phrasings like "consider it done", "I'll get back to you on that", "that's mine"). I wrote
the first held-out meeting before writing the rules, and the other two after the rules were
done. The rules were **not changed** after scoring the held-out set. The same person wrote the
rules and the held-out data, though, so treat held-out numbers as indicative, not as a benchmark.

## Metrics

- **Matching.** A predicted item matches a gold item if it cites at least one of the gold item's
  turns. Ties (several items from the same turn) are broken by description similarity, and
  matching is one-to-one (greedy, best pair first).
- **Action items:** precision, recall and F1. Explicit and implicit are also scored separately:
  recall by gold kind, precision by predicted kind.
- **Owner accuracy / deadline accuracy:** computed over matched items. `None == None` counts as
  correct, so the score rewards *not* inventing deadlines. "Deadline accuracy (gold has a date)"
  covers only items whose gold annotation has a date.
- **Evidence-link validity:** the share of emitted evidence quotes found in the cited turn.
  *Raw* is measured before the hallucination guard, *final* after it.
- **Decisions / risks:** P / R / F1 with the same turn-overlap matching.
- **Review-flag usefulness:** the share of false positives (and of true positives) flagged
  `needs_review`. You want a high FP rate and a low TP rate.
- **Tracking:** precision / recall of "gold item X was completed in meeting N" events after
  ingesting the dev series in order.

## Results (offline rule engine)

| Metric | Dev (3 meetings, 23 gold items) | Held-out (3 meetings, 20 gold items) |
|---|---:|---:|
| Action items - precision | 100.0% | 87.0% |
| Action items - recall | 100.0% | 100.0% |
| Action items - F1 | 100.0% | 93.0% |
| Explicit items - F1 | 100.0% | 93.8% |
| Implicit items - F1 | 100.0% | 90.9% |
| Implicit items - recall | 100.0% | 100.0% |
| Owner accuracy (matched) | 100.0% | 100.0% |
| Deadline accuracy (matched) | 100.0% | 95.0% |
| Deadline accuracy (gold has a date) | 100.0% | 91.7% |
| Explicit/implicit label accuracy | 100.0% | 95.0% |
| Evidence-link validity (final) | 100.0% | 100.0% |
| Evidence-link validity (raw model output) | 100.0% | 100.0% |
| Decisions - F1 | 100.0% | 28.6% |
| Risks/blockers - F1 | 100.0% | 50.0% |
| False positives flagged for review | 0.0% | 66.7% |
| True positives flagged for review | 13.0% | 10.0% |

Cross-meeting completion detection (dev series): precision 100.0%, recall 100.0% (14 of 14 gold
completions, 0 false completions).

Held-out counts: action items TP 20 / FP 3 / FN 0. Decisions TP 1 / FP 1 / FN 4. Risks TP 2 /
FP 0 / FN 4.

Cost: the offline provider makes one "call" per chunk (6 calls for 6 meetings); token counts are
estimates (6,111 in / 1,864 out) and latency is about 16 ms in total.

## Error analysis (held-out)

Action items, all 4 errors:

1. **FP "Walk us through the timeline"** (incident review, t001-t002). "Omar, can you walk us
   through the timeline?" followed by "Sure." looks like an accepted request, but it is done on
   the spot. Telling in-meeting requests apart from follow-up work needs semantics the rules lack.
2. **FP "Add a freshness banner..."** (t013). "We should also add a freshness banner" becomes an
   unassigned item. The gold annotation treats it as the lead-in to a decision. It *was* flagged
   for review (no owner), which is the intended safety net.
3. **FP "Renew for one more month"** (support ops, t012). "We'll renew for one more month" is a
   decision phrased like a team commitment. It was flagged for review (team "we" ownership).
4. **Deadline miss** (support ops, t004). "report back Friday" has no preposition. Bare weekdays
   are deliberately ignored, because narrative sentences ("the mockups went out Wednesday") would
   otherwise become deadlines. That is a precision/recall trade-off.

Decisions and risks are the weakest areas: the trigger lexicons ("let's go with", "decision:",
"risk", "blocked on", ...) miss phrasings such as "going forward, any schema change needs a
heads-up" and "we're holding the 28th unless...". This is where an LLM should clearly beat the
baseline.

The review flags work as intended on held-out data: 2 of 3 false positives were flagged, against
2 of 20 true positives.

## Caveats

- **Small, self-authored data.** 43 gold action items in total. One or two items move a metric by
  several points.
- **Dev = training set** for the rule engine, and for the tracker thresholds (completion threshold
  0.30, owner bonus 0.10, verb bonus 0.15). There is no held-out *series*, so the tracking numbers
  are not evidence of generalisation.
- **Evidence validity of 100% is expected offline**: the rule engine quotes sentences verbatim.
  The metric matters for real models, where *raw* validity measures how often the model misquotes.
- **Lenient matching.** Matching uses turn overlap and does not judge description quality. Rule
  descriptions such as "Triage them" or "Run it against staging" keep unresolved pronouns; an LLM
  would write better ones.

## Reproduce

```bash
make eval                                  # offline rules, writes evals/results/latest.{json,md}
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=... actiongraph eval --provider anthropic --out-dir evals/results/anthropic
```
