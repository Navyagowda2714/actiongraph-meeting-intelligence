"""Evaluation against hand-written gold annotations.

Matching: a predicted item matches a gold item when it cites at least one of the gold item's
turns; ties (two items from the same turn) are broken by description similarity, and matching
is one-to-one (greedy, best pair first). Reported metrics:

* action items: precision / recall / F1 overall, and separately for explicit vs implicit
* owner accuracy and deadline accuracy over matched items (None == None counts as correct)
* evidence-link validity: share of emitted evidence quotes verified verbatim in the cited turn
  (final output) and before the hallucination guard (raw model output)
* decisions and risks: P / R / F1 with the same turn-overlap matching
* cross-meeting tracking: precision / recall of "item completed in meeting N" events
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from actiongraph.extract.pipeline import ExtractionPipeline, quote_supported
from actiongraph.ingest import load_transcript
from actiongraph.llm import LLMClient
from actiongraph.models import ActionItem, MeetingAnalysis, Transcript
from actiongraph.similarity import similarity
from actiongraph.tracking.store import Store
from actiongraph.tracking.tracker import Tracker


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def add(self, other: PRF) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1": round(self.f1, 3),
        }


@dataclass
class Ratio:
    num: int = 0
    den: int = 0

    @property
    def value(self) -> float:
        return self.num / self.den if self.den else 0.0

    def add(self, other: Ratio) -> None:
        self.num += other.num
        self.den += other.den

    def as_dict(self) -> dict[str, float | int]:
        return {"correct": self.num, "total": self.den, "value": round(self.value, 3)}


def match(
    pred_turns: Sequence[set[str]],
    pred_text: Sequence[str],
    gold: Sequence[dict[str, Any]],
    text_key: str = "description",
) -> list[tuple[int, int]]:
    cands = []
    for pi, pt in enumerate(pred_turns):
        for gi, g in enumerate(gold):
            ov = len(pt & set(g["turns"]))
            if ov:
                sim = similarity(pred_text[pi], g.get(text_key) or g.get("summary", ""))
                cands.append((sim + 0.2 * ov, pi, gi))
    cands.sort(reverse=True)
    used_p, used_g, pairs = set(), set(), []
    for _, pi, gi in cands:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        pairs.append((pi, gi))
    return pairs


@dataclass
class MeetingScores:
    meeting: str
    split: str
    actions: PRF = field(default_factory=PRF)
    explicit: PRF = field(default_factory=PRF)
    implicit: PRF = field(default_factory=PRF)
    owner: Ratio = field(default_factory=Ratio)
    deadline: Ratio = field(default_factory=Ratio)
    deadline_when_gold_has_one: Ratio = field(default_factory=Ratio)
    kind: Ratio = field(default_factory=Ratio)
    evidence_valid: Ratio = field(default_factory=Ratio)
    raw_evidence_valid: Ratio = field(default_factory=Ratio)
    decisions: PRF = field(default_factory=PRF)
    risks: PRF = field(default_factory=PRF)
    fp_flagged_for_review: Ratio = field(default_factory=Ratio)
    tp_flagged_for_review: Ratio = field(default_factory=Ratio)
    item_map: dict[str, str] = field(default_factory=dict)  # predicted id -> gold id
    errors: list[str] = field(default_factory=list)


def score_meeting(
    analysis: MeetingAnalysis, transcript: Transcript, gold: dict[str, Any]
) -> MeetingScores:
    sc = MeetingScores(meeting=gold["meeting"], split=gold.get("split", "dev"))
    preds: list[ActionItem] = analysis.action_items
    golds = gold["action_items"]
    pairs = match(
        [{e.turn_id for e in p.evidence} for p in preds], [p.description for p in preds], golds
    )
    matched_p = {pi for pi, _ in pairs}
    matched_g = {gi for _, gi in pairs}
    sc.actions = PRF(len(pairs), len(preds) - len(pairs), len(golds) - len(pairs))
    for kind, prf in (("explicit", sc.explicit), ("implicit", sc.implicit)):
        prf.tp = sum(1 for _, gi in pairs if golds[gi]["kind"] == kind)
        prf.fn = sum(1 for gi, g in enumerate(golds) if g["kind"] == kind and gi not in matched_g)
        prf.fp = sum(
            1 for pi, p in enumerate(preds) if p.kind.value == kind and pi not in matched_p
        )
    for pi, gi in pairs:
        p, g = preds[pi], golds[gi]
        sc.item_map[p.id] = g["id"]
        ok_owner = p.owner == g["owner"]
        sc.owner.num += ok_owner
        sc.owner.den += 1
        pd = p.deadline.isoformat() if p.deadline else None
        sc.deadline.num += pd == g["deadline"]
        sc.deadline.den += 1
        if g["deadline"]:
            sc.deadline_when_gold_has_one.num += pd == g["deadline"]
            sc.deadline_when_gold_has_one.den += 1
        sc.kind.num += p.kind.value == g["kind"]
        sc.kind.den += 1
        sc.tp_flagged_for_review.num += p.needs_review
        sc.tp_flagged_for_review.den += 1
        if not ok_owner:
            sc.errors.append(f"owner {g['id']}: predicted {p.owner!r}, gold {g['owner']!r}")
        if pd != g["deadline"]:
            sc.errors.append(f"deadline {g['id']}: predicted {pd}, gold {g['deadline']}")
    for pi, p in enumerate(preds):
        if pi not in matched_p:
            sc.fp_flagged_for_review.num += p.needs_review
            sc.fp_flagged_for_review.den += 1
            sc.errors.append(f"FP: {p.description!r} ({', '.join(e.turn_id for e in p.evidence)})")
    for gi, g in enumerate(golds):
        if gi not in matched_g:
            sc.errors.append(f"FN: {g['id']} {g['description']!r}")
    for rec in [*preds, *analysis.decisions, *analysis.risks]:
        for e in rec.evidence:
            turn = transcript.turn(e.turn_id)
            sc.evidence_valid.num += bool(turn and quote_supported(e.quote, turn.text))
            sc.evidence_valid.den += 1
    sc.raw_evidence_valid = Ratio(
        analysis.usage.raw_evidence_valid, analysis.usage.raw_evidence_total
    )

    for recs, gl, attr, key in (
        (analysis.decisions, gold["decisions"], "decisions", "summary"),
        (analysis.risks, gold["risks"], "risks", "description"),
    ):
        prs = match(
            [{e.turn_id for e in r.evidence} for r in recs],
            [getattr(r, key) for r in recs],
            gl,
            text_key="summary",
        )
        setattr(sc, attr, PRF(len(prs), len(recs) - len(prs), len(gl) - len(prs)))
    return sc


def _sample_for(samples_dir: Path, stem: str) -> Path:
    for p in sorted(samples_dir.glob(stem + ".*")):
        if not p.name.endswith(".speakers.json"):
            return p
    raise FileNotFoundError(stem)


def run_eval(samples_dir: Path, gold_dir: Path, llm: LLMClient | None = None) -> dict[str, Any]:
    pipeline = ExtractionPipeline(llm)
    golds = [json.loads(p.read_text()) for p in sorted(gold_dir.glob("*.json"))]
    store = Store(":memory:")
    tracker = Tracker(store)
    per_meeting: list[MeetingScores] = []
    completion = PRF()
    gold_to_store: dict[str, str] = {}
    usage_tokens = {"input": 0, "output": 0, "calls": 0, "latency_ms": 0.0}
    for gold in sorted(golds, key=lambda g: g["meeting_date"]):
        transcript = load_transcript(
            _sample_for(samples_dir, gold["meeting"]), series=gold.get("series", gold["split"])
        )
        analysis = pipeline.run(transcript)
        report = tracker.ingest(transcript, analysis)
        sc = score_meeting(analysis, transcript, gold)
        per_meeting.append(sc)
        usage_tokens["input"] += analysis.usage.input_tokens
        usage_tokens["output"] += analysis.usage.output_tokens
        usage_tokens["calls"] += analysis.usage.llm_calls
        usage_tokens["latency_ms"] += analysis.usage.latency_ms
        for pid, gid in sc.item_map.items():
            gold_to_store[gid] = pid
        predicted = {e.item_id for e in report.completed}
        expected = {
            gold_to_store.get(c["item"], f"missing:{c['item']}") for c in gold["completions"]
        }
        completion.add(
            PRF(len(predicted & expected), len(predicted - expected), len(expected - predicted))
        )

    def agg(split: str | None) -> dict[str, Any]:
        rows = [s for s in per_meeting if split is None or s.split == split]
        tot: dict[str, Any] = {}
        for name in ("actions", "explicit", "implicit", "decisions", "risks"):
            prf = PRF()
            for r in rows:
                prf.add(getattr(r, name))
            tot[name] = prf.as_dict()
        for name in (
            "owner",
            "deadline",
            "deadline_when_gold_has_one",
            "kind",
            "evidence_valid",
            "raw_evidence_valid",
            "fp_flagged_for_review",
            "tp_flagged_for_review",
        ):
            ratio = Ratio()
            for r in rows:
                ratio.add(getattr(r, name))
            tot[name] = ratio.as_dict()
        tot["meetings"] = len(rows)
        tot["gold_action_items"] = sum(r.actions.tp + r.actions.fn for r in rows)
        return tot

    return {
        "provider": getattr(pipeline.llm, "model", "unknown"),
        "dev": agg("dev"),
        "heldout": agg("heldout"),
        "all": agg(None),
        "tracking_completions": completion.as_dict(),
        "usage": usage_tokens,
        "errors": {s.meeting: s.errors for s in per_meeting},
    }


def results_markdown(res: dict[str, Any]) -> str:
    def pct(x: float) -> str:
        return f"{100 * x:.1f}%"

    rows = [
        ("Action items - precision", "actions", "precision"),
        ("Action items - recall", "actions", "recall"),
        ("Action items - F1", "actions", "f1"),
        ("Explicit items - F1", "explicit", "f1"),
        ("Implicit items - F1", "implicit", "f1"),
        ("Implicit items - recall", "implicit", "recall"),
        ("Owner accuracy (matched)", "owner", "value"),
        ("Deadline accuracy (matched)", "deadline", "value"),
        ("Deadline accuracy (gold has a date)", "deadline_when_gold_has_one", "value"),
        ("Explicit/implicit label accuracy", "kind", "value"),
        ("Evidence-link validity (final)", "evidence_valid", "value"),
        ("Evidence-link validity (raw model output)", "raw_evidence_valid", "value"),
        ("Decisions - F1", "decisions", "f1"),
        ("Risks/blockers - F1", "risks", "f1"),
        ("False positives flagged for review", "fp_flagged_for_review", "value"),
        ("True positives flagged for review", "tp_flagged_for_review", "value"),
    ]
    d, h = res["dev"], res["heldout"]
    out = [
        f"Provider: `{res['provider']}`",
        "",
        f"| Metric | Dev ({d['meetings']} meetings, {d['gold_action_items']} gold items) "
        f"| Held-out ({h['meetings']} meetings, {h['gold_action_items']} gold items) |",
        "|---|---:|---:|",
    ]
    for label, k, f in rows:
        out.append(f"| {label} | {pct(d[k][f])} | {pct(h[k][f])} |")
    t = res["tracking_completions"]
    out += [
        "",
        f"Cross-meeting completion detection (dev series): precision {pct(t['precision'])}, "
        f"recall {pct(t['recall'])} ({t['tp']} of {t['tp'] + t['fn']} gold completions, "
        f"{t['fp']} false completions).",
    ]
    return "\n".join(out) + "\n"
