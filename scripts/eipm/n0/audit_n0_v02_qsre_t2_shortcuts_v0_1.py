from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path

NONE_RELATION = 6
CONTROL_RELATIONAL = 1


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def target(row: dict) -> dict[str, str]:
    t = row["operator_target"]
    seq = [int(x) for x in t.get("relation_sequence_id", [])][:2]
    seq += [NONE_RELATION] * (2 - len(seq))
    control = int(t["control_id"])
    operation = int(t["operation_id"]) if control == CONTROL_RELATIONAL else -1
    role = int(t["role_id"])
    rel = ",".join(str(x) for x in seq)
    return {
        "relation": rel,
        "role": str(role),
        "operation": str(operation),
        "control": str(control),
        "signature": f"{rel}|{role}|{operation}|{control}",
    }


class MultinomialNB:
    def __init__(self) -> None:
        self.classes: list[str] = []
        self.class_docs: dict[str, int] = {}
        self.word_counts: dict[str, collections.Counter[str]] = {}
        self.totals: dict[str, int] = {}
        self.vocab: set[str] = set()
        self.total_docs = 0

    def fit(self, texts: list[str], labels: list[str]) -> None:
        self.total_docs = len(texts)
        class_docs = collections.Counter(labels)
        self.classes = sorted(class_docs)
        self.class_docs = dict(class_docs)
        self.word_counts = {c: collections.Counter() for c in self.classes}
        for text, label in zip(texts, labels):
            ts = tokens(text)
            self.word_counts[label].update(ts)
            self.vocab.update(ts)
        self.totals = {c: sum(self.word_counts[c].values()) for c in self.classes}

    def predict_one(self, text: str) -> str:
        ts = tokens(text)
        vocab_size = max(len(self.vocab), 1)
        best = None
        best_score = -float("inf")
        for c in self.classes:
            prior = math.log((self.class_docs[c] + 1) / (self.total_docs + len(self.classes)))
            denom = self.totals[c] + vocab_size
            score = prior
            counts = self.word_counts[c]
            for tok in ts:
                score += math.log((counts[tok] + 1) / denom)
            if score > best_score:
                best_score = score
                best = c
        assert best is not None
        return best

    def predict(self, texts: list[str]) -> list[str]:
        return [self.predict_one(t) for t in texts]


def accuracy(pred: list[str], gold: list[str]) -> float:
    if not gold:
        return 0.0
    return sum(int(a == b) for a, b in zip(pred, gold)) / len(gold)


def majority(train_labels: list[str], count: int) -> list[str]:
    value = collections.Counter(train_labels).most_common(1)[0][0]
    return [value] * count


def length_predict(
    train_text: list[str],
    train_labels: list[str],
    eval_text: list[str],
) -> list[str]:
    sums: dict[str, list[int]] = collections.defaultdict(list)
    for text, label in zip(train_text, train_labels):
        sums[label].append(len(tokens(text)))
    means = {label: sum(vals) / len(vals) for label, vals in sums.items()}
    return [
        min(means, key=lambda label: abs(len(tokens(text)) - means[label]))
        for text in eval_text
    ]


def family_predict(
    train_rows: list[dict],
    label_name: str,
    eval_rows: list[dict],
) -> list[str]:
    by_family: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    all_labels: list[str] = []
    for row in train_rows:
        label = target(row)[label_name]
        by_family[str(row["family"])][label] += 1
        all_labels.append(label)
    global_label = collections.Counter(all_labels).most_common(1)[0][0]
    mapping = {
        fam: counter.most_common(1)[0][0]
        for fam, counter in by_family.items()
    }
    return [mapping.get(str(row.get("family", "")), global_label) for row in eval_rows]


RELATION_CUES = {
    "0,6": ("support", "supports", "back", "backs", "evidence", "strengthen"),
    "1,6": ("correct", "correction", "erratum", "repair", "revise", "amend"),
    "2,6": ("replace", "replaced", "supersede", "precedence", "displace", "obsolete"),
    "3,6": ("derived", "basis", "source", "came from", "produced from", "origin"),
    "4,6": ("cause", "caused", "led to", "resulting", "consequence", "triggered"),
    "5,6": ("after", "before", "successor", "predecessor", "chronolog", "came later"),
}

OP_CUES = {
    "1": ("trace", "chain", "terminal", "end if", "after both", "two connections"),
    "2": ("several", "multiple", "three", "more than one", "which records"),
    "3": ("calibrated", "audited", "verified", "trust", "authoritative", "known fault"),
    "4": ("post-", "after the", "before the", "during spring", "fall audit", "rerun finished"),
}

CONTROL_CUES = {
    "2": ("how are", "what relation", "which relational operator", "what kind of relationship", "does not say whether", "without specifying"),
}


def cue_predict_relation(text: str) -> str:
    lower = text.lower()
    scores = {
        label: sum(int(cue in lower) for cue in cues)
        for label, cues in RELATION_CUES.items()
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "6,6"
    return best


def cue_predict_operation(text: str) -> str:
    lower = text.lower()
    scores = {
        label: sum(int(cue in lower) for cue in cues)
        for label, cues in OP_CUES.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "0"


def cue_predict_control(text: str) -> str:
    lower = text.lower()
    defer = sum(int(cue in lower) for cue in CONTROL_CUES["2"])
    if defer:
        return "2"
    # This heuristic intentionally cannot distinguish all ordinary fallback
    # questions from relational role-select questions.
    if not any(word in lower for word in ("record", "relation", "link", "cause", "support", "correct", "replace", "source", "successor", "predecessor")):
        return "0"
    return "1"


def evaluate_model(
    train_rows: list[dict],
    eval_rows: list[dict],
) -> dict:
    train_text = [str(r["query_text"]) for r in train_rows]
    eval_text = [str(r["query_text"]) for r in eval_rows]
    result: dict[str, dict] = {}

    for label_name in ("relation", "role", "operation", "control", "signature"):
        train_labels = [target(r)[label_name] for r in train_rows]
        eval_labels = [target(r)[label_name] for r in eval_rows]

        nb = MultinomialNB()
        nb.fit(train_text, train_labels)
        result[label_name] = {
            "majority": accuracy(majority(train_labels, len(eval_rows)), eval_labels),
            "length_only": accuracy(
                length_predict(train_text, train_labels, eval_text),
                eval_labels,
            ),
            "family_oracle": accuracy(
                family_predict(train_rows, label_name, eval_rows),
                eval_labels,
            ),
            "unigram_nb": accuracy(nb.predict(eval_text), eval_labels),
        }

    result["relation"]["cue_lexicon"] = accuracy(
        [cue_predict_relation(t) for t in eval_text],
        [target(r)["relation"] for r in eval_rows],
    )
    result["operation"]["cue_lexicon"] = accuracy(
        [cue_predict_operation(t) for t in eval_text],
        [target(r)["operation"] for r in eval_rows],
    )
    result["control"]["cue_lexicon"] = accuracy(
        [cue_predict_control(t) for t in eval_text],
        [target(r)["control"] for r in eval_rows],
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t2-curriculum", required=True)
    parser.add_argument("--reality-corpus", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.t2_curriculum))
    train = [r for r in rows if r.get("split") == "train"]
    dev = [r for r in rows if r.get("split") == "dev"]
    reality = read_jsonl(Path(args.reality_corpus))

    if len(train) != 720 or len(dev) != 288:
        raise SystemExit("T2 curriculum count drift")
    if any(r.get("private_identity_data") is not False for r in reality):
        raise SystemExit("reality corpus private-data drift")

    payload = {
        "schema": "alice.eipm.n0.qsre-t2-shortcut-audit.v0.1",
        "status": "PASS_QSRE_T2_SHORTCUT_AUDIT",
        "train_rows": len(train),
        "dev_rows": len(dev),
        "reality_rows": len(reality),
        "t2_train_to_t2_dev": evaluate_model(train, dev),
        "t2_train_to_reality": evaluate_model(train, reality),
        "interpretation": {
            "family_oracle_is_diagnostic_only": True,
            "unigram_nb_uses_query_text_only": True,
            "no_neural_model": True,
            "no_gradient": True,
            "private_identity_data": False,
        },
    }

    dev_sig = payload["t2_train_to_t2_dev"]["signature"]["unigram_nb"]
    real_sig = payload["t2_train_to_reality"]["signature"]["unigram_nb"]
    payload["summary"] = {
        "unigram_nb_signature_t2_dev": dev_sig,
        "unigram_nb_signature_reality": real_sig,
        "template_shortcut_warning": bool(dev_sig >= 0.70),
        "distribution_gap_warning": bool(dev_sig - real_sig >= 0.20),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
