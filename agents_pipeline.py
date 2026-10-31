"""Two-agent customer-support baseline with leakage-aware validation.

CSV columns:
  train.csv: instruction, intent, category, response (flags optional)
  validation.csv: instruction, intent, category (response/flags optional)

Run:
  python agents_pipeline.py train --train C:/Users/hp/Downloads/train_clean.csv --validation C:/Users/hp/Downloads/validation_unseen.csv
  python agents_pipeline.py predict "Where is my order?"
  python agents_pipeline.py interactive

Model selection uses a reproducible, exact-query-grouped holdout from train.csv.
validation.csv is kept out of model selection and reported as the final evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = HERE / "agent_models"


def normalize_query(value: Any) -> str:
    """Normalize wording for duplicate detection, not for model input."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def template_group(value: Any) -> str:
    """Conservative grouping key for placeholder/numeric template variants."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"\{\{.*?\}\}", " slot ", text)
    text = re.sub(r"\b\d+\b", " number ", text)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def load_frame(path: str | Path, required: set[str], name: str) -> pd.DataFrame:
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f"{name} not found: {csv_path.resolve()}")
    frame = pd.read_csv(csv_path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{name} is missing columns: {', '.join(sorted(missing))}. "
            f"Found: {', '.join(map(str, frame.columns))}"
        )
    before = len(frame)
    frame = frame.dropna(subset=list(required)).copy()
    for column in required:
        frame[column] = frame[column].astype(str).str.strip()
    for column in required:
        frame = frame[frame[column] != ""]
    frame = frame.reset_index(drop=True)
    dropped = before - len(frame)
    if frame.empty:
        raise ValueError(f"{name} has no complete, nonblank rows.")
    if dropped:
        print(f"{name}: removed {dropped} row(s) with missing/blank required fields.")
    return frame


def build_features() -> FeatureUnion:
    # The dataset is close to balanced by intent, so do not up-weight classes.
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    stop_words="english",
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=100000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return features


def build_model(C: float) -> Pipeline:
    return Pipeline([
        ("features", build_features()),
        ("classifier", LogisticRegression(
            C=C, max_iter=2500, solver="lbfgs", class_weight=None
        )),
    ])


def category_mapping(frame: pd.DataFrame) -> dict[str, str]:
    counts = frame.groupby("intent")["category"].nunique()
    conflicts = counts[counts > 1].index.astype(str).tolist()
    if conflicts:
        raise ValueError(f"Intent maps to multiple categories: {conflicts}")
    return frame.groupby("intent")["category"].first().astype(str).to_dict()


class IntentAnalyst:
    """Agent 1: intent/category prediction and a probability-based uncertainty flag."""

    def __init__(self, model: Pipeline, intent_to_category: dict[str, str], uncertainty_thresholds: tuple[float, float] = (0.05, 0.005)):
        self.model = model
        self.intent_to_category = intent_to_category
        self.uncertainty_thresholds = uncertainty_thresholds

    def analyze(self, query: str) -> dict[str, Any]:
        query = unicodedata.normalize("NFKC", str(query)).strip()
        if not query:
            return {
                "query": query,
                "intent": None,
                "category": None,
                "intent_confidence": 0.0,
                "top_two_margin": 0.0,
                "uncertain": True,
                "action": "clarify_or_review",
            }
        probs = self.model.predict_proba([query])[0]
        classes = self.model.named_steps["classifier"].classes_
        order = np.argsort(probs)[::-1]
        idx = int(order[0])
        confidence = float(probs[idx])
        runner_up = float(probs[order[1]]) if len(order) > 1 else 0.0
        margin = confidence - runner_up
        intent = str(classes[idx])
        # These are conservative routing thresholds, not calibrated probabilities.
        min_confidence, min_margin = self.uncertainty_thresholds
        uncertain = confidence < min_confidence or margin < min_margin
        return {
            "query": query,
            "intent": intent,
            "category": self.intent_to_category.get(intent, "UNKNOWN"),
            "intent_confidence": round(confidence, 4),
            "top_two_margin": round(margin, 4),
            "uncertain": uncertain,
            "action": "clarify_or_review" if uncertain else "send_to_agent_2",
        }


class ResponseSpecialistAgent:
    """Agent 2: rank support examples using word/character similarity and intent."""

    def __init__(
        self,
        knowledge_df: pd.DataFrame,
        analyst: IntentAnalyst,
        min_similarity: float = 0.08,
        index_bundle: dict[str, Any] | None = None,
    ):
        self.kb = knowledge_df.dropna(subset=["instruction", "intent", "response"]).copy()
        for col in ("instruction", "intent", "response"):
            self.kb[col] = self.kb[col].astype(str).str.strip()
        self.kb = self.kb[
            (self.kb.instruction != "") & (self.kb.intent != "") & (self.kb.response != "")
        ].reset_index(drop=True)
        if self.kb.empty:
            raise ValueError("The response knowledge base has no usable examples.")
        self.analyst = analyst
        self.min_similarity = min_similarity
        self.kb_norm = self.kb.instruction.map(normalize_query).to_numpy()
        if index_bundle:
            self.word_vectorizer = index_bundle["word_vectorizer"]
            self.char_vectorizer = index_bundle["char_vectorizer"]
            self.word_vectors = index_bundle["word_vectors"]
            self.char_vectors = index_bundle["char_vectors"]
            self.kb_norm = index_bundle.get("kb_norm", self.kb.instruction.map(normalize_query).to_numpy())
        else:
            self.word_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), min_df=1, max_features=100000,
                sublinear_tf=True, strip_accents="unicode", stop_words="english"
            )
            self.char_vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                max_features=100000, sublinear_tf=True
            )
            instructions = self.kb.instruction.tolist()
            self.word_vectors = self.word_vectorizer.fit_transform(instructions)
            self.char_vectors = self.char_vectorizer.fit_transform(instructions)

    def export_index(self) -> dict[str, Any]:
        return {"word_vectorizer": self.word_vectorizer, "char_vectorizer": self.char_vectorizer,
                "word_vectors": self.word_vectors, "char_vectors": self.char_vectors, "kb_norm": self.kb_norm}

    def rank(self, query: str, exclude_exact_query: bool = False) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        analysis = self.analyst.analyze(query)
        word_scores = cosine_similarity(
            self.word_vectorizer.transform([query]), self.word_vectors
        ).ravel()
        char_scores = cosine_similarity(
            self.char_vectorizer.transform([query]), self.char_vectors
        ).ravel()
        similarity = 0.65 * word_scores + 0.35 * char_scores

        # Keep intent evidence as a modest reranking signal; lexical relevance remains primary.
        probs = self.analyst.model.predict_proba([query])[0]
        classes = self.analyst.model.named_steps["classifier"].classes_
        prob_by_intent = {str(label): float(prob) for label, prob in zip(classes, probs)}
        max_prob = max(prob_by_intent.values()) if len(prob_by_intent) else 0.0
        prior = np.array(
            [prob_by_intent.get(str(label), 0.0) / max_prob if max_prob else 0.0
             for label in self.kb.intent]
        )
        scores = similarity * (0.85 + 0.15 * prior)
        eligible = np.ones(len(self.kb), dtype=bool)
        if exclude_exact_query:
            norm = normalize_query(query)
            eligible &= self.kb_norm != norm
        if not eligible.any():
            eligible[:] = True
        ranked = np.argsort(np.where(eligible, scores, -1.0))[::-1]
        return analysis, scores, ranked

    def generate_response(self, query: str, top_k: int = 3) -> dict[str, Any]:
        query = str(query).strip()
        if not query:
            return {"status": "Clarification Required", "response": "Please describe your issue.", "references": []}
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        analysis, scores, ranked = self.rank(query)
        best_i = int(ranked[0])
        best = self.kb.iloc[best_i]
        score = float(scores[best_i])
        margin = score - (float(scores[ranked[1]]) if len(ranked) > 1 else 0.0)

        if analysis["uncertain"]:
            return {
                "status": "Clarification Required",
                "analysis": analysis,
                "response": "I’m not certain which issue you mean. Could you add a little more detail?",
                "references": [],
            }
        if score < self.min_similarity:
            return {
                "status": "No Confident Match",
                "analysis": analysis,
                "similarity_score": round(score, 4),
                "response": "I couldn’t find a sufficiently similar support answer. Please clarify or contact support.",
                "references": [],
            }

        refs = []
        for i in ranked[:top_k]:
            row = self.kb.iloc[int(i)]
            refs.append({
                "matched_instruction": str(row.instruction),
                "matched_intent": str(row.intent),
                "response": str(row.response),
                "similarity": round(float(scores[i]), 4),
            })
        response_text = str(best.response)
        ask_phrases = (
            "could you please provide", "can you provide", "please provide more details",
            "could you share", "please confirm", "could you clarify", "may i have your",
        )
        status = "Needs Information" if any(p in response_text.lower() for p in ask_phrases) else "Candidate Answer"
        return {
            "status": status,
            "analysis": analysis,
            "predicted_intent": analysis["intent"],
            "retrieved_intent": str(best.intent),
            "similarity_score": round(score, 4),
            "score_margin": round(margin, 4),
            "matched_instruction": str(best.instruction),
            "response": response_text,
            "references": refs,
        }


class TwoAgentSystem:
    def __init__(self, model: Pipeline, intent_to_category: dict[str, str], knowledge_df: pd.DataFrame, uncertainty_thresholds: tuple[float, float] = (0.05, 0.005), retrieval_index: dict[str, Any] | None = None):
        self.agent1 = IntentAnalyst(model, intent_to_category, uncertainty_thresholds)
        self.agent2 = ResponseSpecialistAgent(knowledge_df, self.agent1, index_bundle=retrieval_index)

    def ask(self, query: str, top_k: int = 3) -> dict[str, Any]:
        return self.agent2.generate_response(query, top_k)


def classification_metrics(model: Pipeline, frame: pd.DataFrame, mapping: dict[str, str]) -> dict[str, Any]:
    x = frame.instruction.astype(str)
    y = frame.intent.astype(str)
    pred = model.predict(x)
    cats_true = frame.category.astype(str).to_numpy()
    cats_pred = np.array([mapping.get(str(label), "UNKNOWN") for label in pred])
    probabilities = model.predict_proba(x)
    classes = model.named_steps["classifier"].classes_
    one_hot = np.column_stack([y.to_numpy() == label for label in classes]).astype(float)
    multiclass_brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    return {
        "rows": int(len(frame)),
        "intent_accuracy": float(accuracy_score(y, pred)),
        "intent_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "intent_weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "category_accuracy": float(accuracy_score(cats_true, cats_pred)),
        "intent_report": classification_report(y, pred, output_dict=True, zero_division=0),
        "intent_confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist(),
        "multiclass_brier_score_uncalibrated": multiclass_brier,
    }


def retrieval_metrics(agent: ResponseSpecialistAgent, frame: pd.DataFrame, k: int = 3) -> dict[str, Any]:
    """Measure whether relevant intent examples occur in top retrieval results.

    Exact instruction matches in the KB are excluded for this evaluation to avoid
    rewarding a copied train/validation row. This is an intent relevance proxy;
    it does not claim to measure human response quality.
    """
    hit1, hitk, reciprocal = [], [], []
    kb_labels = agent.kb.intent.astype(str).to_numpy()
    instructions = frame.instruction.astype(str).tolist()
    expected = frame.intent.astype(str).tolist()
    for offset in range(0, len(instructions), 64):
        queries = instructions[offset:offset + 64]
        word_query = agent.word_vectorizer.transform(queries)
        char_query = agent.char_vectorizer.transform(queries)
        word_scores = cosine_similarity(word_query, agent.word_vectors)
        char_scores = cosine_similarity(char_query, agent.char_vectors)
        similarities = 0.65 * word_scores + 0.35 * char_scores
        probabilities = agent.analyst.model.predict_proba(queries)
        classes = agent.analyst.model.named_steps["classifier"].classes_
        for local_i, query in enumerate(queries):
            probs = {str(label): float(p) for label, p in zip(classes, probabilities[local_i])}
            max_prob = max(probs.values(), default=0.0)
            priors = np.array([probs.get(label, 0.0) / max_prob if max_prob else 0.0 for label in kb_labels])
            scores = similarities[local_i] * (0.85 + 0.15 * priors)
            eligible = agent.kb_norm != normalize_query(query)
            if not eligible.any(): eligible = np.ones(len(agent.kb), dtype=bool)
            ranked = np.argsort(np.where(eligible, scores, -1.0))[::-1]
            labels = kb_labels[ranked]
            relevant = labels == expected[offset + local_i]
            hit1.append(bool(relevant[:1].any()))
            hitk.append(bool(relevant[:k].any()))
            positions = np.flatnonzero(relevant)
            reciprocal.append(1.0 / (int(positions[0]) + 1) if len(positions) else 0.0)
    return {
        "rows": len(frame),
        "intent_hit_at_1": float(np.mean(hit1)) if hit1 else 0.0,
        f"intent_hit_at_{k}": float(np.mean(hitk)) if hitk else 0.0,
        "intent_mrr": float(np.mean(reciprocal)) if reciprocal else 0.0,
    }


def select_uncertainty_thresholds(probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray, max_confident_error: float = 0.05) -> dict[str, Any]:
    """Choose an abstention operating point on the training-only grouped holdout."""
    top = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    ordered = np.partition(probabilities, -2, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    correct = classes[top].astype(str) == labels.astype(str)
    confidence_grid = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20)
    margin_grid = (0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.08)
    candidates = []
    for min_confidence in confidence_grid:
        for min_margin in margin_grid:
            accepted = (confidence >= min_confidence) & (margin >= min_margin)
            coverage = float(accepted.mean())
            error = float((~correct[accepted]).mean()) if accepted.any() else 1.0
            if accepted.any() and error <= max_confident_error:
                candidates.append((coverage, -error, min_confidence + min_margin, min_confidence, min_margin, error))
    if not candidates:
        return {"min_confidence": 1.0, "min_margin": 1.0, "coverage": 0.0, "confident_error_rate": None, "max_confident_error": max_confident_error}
    best = max(candidates)
    return {"min_confidence": best[3], "min_margin": best[4], "coverage": best[0], "confident_error_rate": best[5], "max_confident_error": max_confident_error}


def run_training(
    train_path: str,
    validation_path: str,
    output_dir: str,
    seed: int,
    force: bool = False,
    target_min_accuracy: float = 0.92,
    target_max_accuracy: float = 0.95,
) -> None:
    required_train = {"instruction", "intent", "category", "response"}
    required_val = {"instruction", "intent", "category"}
    train = load_frame(train_path, required_train, "Training data")
    val = load_frame(validation_path, required_val, "Validation data")
    mapping = category_mapping(train)

    train_intents = set(train.intent.astype(str))
    unseen = sorted(set(val.intent.astype(str)) - train_intents)
    if unseen:
        raise ValueError(f"Validation contains intent(s) absent from training: {unseen}")
    val_mapping_conflict = val.groupby("intent").category.nunique()
    if (val_mapping_conflict > 1).any():
        raise ValueError("Validation data contains an intent mapped to multiple categories.")

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    bundle_path = out / "two_agent_system.joblib"
    if bundle_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing model artifact without --force: {bundle_path}")

    train_norm = train.instruction.map(normalize_query).to_numpy()
    train_groups = train.instruction.map(template_group).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    fit_idx, tune_idx = next(splitter.split(train, train.intent, groups=train_groups))
    fit_part, tune_part = train.iloc[fit_idx], train.iloc[tune_idx]
    if fit_part.intent.nunique() != train.intent.nunique():
        raise ValueError("Grouped tuning split lost one or more intents; use more training rows per intent.")

    print("\nDATA CHECK")
    print(f"Training rows: {len(train):,}; validation rows: {len(val):,}")
    print(f"Intents: {train.intent.nunique()}; categories: {train.category.nunique()}")
    print(f"Training flags: {train['flags'].nunique() if 'flags' in train else 'not supplied'}")
    print(f"Validation flags: {val['flags'].nunique() if 'flags' in val else 'not supplied'}")
    print(f"Template-grouped tuning split: {len(fit_part):,} fit / {len(tune_part):,} tune rows (seed={seed})")

    # Apply the requested operating band on grouped training-only data. This is
    # model selection, not a post-hoc adjustment to the untouched validation score.
    candidates: list[dict[str, float]] = []
    candidate_probabilities: dict[float, np.ndarray] = {}
    print("\nMODEL SELECTION (training-only grouped holdout)")
    if not (0.0 <= target_min_accuracy <= target_max_accuracy <= 1.0):
        raise ValueError("Target tuning accuracy bounds must satisfy 0 <= min <= max <= 1.")
    print(f"Requested training-only accuracy band: {target_min_accuracy:.0%}–{target_max_accuracy:.0%}")
    # Fit text features once; only the lightweight classifier is repeated per C.
    tune_features = build_features()
    x_fit = tune_features.fit_transform(fit_part.instruction.astype(str))
    x_tune = tune_features.transform(tune_part.instruction.astype(str))
    y_fit = fit_part.intent.astype(str)
    y_tune = tune_part.intent.astype(str)
    for c_value in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        candidate = LogisticRegression(C=c_value, max_iter=2500, solver="lbfgs", class_weight=None)
        candidate.fit(x_fit, y_fit)
        prediction = candidate.predict(x_tune)
        candidate_probabilities[c_value] = candidate.predict_proba(x_tune)
        accuracy = float(accuracy_score(y_tune, prediction))
        macro = float(f1_score(y_tune, prediction, average="macro", zero_division=0))
        candidates.append({"C": c_value, "accuracy": accuracy, "macro_f1": macro})
        print(f"C={c_value:<6} tune accuracy={accuracy:.4f} | macro-F1={macro:.4f}")

    in_band = [item for item in candidates if target_min_accuracy <= item["accuracy"] <= target_max_accuracy]
    if in_band:
        selected = max(in_band, key=lambda item: (item["macro_f1"], item["accuracy"], -item["C"]))
        selection_note = "best macro-F1 among candidates in the requested training-only band"
    else:
        def distance(item):
            if item["accuracy"] < target_min_accuracy: return target_min_accuracy - item["accuracy"]
            if item["accuracy"] > target_max_accuracy: return item["accuracy"] - target_max_accuracy
            return 0.0
        selected = min(candidates, key=lambda item: (distance(item), -item["macro_f1"]))
        selection_note = "closest training-only candidate; no candidate fell in requested band"
    best_c = selected["C"]
    best_score = selected["macro_f1"]
    uncertainty_selection = select_uncertainty_thresholds(
        candidate_probabilities[best_c], y_tune.to_numpy(), candidate.classes_
    )
    print(f"Training-only uncertainty operating point: {uncertainty_selection}")
    print(f"Selected C={best_c}: tuning accuracy={selected['accuracy']:.4f} ({selection_note})")

    # Retrain on all training rows; validation remains a final holdout.
    model = build_model(float(best_c))
    model.fit(train.instruction.astype(str), train.intent.astype(str))
    dataset_hash = hashlib.sha256(Path(train_path).expanduser().read_bytes()).hexdigest()
    bundle_data = {
        "model": model,
        "intent_to_category": mapping,
        "knowledge_df": train[["instruction", "intent", "category", "response"]].copy(),
        "metrics": {"status": "metrics_pending"},
        "uncertainty_thresholds": [uncertainty_selection["min_confidence"], uncertainty_selection["min_margin"]],
        "metadata": {"model_version": f"word-char-tfidf-logreg-C{best_c}", "train_rows": len(train), "train_sha256": dataset_hash, "grouping": "normalized text with placeholders/numbers collapsed", "random_seed": seed},
    }
    joblib.dump(bundle_data, bundle_path)
    print(f"Saved fitted model before validation metrics: {bundle_path}")
    analyst = IntentAnalyst(model, mapping)
    response_agent = ResponseSpecialistAgent(train, analyst)
    bundle_data["retrieval_index"] = response_agent.export_index()
    joblib.dump(bundle_data, bundle_path)

    val_norm = val.instruction.map(normalize_query)
    val_templates = val.instruction.map(template_group)
    train_norm_set = set(train_norm)
    train_template_set = set(train_groups)
    clean_mask = ~val_norm.isin(train_norm_set)
    clean_val = val.loc[clean_mask].reset_index(drop=True)
    duplicate_count = int((~clean_mask).sum())
    group_clean_mask = ~val_templates.isin(train_template_set)
    group_clean_val = val.loc[group_clean_mask].reset_index(drop=True)

    final_metrics = {
        "seed": seed,
        "requested_tuning_accuracy_band": [target_min_accuracy, target_max_accuracy],
        "selected_C": float(best_c),
        "selection_note": selection_note,
        "uncertainty_threshold_selection": uncertainty_selection,
        "tuning_holdout_accuracy": float(selected["accuracy"]),
        "tuning_holdout_macro_f1": float(best_score),
        "validation_exact_or_normalized_query_overlap": duplicate_count,
        "validation_template_group_overlap": int(val_templates.isin(train_template_set).sum()),
        "training_normalized_duplicate_rows": int(len(train_norm) - len(set(train_norm))),
        "training_template_duplicate_rows": int(len(train_groups) - len(set(train_groups))),
        "validation_total": classification_metrics(model, val, mapping),
        "validation_no_train_query_overlap": classification_metrics(model, clean_val, mapping) if len(clean_val) else None,
        "validation_no_train_template_overlap": classification_metrics(model, group_clean_val, mapping) if len(group_clean_val) else None,
        "retrieval_validation_no_exact_query_copy": retrieval_metrics(response_agent, clean_val) if len(clean_val) else None,
        "retrieval_validation_no_template_overlap": retrieval_metrics(response_agent, group_clean_val) if len(group_clean_val) else None,
        "limitations": [
            "Logistic-regression probabilities are ranking scores and are not calibrated confidence estimates.",
            "Retrieval hit-rate measures matching intent, not whether a human considers the reply helpful.",
        ],
    }

    bundle_data["metrics"] = final_metrics
    joblib.dump(bundle_data, bundle_path)
    (out / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")

    print("\nFINAL HELD-OUT METRICS")
    print(f"Chosen C: {best_c}; tuning accuracy: {selected['accuracy']:.4f}; "
          f"tuning macro-F1: {best_score:.4f} ({selection_note})")
    print(f"Validation normalized-query overlap with train: {duplicate_count}/{len(val)}")
    for label, section in (("All supplied validation", final_metrics["validation_total"]),
                           ("No normalized-query overlap", final_metrics["validation_no_train_query_overlap"]),
                           ("No template-group overlap", final_metrics["validation_no_train_template_overlap"])):
        if section is None:
            continue
        print(f"{label} (n={section['rows']}): intent accuracy={section['intent_accuracy']:.4f}, "
              f"intent macro-F1={section['intent_macro_f1']:.4f}, "
              f"category accuracy={section['category_accuracy']:.4f}")
    print("Retrieval relevance proxy:", final_metrics["retrieval_validation_no_exact_query_copy"])
    report_frame = clean_val if len(clean_val) else val
    report_prediction = model.predict(report_frame.instruction.astype(str))
    print("\nClassification report (no-overlap validation when available):")
    print(classification_report(report_frame.intent.astype(str), report_prediction, zero_division=0))

    system = TwoAgentSystem(model, mapping, train, (uncertainty_selection["min_confidence"], uncertainty_selection["min_margin"]))
    examples = val.loc[clean_mask].head(5) if len(clean_val) else val.head(5)
    print("\nEND-TO-END EXAMPLES")
    for row in examples.itertuples(index=False):
        result = system.ask(str(row.instruction), top_k=3)
        print(json.dumps({
            "query": str(row.instruction),
            "expected_intent": str(row.intent),
            "agent1": result.get("analysis"),
            "agent2_status": result.get("status"),
            "retrieved_intent": result.get("retrieved_intent"),
            "response": result.get("response"),
            "references": result.get("references", []),
        }, ensure_ascii=False))
    print(f"\nSaved model: {bundle_path}")
    print(f"Saved metrics: {out / 'metrics.json'}")


def load_system(path: str | Path) -> TwoAgentSystem:
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}. Run the train command first.")
    bundle = joblib.load(model_path)
    thresholds = tuple(bundle.get("uncertainty_thresholds", (0.05, 0.005)))
    system = TwoAgentSystem(bundle["model"], bundle["intent_to_category"], bundle["knowledge_df"], thresholds, bundle.get("retrieval_index"))
    system.model_version = bundle.get("metadata", {}).get("model_version", "legacy-joblib")
    return system


def main() -> None:
    parser = argparse.ArgumentParser(description="Intent analyst + response specialist.")
    sub = parser.add_subparsers(dest="command", required=True)
    train_p = sub.add_parser("train", help="Tune on a grouped training holdout and evaluate validation.")
    train_p.add_argument("--train", default=r"C:\Users\hp\Downloads\train_clean.csv")
    train_p.add_argument("--validation", default=r"C:\Users\hp\Downloads\validation_unseen.csv")
    train_p.add_argument("--output-dir", default=str(DEFAULT_MODEL_DIR))
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--target-min-accuracy", type=float, default=0.92)
    train_p.add_argument("--target-max-accuracy", type=float, default=0.95)
    train_p.add_argument("--force", action="store_true", help="Allow replacing an existing model bundle after reviewing its metrics.")
    predict_p = sub.add_parser("predict", help="Run both agents for one query.")
    predict_p.add_argument("query")
    predict_p.add_argument("--model", default=str(DEFAULT_MODEL_DIR / "two_agent_system.joblib"))
    predict_p.add_argument("--top-k", type=int, default=3)
    interactive_p = sub.add_parser("interactive", help="Ask questions interactively.")
    interactive_p.add_argument("--model", default=str(DEFAULT_MODEL_DIR / "two_agent_system.joblib"))

    args = parser.parse_args()
    if args.command == "train":
        run_training(
            args.train,
            args.validation,
            args.output_dir,
            args.seed,
            args.force,
            args.target_min_accuracy,
            args.target_max_accuracy,
        )
    elif args.command == "predict":
        system = load_system(args.model)
        print(json.dumps(system.ask(args.query, args.top_k), indent=2, ensure_ascii=False))
    else:
        system = load_system(args.model)
        print("Ready. Type quit or exit to stop.")
        while True:
            try:
                query = input("\nCustomer: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if query.lower() in {"quit", "exit"}:
                break
            print(json.dumps(system.ask(query), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
