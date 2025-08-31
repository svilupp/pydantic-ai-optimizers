"""Shared utilities and data models for customer support classification."""

from __future__ import annotations

from agent import SupportClassification


def calculate_classification_score(
    predicted: SupportClassification, expected: SupportClassification
) -> float:
    """Calculate similarity score between predicted and expected classifications.

    Scoring weights:
    - Category: 40% (most important)
    - Urgency: 20%
    - Surface: 15%
    - Component: 15%
    - Identifiers: 10%
    """
    score = 0.0

    # Category match (40%)
    if predicted.category == expected.category:
        score += 0.4

    # Urgency match (20%)
    if predicted.urgency == expected.urgency:
        score += 0.2

    # Surface match (15%)
    if predicted.surface == expected.surface:
        score += 0.15

    # Component match (15%)
    if predicted.component == expected.component:
        score += 0.15
    elif predicted.component is None and expected.component is None:
        score += 0.15
    elif predicted.component is not None and expected.component is not None:
        # Partial credit for similar components
        if (
            predicted.component.lower() in expected.component.lower()
            or expected.component.lower() in predicted.component.lower()
        ):
            score += 0.075

    # Identifiers match (10%)
    if set(predicted.identifiers) == set(expected.identifiers):
        score += 0.1
    elif predicted.identifiers and expected.identifiers:
        # Partial credit based on intersection
        intersection = set(predicted.identifiers) & set(expected.identifiers)
        union = set(predicted.identifiers) | set(expected.identifiers)
        if union:
            jaccard = len(intersection) / len(union)
            score += 0.1 * jaccard
    elif not predicted.identifiers and not expected.identifiers:
        score += 0.1

    return min(1.0, score)
