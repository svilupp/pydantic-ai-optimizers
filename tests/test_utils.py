"""Tests for example utility functions."""

import pytest

# Import the SupportClassification from the customer support example
# to test the calculate_classification_score function


class MockSupportClassification:
    """Mock SupportClassification for testing."""

    def __init__(
        self, category="billing", urgency="medium", surface="web", component=None, identifiers=None
    ):
        self.category = category
        self.urgency = urgency
        self.surface = surface
        self.component = component
        self.identifiers = identifiers if identifiers is not None else []


class TestClassificationScore:
    """Tests for calculate_classification_score utility function."""

    def test_perfect_match(self):
        """Test scoring when all fields match perfectly."""
        # Import here to avoid import issues in CI
        try:
            from examples.customer_support.utils import calculate_classification_score
        except ImportError:
            pytest.skip("Customer support example not available")

        predicted = MockSupportClassification(
            category="billing",
            urgency="high",
            surface="web",
            component="payment_processing",
            identifiers=["user_123", "invoice_456"],
        )

        expected = MockSupportClassification(
            category="billing",
            urgency="high",
            surface="web",
            component="payment_processing",
            identifiers=["user_123", "invoice_456"],
        )

        score = calculate_classification_score(predicted, expected)
        assert score == 1.0

    def test_category_mismatch(self):
        """Test scoring when category doesn't match."""
        try:
            from examples.customer_support.utils import calculate_classification_score
        except ImportError:
            pytest.skip("Customer support example not available")

        predicted = MockSupportClassification(category="authentication")
        expected = MockSupportClassification(category="billing")

        # Category is worth 40%, so max score is 0.6
        score = calculate_classification_score(predicted, expected)
        assert score <= 0.6

    def test_urgency_mismatch(self):
        """Test scoring when urgency doesn't match."""
        try:
            from examples.customer_support.utils import calculate_classification_score
        except ImportError:
            pytest.skip("Customer support example not available")

        predicted = MockSupportClassification(category="billing", urgency="low")
        expected = MockSupportClassification(category="billing", urgency="high")

        # Missing urgency (20%), so max score is 0.8
        score = calculate_classification_score(predicted, expected)
        assert score == 0.8

    def test_partial_component_match(self):
        """Test scoring with partial component matching."""
        try:
            from examples.customer_support.utils import calculate_classification_score
        except ImportError:
            pytest.skip("Customer support example not available")

        predicted = MockSupportClassification(
            category="billing",
            urgency="medium",
            surface="web",
            component="payment_gateway",
            identifiers=[],
        )

        expected = MockSupportClassification(
            category="billing",
            urgency="medium",
            surface="web",
            component="payment_system_gateway",
            identifiers=[],
        )

        score = calculate_classification_score(predicted, expected)
        # Should get partial credit for component match
        assert 0.4 + 0.2 + 0.15 + 0.075 + 0.1 == score  # Perfect except partial component

    def test_identifier_overlap(self):
        """Test scoring with overlapping identifiers."""
        try:
            from examples.customer_support.utils import calculate_classification_score
        except ImportError:
            pytest.skip("Customer support example not available")

        predicted = MockSupportClassification(
            category="billing",
            urgency="medium",
            surface="web",
            component=None,
            identifiers=["user_123", "extra_id"],
        )

        expected = MockSupportClassification(
            category="billing",
            urgency="medium",
            surface="web",
            component=None,
            identifiers=["user_123", "different_id"],
        )

        score = calculate_classification_score(predicted, expected)

        # Jaccard similarity: intersection=1, union=3, so jaccard=1/3
        expected_score = 0.4 + 0.2 + 0.15 + 0.15 + (0.1 * (1 / 3))
        assert abs(score - expected_score) < 0.001

    def test_empty_identifiers_match(self):
        """Test scoring when both have empty identifiers."""
        try:
            from examples.customer_support.utils import calculate_classification_score
        except ImportError:
            pytest.skip("Customer support example not available")

        predicted = MockSupportClassification(category="billing", identifiers=[])

        expected = MockSupportClassification(category="billing", identifiers=[])

        score = calculate_classification_score(predicted, expected)

        # Should get full credit for empty identifiers matching
        assert score >= 0.9  # At least category + identifiers + some others

    def test_none_components_match(self):
        """Test scoring when both components are None."""
        try:
            from examples.customer_support.utils import calculate_classification_score
        except ImportError:
            pytest.skip("Customer support example not available")

        predicted = MockSupportClassification(
            category="billing", urgency="low", surface="mobile", component=None, identifiers=[]
        )

        expected = MockSupportClassification(
            category="billing", urgency="low", surface="mobile", component=None, identifiers=[]
        )

        score = calculate_classification_score(predicted, expected)
        assert score == 1.0

    def test_score_capped_at_one(self):
        """Test that score never exceeds 1.0."""
        try:
            from examples.customer_support.utils import calculate_classification_score
        except ImportError:
            pytest.skip("Customer support example not available")

        predicted = MockSupportClassification()
        expected = MockSupportClassification()

        score = calculate_classification_score(predicted, expected)
        assert score <= 1.0
