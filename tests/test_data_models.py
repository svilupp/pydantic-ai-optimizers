"""Tests for data model classes."""

from pathlib import Path

import pytest

from pydantic_ai_optimizers.optimizer import Candidate, CaseEval, FailedMutation


class TestCandidate:
    """Tests for Candidate dataclass."""

    def test_candidate_creation_with_minimal_args(self):
        """Test creating a Candidate with minimal arguments."""
        prompt_path = Path("test_prompt.txt")
        candidate = Candidate(prompt_path=prompt_path)

        assert candidate.prompt_path == prompt_path
        assert candidate.parent_index is None
        assert candidate.note == ""

    def test_candidate_creation_with_all_args(self):
        """Test creating a Candidate with all arguments."""
        prompt_path = Path("test_prompt.txt")
        parent_index = 2
        note = "test note"

        candidate = Candidate(prompt_path=prompt_path, parent_index=parent_index, note=note)

        assert candidate.prompt_path == prompt_path
        assert candidate.parent_index == parent_index
        assert candidate.note == note

    def test_candidate_is_frozen(self):
        """Test that Candidate is immutable."""
        candidate = Candidate(prompt_path=Path("test.txt"))

        with pytest.raises(AttributeError):
            candidate.prompt_path = Path("different.txt")

    def test_candidate_equality(self):
        """Test Candidate equality comparison."""
        path1 = Path("test1.txt")
        path2 = Path("test2.txt")

        candidate1 = Candidate(prompt_path=path1, parent_index=1, note="note1")
        candidate2 = Candidate(prompt_path=path1, parent_index=1, note="note1")
        candidate3 = Candidate(prompt_path=path2, parent_index=1, note="note1")

        assert candidate1 == candidate2
        assert candidate1 != candidate3

    def test_candidate_repr(self):
        """Test Candidate string representation."""
        candidate = Candidate(prompt_path=Path("test.txt"), parent_index=5, note="test note")

        repr_str = repr(candidate)
        assert "Candidate" in repr_str
        assert "test.txt" in repr_str
        assert "5" in repr_str
        assert "test note" in repr_str


class TestCaseEval:
    """Tests for CaseEval dataclass."""

    def test_case_eval_creation_with_all_args(self):
        """Test creating a CaseEval with all arguments."""
        score = 0.85
        reasons = ["reason1", "reason2"]
        output = {"result": "success"}

        case_eval = CaseEval(score=score, reasons=reasons, output=output)

        assert case_eval.score == score
        assert case_eval.reasons == reasons
        assert case_eval.output == output

    def test_case_eval_creation_with_minimal_args(self):
        """Test creating a CaseEval with minimal arguments."""
        score = 0.5
        reasons = []
        output = "simple output"

        case_eval = CaseEval(score=score, reasons=reasons, output=output)

        assert case_eval.score == score
        assert case_eval.reasons == reasons
        assert case_eval.output == output

    def test_case_eval_is_frozen(self):
        """Test that CaseEval is immutable."""
        case_eval = CaseEval(score=0.8, reasons=["test"], output="output")

        with pytest.raises(AttributeError):
            case_eval.score = 0.9

    def test_case_eval_equality(self):
        """Test CaseEval equality comparison."""
        case_eval1 = CaseEval(score=0.8, reasons=["test"], output="output1")
        case_eval2 = CaseEval(score=0.8, reasons=["test"], output="output1")
        case_eval3 = CaseEval(score=0.7, reasons=["test"], output="output1")

        assert case_eval1 == case_eval2
        assert case_eval1 != case_eval3

    def test_case_eval_with_different_output_types(self):
        """Test CaseEval with various output types."""
        outputs = [
            "string output",
            {"dict": "output"},
            ["list", "output"],
            42,
            None,
        ]

        for output in outputs:
            case_eval = CaseEval(score=0.5, reasons=[], output=output)
            assert case_eval.output == output


class TestFailedMutation:
    """Tests for FailedMutation dataclass."""

    def test_failed_mutation_creation_with_all_args(self):
        """Test creating a FailedMutation with all arguments."""
        prompt_text = "This is a test prompt"
        parent_index = 3
        failure_reasons = ["reason1", "reason2"]
        average_score = 0.35
        attempt_number = 5

        failed_mutation = FailedMutation(
            prompt_text=prompt_text,
            parent_index=parent_index,
            failure_reasons=failure_reasons,
            average_score=average_score,
            attempt_number=attempt_number,
        )

        assert failed_mutation.prompt_text == prompt_text
        assert failed_mutation.parent_index == parent_index
        assert failed_mutation.failure_reasons == failure_reasons
        assert failed_mutation.average_score == average_score
        assert failed_mutation.attempt_number == attempt_number

    def test_failed_mutation_is_frozen(self):
        """Test that FailedMutation is immutable."""
        failed_mutation = FailedMutation(
            prompt_text="test",
            parent_index=1,
            failure_reasons=["test"],
            average_score=0.3,
            attempt_number=1,
        )

        with pytest.raises(AttributeError):
            failed_mutation.attempt_number = 2

    def test_failed_mutation_equality(self):
        """Test FailedMutation equality comparison."""
        failed_mutation1 = FailedMutation(
            prompt_text="test",
            parent_index=1,
            failure_reasons=["reason"],
            average_score=0.3,
            attempt_number=1,
        )
        failed_mutation2 = FailedMutation(
            prompt_text="test",
            parent_index=1,
            failure_reasons=["reason"],
            average_score=0.3,
            attempt_number=1,
        )
        failed_mutation3 = FailedMutation(
            prompt_text="test",
            parent_index=2,  # Different parent
            failure_reasons=["reason"],
            average_score=0.3,
            attempt_number=1,
        )

        assert failed_mutation1 == failed_mutation2
        assert failed_mutation1 != failed_mutation3

    def test_failed_mutation_with_empty_reasons(self):
        """Test FailedMutation with empty failure reasons."""
        failed_mutation = FailedMutation(
            prompt_text="test prompt",
            parent_index=0,
            failure_reasons=[],
            average_score=0.2,
            attempt_number=3,
        )

        assert failed_mutation.failure_reasons == []
        assert len(failed_mutation.failure_reasons) == 0

    def test_failed_mutation_repr(self):
        """Test FailedMutation string representation."""
        failed_mutation = FailedMutation(
            prompt_text="Short prompt",
            parent_index=2,
            failure_reasons=["Low accuracy", "Poor reasoning"],
            average_score=0.25,
            attempt_number=7,
        )

        repr_str = repr(failed_mutation)
        assert "FailedMutation" in repr_str
        assert "0.25" in repr_str
        assert "7" in repr_str

    def test_failed_mutation_with_long_prompt_text(self):
        """Test FailedMutation with very long prompt text."""
        long_prompt = "This is a very long prompt " * 100  # 2700+ characters

        failed_mutation = FailedMutation(
            prompt_text=long_prompt,
            parent_index=1,
            failure_reasons=["Too verbose"],
            average_score=0.1,
            attempt_number=10,
        )

        assert failed_mutation.prompt_text == long_prompt
        assert len(failed_mutation.prompt_text) > 1000


class TestDataModelInteractions:
    """Tests for interactions between data model classes."""

    def test_candidate_with_failed_mutation_workflow(self):
        """Test typical workflow involving Candidate and FailedMutation."""
        # Create a parent candidate
        parent_path = Path("parent_prompt.txt")
        parent_candidate = Candidate(prompt_path=parent_path, note="seed")

        # Create a child candidate
        child_path = Path("child_prompt.txt")
        child_candidate = Candidate(prompt_path=child_path, parent_index=0, note="mutated")

        # Create a failed mutation representing the child's failure
        failed_mutation = FailedMutation(
            prompt_text="Failed child prompt text",
            parent_index=0,
            failure_reasons=["Low performance", "Poor accuracy"],
            average_score=0.3,
            attempt_number=1,
        )

        # Verify the relationships
        assert child_candidate.parent_index == 0  # References parent
        assert failed_mutation.parent_index == 0  # Also references parent
        assert parent_candidate.parent_index is None  # Parent has no parent

    def test_case_eval_aggregation(self):
        """Test aggregating multiple CaseEval instances."""
        case_evals = [
            CaseEval(score=0.8, reasons=["minor issue"], output="result1"),
            CaseEval(score=0.9, reasons=[], output="result2"),
            CaseEval(score=0.6, reasons=["major issue", "formatting"], output="result3"),
        ]

        # Test aggregation operations
        total_score = sum(ce.score for ce in case_evals)
        avg_score = total_score / len(case_evals)
        all_reasons = [reason for ce in case_evals for reason in ce.reasons]

        assert abs(avg_score - 0.7666666666666667) < 0.001  # (0.8 + 0.9 + 0.6) / 3
        assert len(all_reasons) == 3
        assert "minor issue" in all_reasons
        assert "major issue" in all_reasons
        assert "formatting" in all_reasons
