"""Tests for optimizer helper functions."""

import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from pydantic_ai_optimizers.optimizer import (
    _read_prompt_text,
    _reason_from_report_case,
    _score_from_report_case,
    _write_new_prompt_file,
)


class TestScoreFromReportCase:
    """Tests for _score_from_report_case function."""

    def test_score_from_scores_single(self):
        """Test score calculation from single score."""
        mock_score = Mock()
        mock_score.value = 0.8
        rc = Mock()
        rc.scores = {"test": mock_score}
        rc.assertions = {}

        result = _score_from_report_case(rc)
        assert result == 0.8

    def test_score_from_scores_multiple(self):
        """Test score calculation from multiple scores."""
        mock_score1 = Mock()
        mock_score1.value = 0.8
        mock_score2 = Mock()
        mock_score2.value = 0.6
        rc = Mock()
        rc.scores = {"test1": mock_score1, "test2": mock_score2}
        rc.assertions = {}

        result = _score_from_report_case(rc)
        assert result == 0.7  # (0.8 + 0.6) / 2

    def test_score_from_assertions_all_true(self):
        """Test score calculation from assertions when all pass."""
        mock_assertion1 = Mock()
        mock_assertion1.value = True
        mock_assertion2 = Mock()
        mock_assertion2.value = True
        rc = Mock()
        rc.scores = {}
        rc.assertions = {"test1": mock_assertion1, "test2": mock_assertion2}

        result = _score_from_report_case(rc)
        assert result == 1.0

    def test_score_from_assertions_mixed(self):
        """Test score calculation from assertions with mixed results."""
        mock_assertion1 = Mock()
        mock_assertion1.value = True
        mock_assertion2 = Mock()
        mock_assertion2.value = False
        rc = Mock()
        rc.scores = {}
        rc.assertions = {"test1": mock_assertion1, "test2": mock_assertion2}

        result = _score_from_report_case(rc)
        assert result == 0.5

    def test_score_from_empty(self):
        """Test score calculation when no scores or assertions."""
        rc = Mock()
        rc.scores = {}
        rc.assertions = {}

        result = _score_from_report_case(rc)
        assert result == 0.0

    def test_score_prefers_scores_over_assertions(self):
        """Test that scores take precedence over assertions."""
        mock_score = Mock()
        mock_score.value = 0.8
        mock_assertion = Mock()
        mock_assertion.value = True
        rc = Mock()
        rc.scores = {"test": mock_score}
        rc.assertions = {"test": mock_assertion}

        result = _score_from_report_case(rc)
        assert result == 0.8


class TestReasonFromReportCase:
    """Tests for _reason_from_report_case function."""

    def test_reason_from_failed_assertions(self):
        """Test reason extraction from failed assertions."""
        mock_assertion1 = Mock()
        mock_assertion1.value = False
        mock_assertion1.reason = "Failed test 1"
        mock_assertion2 = Mock()
        mock_assertion2.value = True
        mock_assertion2.reason = "Passed test 2"
        rc = Mock()
        rc.assertions = {"test1": mock_assertion1, "test2": mock_assertion2}
        rc.scores = {}

        result = _reason_from_report_case(rc)
        assert result == ["Failed test 1"]

    def test_reason_from_imperfect_scores(self):
        """Test reason extraction from non-perfect scores."""
        mock_score1 = Mock()
        mock_score1.value = 0.8
        mock_score1.reason = "Not quite perfect"
        mock_score2 = Mock()
        mock_score2.value = 1.0
        mock_score2.reason = "Perfect score"
        rc = Mock()
        rc.scores = {"test1": mock_score1, "test2": mock_score2}
        rc.assertions = {}

        result = _reason_from_report_case(rc)
        assert result == ["Not quite perfect"]

    def test_reason_mixed_sources(self):
        """Test reason extraction from both assertions and scores."""
        mock_assertion = Mock()
        mock_assertion.value = False
        mock_assertion.reason = "Failed assertion"
        mock_score = Mock()
        mock_score.value = 0.5
        mock_score.reason = "Low score"
        rc = Mock()
        rc.assertions = {"assertion": mock_assertion}
        rc.scores = {"score": mock_score}

        result = _reason_from_report_case(rc)
        assert set(result) == {"Failed assertion", "Low score"}

    def test_reason_ignores_empty_reasons(self):
        """Test that empty or whitespace-only reasons are ignored."""
        mock_assertion1 = Mock()
        mock_assertion1.value = False
        mock_assertion1.reason = "  "  # whitespace only
        mock_assertion2 = Mock()
        mock_assertion2.value = False
        mock_assertion2.reason = ""  # empty
        mock_assertion3 = Mock()
        mock_assertion3.value = False
        mock_assertion3.reason = "Real reason"
        rc = Mock()
        rc.assertions = {
            "test1": mock_assertion1,
            "test2": mock_assertion2,
            "test3": mock_assertion3,
        }
        rc.scores = {}

        result = _reason_from_report_case(rc)
        assert result == ["Real reason"]

    def test_reason_strips_whitespace(self):
        """Test that reasons have whitespace stripped."""
        mock_assertion = Mock()
        mock_assertion.value = False
        mock_assertion.reason = "  Leading and trailing spaces  "
        rc = Mock()
        rc.assertions = {"test": mock_assertion}
        rc.scores = {}

        result = _reason_from_report_case(rc)
        assert result == ["Leading and trailing spaces"]


class TestWriteNewPromptFile:
    """Tests for _write_new_prompt_file function."""

    def test_write_new_prompt_file(self):
        """Test writing a new prompt file."""
        with TemporaryDirectory() as tmp_dir:
            pool_dir = Path(tmp_dir)
            text = "This is a test prompt"
            n_candidates = 5

            # Mock uuid to make test deterministic
            with patch.object(uuid, "uuid4") as mock_uuid:
                mock_uuid.return_value.hex = "abcd1234" * 4

                result_path = _write_new_prompt_file(text, pool_dir, n_candidates)

                expected_name = "prompt_5_abcd1234.txt"
                assert result_path.name == expected_name
                assert result_path.parent == pool_dir
                assert result_path.read_text(encoding="utf-8") == text

    def test_write_creates_unique_names(self):
        """Test that multiple calls create files with unique names."""
        with TemporaryDirectory() as tmp_dir:
            pool_dir = Path(tmp_dir)
            text = "Test prompt"

            path1 = _write_new_prompt_file(text, pool_dir, 1)
            path2 = _write_new_prompt_file(text, pool_dir, 1)

            assert path1 != path2
            assert path1.exists()
            assert path2.exists()


class TestReadPromptText:
    """Tests for _read_prompt_text function."""

    @patch("pydantic_ai_optimizers.optimizer.textprompts")
    def test_read_prompt_text(self, mock_textprompts):
        """Test reading prompt text from file."""
        mock_prompt = Mock()
        mock_prompt.__str__ = Mock(return_value="Test prompt content")
        mock_textprompts.load_prompt.return_value = mock_prompt

        path = Path("/fake/path/prompt.txt")
        result = _read_prompt_text(path)

        mock_textprompts.load_prompt.assert_called_once_with(str(path))
        assert result == "Test prompt content"
