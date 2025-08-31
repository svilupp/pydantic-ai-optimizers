"""Tests for core optimizer methods."""

import random
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from pydantic_ai_optimizers.optimizer import CaseEval, Optimizer


class TestOptimizerCoreMethod:
    """Tests for core optimizer methods."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock dataset
        self.mock_dataset = Mock()
        self.mock_dataset.cases = [Mock() for _ in range(10)]

        # Create a mock run_case function
        async def mock_run_case(prompt_file: str, user_input):
            return "mock_output"

        self.mock_run_case = mock_run_case

        # Create optimizer with minimal setup
        with TemporaryDirectory() as tmp_dir:
            self.pool_dir = Path(tmp_dir)
            # Mock the reflection agent to avoid needing API keys
            mock_reflection_agent = Mock()
            with patch(
                "pydantic_ai_optimizers.optimizer.make_reflection_agent",
                return_value=mock_reflection_agent,
            ):
                self.optimizer = Optimizer(
                    dataset=self.mock_dataset,
                    run_case=self.mock_run_case,
                    pool_dir=self.pool_dir,
                    minibatch_size=3,
                    max_pool_size=5,
                    seed=42,
                )

    def test_best_candidate_index_single_candidate(self):
        """Test _best_candidate_index with single candidate."""
        # Set up eval data
        self.optimizer.eval_rows = {0: [CaseEval(score=0.8, reasons=[], output="test")]}

        result = self.optimizer._best_candidate_index()
        assert result == 0

    def test_best_candidate_index_multiple_candidates(self):
        """Test _best_candidate_index with multiple candidates."""
        # Set up eval data with different average scores
        self.optimizer.eval_rows = {
            0: [
                CaseEval(score=0.5, reasons=[], output="test1"),
                CaseEval(score=0.7, reasons=[], output="test2"),
            ],  # avg = 0.6
            1: [
                CaseEval(score=0.8, reasons=[], output="test3"),
                CaseEval(score=0.9, reasons=[], output="test4"),
            ],  # avg = 0.85
            2: [
                CaseEval(score=0.3, reasons=[], output="test5"),
                CaseEval(score=0.4, reasons=[], output="test6"),
            ],  # avg = 0.35
        }

        result = self.optimizer._best_candidate_index()
        assert result == 1  # Candidate 1 has highest average score (0.85)

    def test_pick_minibatch_indices(self):
        """Test _pick_minibatch_indices returns correct number of indices."""
        # Set random seed for reproducibility
        random.seed(42)

        indices = self.optimizer._pick_minibatch_indices()

        assert len(indices) == 3  # minibatch_size
        assert all(0 <= idx < 10 for idx in indices)  # All indices valid for dataset size
        assert len(set(indices)) == len(indices)  # All indices unique

    def test_pick_minibatch_indices_larger_than_dataset(self):
        """Test _pick_minibatch_indices when minibatch size is larger than dataset."""
        # Create smaller dataset
        self.optimizer.dataset.cases = [Mock() for _ in range(2)]

        indices = self.optimizer._pick_minibatch_indices()

        assert len(indices) == 2  # Should return min(minibatch_size, dataset_size)
        assert indices == [0, 1] or indices == [1, 0]  # Should include all available indices

    def test_sample_candidate_pareto_single_candidate(self):
        """Test _sample_candidate_pareto with single candidate."""
        self.optimizer.eval_rows = {0: [CaseEval(score=0.8, reasons=[], output="test")]}

        result = self.optimizer._sample_candidate_pareto()
        assert result == 0

    def test_sample_candidate_pareto_multiple_candidates(self):
        """Test _sample_candidate_pareto with multiple candidates."""
        # Set up candidates with different per-case performance
        self.optimizer.eval_rows = {
            0: [
                CaseEval(score=1.0, reasons=[], output="test1"),
                CaseEval(score=0.5, reasons=[], output="test2"),
            ],  # Wins case 0
            1: [
                CaseEval(score=0.8, reasons=[], output="test3"),
                CaseEval(score=1.0, reasons=[], output="test4"),
            ],  # Wins case 1
        }

        # Since this uses weighted random sampling, test multiple times
        results = []
        for _ in range(100):
            results.append(self.optimizer._sample_candidate_pareto())

        # Both candidates should be selected sometimes (weighted by wins)
        assert 0 in results
        assert 1 in results
        assert all(r in [0, 1] for r in results)

    def test_prune_pool(self):
        """Test _prune_pool keeps top performers."""
        # Set up 4 candidates (max_pool_size = 5, so keep_n = 2)
        from pydantic_ai_optimizers.optimizer import Candidate

        self.optimizer.candidates = [
            Candidate(prompt_path=Path("prompt1.txt")),
            Candidate(prompt_path=Path("prompt2.txt")),
            Candidate(prompt_path=Path("prompt3.txt")),
            Candidate(prompt_path=Path("prompt4.txt")),
        ]

        self.optimizer.eval_rows = {
            0: [CaseEval(score=0.3, reasons=[], output="test1")],  # worst
            1: [CaseEval(score=0.9, reasons=[], output="test2")],  # best
            2: [CaseEval(score=0.7, reasons=[], output="test3")],  # second best
            3: [CaseEval(score=0.5, reasons=[], output="test4")],  # third
        }

        self.optimizer._prune_pool()

        # Should keep 2 candidates (max_pool_size // 2)
        assert len(self.optimizer.candidates) == 2
        assert len(self.optimizer.eval_rows) == 2

        # Should keep the best performers (indices 1 and 2 originally)
        # After pruning, they become indices 0 and 1
        remaining_scores = [self.optimizer.eval_rows[i][0].score for i in self.optimizer.eval_rows]
        assert set(remaining_scores) == {0.9, 0.7}


class TestOptimizerSamplingLogic:
    """Tests for optimizer sampling and weighting logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_dataset = Mock()
        self.mock_dataset.cases = [Mock() for _ in range(4)]

        async def mock_run_case(prompt_file: str, user_input):
            return "mock_output"

        with TemporaryDirectory() as tmp_dir:
            # Mock the reflection agent to avoid needing API keys
            mock_reflection_agent = Mock()
            with patch(
                "pydantic_ai_optimizers.optimizer.make_reflection_agent",
                return_value=mock_reflection_agent,
            ):
                self.optimizer = Optimizer(
                    dataset=self.mock_dataset,
                    run_case=mock_run_case,
                    pool_dir=Path(tmp_dir),
                    seed=42,
                )

    def test_pareto_sampling_weights_winners(self):
        """Test that Pareto sampling gives higher weight to case winners."""
        # Set up 3 candidates where candidate 2 wins most cases
        self.optimizer.eval_rows = {
            0: [
                CaseEval(score=0.5, reasons=[], output="test"),  # loses case 0
                CaseEval(score=0.6, reasons=[], output="test"),  # loses case 1
                CaseEval(score=0.8, reasons=[], output="test"),  # loses case 2
                CaseEval(score=0.9, reasons=[], output="test"),  # wins case 3
            ],
            1: [
                CaseEval(score=0.7, reasons=[], output="test"),  # loses case 0
                CaseEval(score=0.8, reasons=[], output="test"),  # wins case 1
                CaseEval(score=0.6, reasons=[], output="test"),  # loses case 2
                CaseEval(score=0.5, reasons=[], output="test"),  # loses case 3
            ],
            2: [
                CaseEval(score=1.0, reasons=[], output="test"),  # wins case 0
                CaseEval(score=0.7, reasons=[], output="test"),  # loses case 1
                CaseEval(score=0.9, reasons=[], output="test"),  # wins case 2
                CaseEval(score=0.6, reasons=[], output="test"),  # loses case 3
            ],
        }

        # Count selections over many trials
        selections = {}
        for _ in range(1000):
            selected = self.optimizer._sample_candidate_pareto()
            selections[selected] = selections.get(selected, 0) + 1

        # Candidate 2 should be selected most often (wins 2 cases)
        # Candidates 0 and 1 each win 1 case, candidate 2 wins 2 cases
        assert selections[2] > selections[0]
        assert selections[2] > selections[1]

    def test_pareto_sampling_ties_handled(self):
        """Test Pareto sampling when candidates tie on cases."""
        # Set up candidates that tie on all cases
        self.optimizer.eval_rows = {
            0: [
                CaseEval(score=0.8, reasons=[], output="test"),
                CaseEval(score=0.6, reasons=[], output="test"),
            ],
            1: [
                CaseEval(score=0.8, reasons=[], output="test"),
                CaseEval(score=0.6, reasons=[], output="test"),
            ],
        }

        # Should still select candidates (both get equal weight for ties)
        selections = [self.optimizer._sample_candidate_pareto() for _ in range(100)]
        assert all(s in [0, 1] for s in selections)

        # Both should be selected roughly equally since they tie
        selections_0 = selections.count(0)
        selections_1 = selections.count(1)
        ratio = max(selections_0, selections_1) / max(min(selections_0, selections_1), 1)
        assert ratio < 3  # Shouldn't be too skewed for equal performers
