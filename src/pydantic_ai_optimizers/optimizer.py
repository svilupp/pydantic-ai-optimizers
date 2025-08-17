from __future__ import annotations

import random
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import textprompts
from loguru import logger

from .dataset import Dataset, ReportCase

RunCase = Callable[[str, Any], Awaitable[Any]]


@dataclass(frozen=True)
class Candidate:
    prompt_path: Path
    parent_index: int | None = None
    note: str = ""


@dataclass(frozen=True)
class CaseEval:
    score: float
    reasons: list[str]
    output: Any


@dataclass(frozen=True)
class FailedMutation:
    prompt_text: str
    parent_index: int
    failure_reasons: list[str]
    average_score: float
    attempt_number: int


class Optimizer:
    """
    A filesystem-based prompt optimization engine that iteratively improves agent instructions
    through reflection and evaluation feedback.

    This optimizer uses a population-based approach where:
    1. A seed prompt generates a candidate pool
    2. Reflection agents analyze evaluation feedback to create new prompt variants
    3. Minibatch gating filters improvements based on performance
    4. Failed attempts are tracked to guide future optimizations

    Features:
    - Multi-evaluator feedback aggregation (only failures collected)
    - Failed mutation memory to detect optimization patterns
    - Pareto-efficient candidate selection
    - Configurable pool size and validation budgets
    - Filesystem persistence for prompt variants

    The system is designed to work with any evaluation framework that provides
    per-case scores and reasons via a simple interface.
    """

    def __init__(
        self,
        dataset: Dataset,
        run_case: RunCase,
        reflection_agent,
        pool_dir: str | Path = "prompt_pool",
        minibatch_size: int = 4,
        max_pool_size: int = 16,
        seed: int = 0,
        keep_failed_mutations: bool = False,
    ):
        self.dataset = dataset
        self.run_case = run_case
        self.reflection_agent = reflection_agent
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.minibatch_size = minibatch_size
        self.max_pool_size = max_pool_size
        self.keep_failed_mutations = keep_failed_mutations
        random.seed(seed)

        self.candidates: list[Candidate] = []
        self.eval_rows: dict[int, list[CaseEval]] = {}
        self.failed_mutations: list[FailedMutation] = []
        self.mutation_attempt_count: int = 0

    async def optimize(self, seed_prompt_file: Path, full_validation_budget: int = 20) -> Candidate:
        logger.info(f"🚀 Starting optimization with seed: {seed_prompt_file}")
        logger.info(f"📋 Budget: {full_validation_budget} validations")

        self._add_candidate(seed_prompt_file, parent=None, note="seed")
        validations = 1

        while validations < full_validation_budget:
            logger.info(f"\n🔄 OPTIMIZATION ITERATION {validations}/{full_validation_budget}")

            parent_idx = self._sample_candidate_pareto()
            parent_candidate = self.candidates[parent_idx]
            parent_avg = mean(r.score for r in self.eval_rows[parent_idx])
            logger.info(
                f"🎯 Selected parent {parent_idx}: {parent_candidate.prompt_path.name} (score: {parent_avg:.3f})"
            )

            mb_ids = self._pick_minibatch_indices()
            logger.info(f"🎲 Testing on minibatch: cases {mb_ids}")

            logger.info("🤔 Generating reflection...")
            new_prompt_path = await self._reflect_to_new_file(parent_idx, mb_ids)
            logger.info(f"💡 Generated new prompt: {new_prompt_path.name}")

            logger.info("🎯 Testing new prompt on minibatch...")
            gate_result = await self._minibatch_gate(parent_idx, new_prompt_path, mb_ids)
            if gate_result["passed"]:
                logger.info("✅ MINIBATCH PASSED! Adding to candidate pool")
                self._add_candidate(
                    new_prompt_path, parent=parent_idx, note="reflected+passed_minibatch"
                )
                # Clear failed mutations on successful improvement
                self.failed_mutations.clear()
                validations += 1
                if len(self.candidates) > self.max_pool_size:
                    logger.info(f"✂️ Pruning pool (current size: {len(self.candidates)})")
                    self._prune_pool()
            else:
                logger.warning("❌ MINIBATCH FAILED! Recording failed mutation")
                logger.warning(f"   Child score: {gate_result['child_avg']:.3f}")
                logger.warning(f"   Parent score: {gate_result['parent_avg']:.3f}")
                logger.warning(f"   Failure reasons: {gate_result['failure_reasons']}")

                # Record the failed mutation
                self.mutation_attempt_count += 1
                prompt_text = _read_prompt_text(new_prompt_path)
                failed_mutation = FailedMutation(
                    prompt_text=prompt_text,
                    parent_index=parent_idx,
                    failure_reasons=gate_result["failure_reasons"],
                    average_score=gate_result["child_avg"],
                    attempt_number=self.mutation_attempt_count,
                )
                self.failed_mutations.append(failed_mutation)
                logger.info(f"📝 Total failed mutations: {len(self.failed_mutations)}")

                if not self.keep_failed_mutations:
                    new_prompt_path.unlink(missing_ok=True)
                    logger.debug(f"🗑️ Deleted failed prompt file: {new_prompt_path.name}")

        best_idx = self._best_candidate_index()
        best_candidate = self.candidates[best_idx]
        best_scores = [r.score for r in self.eval_rows[best_idx]]
        best_avg = mean(best_scores)

        logger.info("\n🏆 OPTIMIZATION COMPLETE!")
        logger.info("📊 Final Results:")
        logger.info(f"   Total candidates evaluated: {len(self.candidates)}")
        logger.info(f"   Failed mutations: {len(self.failed_mutations)}")
        logger.info(f"   Best candidate: {best_candidate.prompt_path.name}")
        logger.info(f"   Best average score: {best_avg:.3f}")
        logger.info(f"   Best individual scores: {[f'{s:.3f}' for s in best_scores]}")
        logger.info(f"   Cases passed: {sum(1 for s in best_scores if s > 0.5)}/{len(best_scores)}")

        # Show score progression
        if len(self.candidates) > 1:
            seed_scores = [r.score for r in self.eval_rows[0]]
            seed_avg = mean(seed_scores)
            total_improvement = best_avg - seed_avg
            logger.info(
                f"🚀 Total improvement: {total_improvement:+.3f} ({seed_avg:.3f} → {best_avg:.3f})"
            )

        return best_candidate

    # ---------- helpers ----------

    def _add_candidate(self, prompt_path: Path, parent: int | None, note: str) -> int:
        self.candidates.append(Candidate(prompt_path=prompt_path, parent_index=parent, note=note))
        idx = len(self.candidates) - 1

        logger.info(f"🔍 Evaluating candidate {idx}: {prompt_path.name} ({note})")
        self.eval_rows[idx] = self._evaluate_full_sync(prompt_path)

        # Calculate and log performance metrics
        scores = [r.score for r in self.eval_rows[idx]]
        avg_score = mean(scores)
        logger.info(f"📊 Candidate {idx} results:")
        logger.info(f"   Average Score: {avg_score:.3f}")
        logger.info(f"   Individual Scores: {[f'{s:.3f}' for s in scores]}")
        logger.info(f"   Cases Passed: {sum(1 for s in scores if s > 0.5)}/{len(scores)}")

        # Compare with parent if available
        if parent is not None and parent in self.eval_rows:
            parent_scores = [r.score for r in self.eval_rows[parent]]
            parent_avg = mean(parent_scores)
            improvement = avg_score - parent_avg
            logger.info(
                f"🔄 vs Parent {parent}: {improvement:+.3f} ({parent_avg:.3f} → {avg_score:.3f})"
            )
            if improvement > 0:
                logger.info(f"✅ IMPROVEMENT! Better by {improvement:.3f}")
            else:
                logger.warning(f"❌ REGRESSION! Worse by {abs(improvement):.3f}")

        return idx

    def _best_candidate_index(self) -> int:
        return max(self.eval_rows, key=lambda i: mean(r.score for r in self.eval_rows[i]))

    def _prune_pool(self) -> None:
        keep_n = max(1, self.max_pool_size // 2)
        top = sorted(
            self.eval_rows, key=lambda i: mean(r.score for r in self.eval_rows[i]), reverse=True
        )[:keep_n]
        idx_map = {old: new for new, old in enumerate(top)}
        self.candidates = [self.candidates[i] for i in top]
        self.eval_rows = {idx_map[i]: self.eval_rows[i] for i in top}

    def _pick_minibatch_indices(self) -> list[int]:
        n = len(self.dataset.cases)
        k = min(self.minibatch_size, n)
        return random.sample(range(n), k)

    def _sample_candidate_pareto(self) -> int:
        if len(self.candidates) == 1:
            return 0
        cand_ids = sorted(self.eval_rows)
        cols = list(
            zip(*[[row.score for row in self.eval_rows[cid]] for cid in cand_ids], strict=False)
        )
        counts = [0] * len(cand_ids)
        for col in cols:
            m = max(col)
            for j, s in enumerate(col):
                counts[j] += s == m
        j = random.choices(range(len(cand_ids)), weights=[c or 1 for c in counts], k=1)[0]
        return cand_ids[j]

    async def _minibatch_gate(
        self, parent_idx: int, child_prompt: Path, case_ids: Sequence[int]
    ) -> dict[str, Any]:
        parent_avg = mean(self.eval_rows[parent_idx][i].score for i in case_ids)
        sub_cases = [self.dataset.cases[i] for i in case_ids]

        logger.info(f"⚖️ MINIBATCH GATE: Comparing child vs parent on {len(case_ids)} cases")
        logger.info(f"   Parent {parent_idx} score on minibatch: {parent_avg:.3f}")

        def eval_subset(prompt: Path) -> dict[str, Any]:
            def task_fn(user_text: str):
                return self.run_case(str(prompt), user_text)

            # build a mini dataset on the fly
            from evals.dataset_lite import Dataset

            mini = Dataset(cases=sub_cases, evaluators=self.dataset.evaluators)
            rep = mini.evaluate_sync(task_fn)
            scores = [_score_from_report_case(rc) for rc in rep.cases]
            all_reasons = []
            for rc in rep.cases:
                all_reasons.extend(_reason_from_report_case(rc))
            avg_score = mean(scores) if scores else 0.0
            return {"avg_score": avg_score, "reasons": all_reasons}

        child_result = eval_subset(child_prompt)
        child_avg = child_result["avg_score"]
        passed = child_avg > parent_avg

        logger.info(f"   Child score on minibatch: {child_avg:.3f}")
        improvement = child_avg - parent_avg
        logger.info(f"   Improvement: {improvement:+.3f}")

        if passed:
            logger.info(f"✅ GATE PASSED! Child beats parent by {improvement:.3f}")
        else:
            logger.warning(f"❌ GATE FAILED! Child worse than parent by {abs(improvement):.3f}")
            if child_result["reasons"]:
                logger.warning(
                    f"   Failure reasons: {child_result['reasons'][:3]}..."
                )  # Show first 3 reasons

        return {
            "passed": passed,
            "child_avg": child_avg,
            "parent_avg": parent_avg,
            "failure_reasons": child_result["reasons"] if not passed else [],
        }

    async def _reflect_to_new_file(self, parent_idx: int, case_ids: Sequence[int]) -> Path:
        parent = self.candidates[parent_idx]
        current_text = _read_prompt_text(parent.prompt_path)

        examples = []
        for i in case_ids:
            ce = self.eval_rows[parent_idx][i]
            inputs = self.dataset.cases[i].user
            feedback = (
                "\n".join(f"- {reason}" for reason in ce.reasons)
                if ce.reasons
                else "No specific feedback"
            )
            examples.append(f"INPUT: {inputs}\nOUTPUT: {ce.output}\nFEEDBACK:\n{feedback}")

        # Add failed mutations context if available
        failed_context = ""
        if self.failed_mutations:
            total_failures = len(self.failed_mutations)
            last_failures = self.failed_mutations[-10:]  # Show last 10 failures

            failed_summaries = []
            for fm in last_failures:
                # Format reasons more clearly
                reasons_text = (
                    fm.failure_reasons[:3] if fm.failure_reasons else ["No specific reasons"]
                )
                failure_summary = f"  • Attempt #{fm.attempt_number}: Score {fm.average_score:.3f}\n    Failure reasons: {'; '.join(reasons_text)}"
                failed_summaries.append(failure_summary)

            failure_note = ""
            if total_failures >= 10:
                failure_note = f"\n⚠️  WARNING: {total_failures} total failed attempts indicates significant optimization challenges. Consider major structural changes."
            elif total_failures >= 5:
                failure_note = f"\n⚠️  {total_failures} failed attempts suggest current approach may need revision."

            failed_context = (
                f"\n\nPREVIOUS FAILED ATTEMPTS:\n"
                f"Total failures: {total_failures} | Showing last {len(last_failures)} attempts\n"
                f"<<<\n" + "\n".join(failed_summaries) + f"\n>>>{failure_note}\n\n"
                f"Note: The above attempts did not improve performance. Consider different approaches or structural changes.\n"
            )

        user_msg = (
            "Rewrite the following agent INSTRUCTIONS to improve performance.\n"
            "Return ONLY the new instructions.\n\n"
            "CURRENT INSTRUCTIONS:\n<<<\n" + current_text + "\n>>>\n\n"
            "EXAMPLES (INPUT / CURRENT OUTPUT / EVALUATOR FEEDBACK):\n<<<\n"
            + "\n---\n".join(examples)
            + "\n>>>"
            + failed_context
        )
        r = await self.reflection_agent.run(user_msg)
        new_text = (r.output or "").strip()
        return _write_new_prompt_file(new_text, self.pool_dir, len(self.candidates))

    def _evaluate_full_sync(self, prompt_path: Path) -> list[CaseEval]:
        def task_fn(user_text: str):
            return self.run_case(str(prompt_path), user_text)

        report = self.dataset.evaluate_sync(task_fn)
        rows: list[CaseEval] = []
        for rc in report.cases:
            score = _score_from_report_case(rc)
            reasons = _reason_from_report_case(rc)
            rows.append(CaseEval(score=score, reasons=reasons, output=rc.output))
        return rows


# ---- small helpers ----


def _score_from_report_case(rc: ReportCase) -> float:
    # Prefer numeric scores if present, else pass-rate of assertions
    if rc.scores:
        return float(sum(float(v.value) for v in rc.scores.values()) / max(1, len(rc.scores)))
    if rc.assertions:
        bools = [1.0 if bool(v.value) else 0.0 for v in rc.assertions.values()]
        return float(sum(bools) / len(bools))
    return 0.0


def _reason_from_report_case(rc: ReportCase) -> list[str]:
    # Collect reasons only from FAILED evaluators (not perfect scores)
    reasons = []

    # For assertions: collect reasons when value is False (failed)
    for v in rc.assertions.values():
        if v.reason and v.reason.strip() and v.value is False:
            reasons.append(v.reason.strip())

    # For scores: collect reasons when value is not 1.0 (not perfect)
    for v in rc.scores.values():
        if v.reason and v.reason.strip() and float(v.value) != 1.0:
            reasons.append(v.reason.strip())

    return reasons


def _read_prompt_text(path: Path) -> str:
    return str(textprompts.load_prompt(str(path)))


def _write_new_prompt_file(text: str, pool_dir: Path, n_candidates: int) -> Path:
    name = f"prompt_{n_candidates}_{uuid.uuid4().hex[:8]}.txt"
    path = pool_dir / name
    path.write_text(text, encoding="utf-8")
    return path
