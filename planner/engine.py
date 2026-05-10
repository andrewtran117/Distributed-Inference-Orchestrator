"""Enumerate all valid layer assignments and pick the best."""

from itertools import combinations, permutations

from pydantic import BaseModel

from planner.presets import ModelConfig
from planner.scorer import PlanScore, score_assignment
from registry.models import MachineRecord


class SingleMachineComparison(BaseModel):
    best_single_machine: str
    predicted_tok_per_sec: float
    verdict: str


class PlanResult(BaseModel):
    plan: PlanScore
    single_machine_comparison: SingleMachineComparison


def _generate_partitions(num_layers: int, num_machines: int):
    """
    Generate all ways to split num_layers into num_machines contiguous chunks.
    Each chunk has at least 1 layer.
    Yields lists of layer counts, e.g. [16, 16] for 32 layers across 2 machines.
    """
    # Choose (num_machines - 1) split points from positions 1..num_layers-1
    for splits in combinations(range(1, num_layers), num_machines - 1):
        counts = []
        prev = 0
        for s in splits:
            counts.append(s - prev)
            prev = s
        counts.append(num_layers - prev)
        yield counts


def _best_single_machine(
    machines: list[MachineRecord], model: ModelConfig
) -> tuple[PlanScore | None, MachineRecord | None]:
    """Find the best single-machine plan."""
    best_score: PlanScore | None = None
    best_machine: MachineRecord | None = None

    for machine in machines:
        result = score_assignment([machine], [model.num_layers], model)
        if result and (best_score is None or result.predicted_tok_per_sec > best_score.predicted_tok_per_sec):
            best_score = result
            best_machine = machine

    return best_score, best_machine


def plan(
    machines: list[MachineRecord], model: ModelConfig
) -> PlanResult:
    """
    Find the optimal layer assignment for the given machines and model.
    Tries all machine counts (1..M), all permutations, all partitions.
    Returns the best plan with a single-machine comparison.
    """
    if not machines:
        raise ValueError("No machines available")

    total_free = sum(m.specs.memory_free_gb for m in machines)
    if total_free < model.total_size_gb:
        raise ValueError(
            f"Insufficient total memory: {total_free:.1f} GB free across "
            f"{len(machines)} machines, model requires {model.total_size_gb} GB"
        )

    best: PlanScore | None = None

    # Try all machine counts: 1, 2, ..., len(machines)
    for m in range(1, len(machines) + 1):
        # Try all subsets of size m? No — try all permutations of all m-sized subsets.
        # For simplicity, permutations of the full set handles subsets when m < len(machines)
        # by iterating permutations of m machines chosen from the pool.
        seen_perms: set[tuple[str, ...]] = set()

        for perm in permutations(machines, m):
            perm_key = tuple(p.machine_id for p in perm)
            if perm_key in seen_perms:
                continue
            seen_perms.add(perm_key)

            for layer_counts in _generate_partitions(model.num_layers, m):
                result = score_assignment(list(perm), layer_counts, model)
                if result and (best is None or result.predicted_tok_per_sec > best.predicted_tok_per_sec):
                    best = result

    if best is None:
        raise ValueError("No valid assignment found — model may not fit in available memory")

    # Single machine comparison
    single_score, single_machine = _best_single_machine(machines, model)

    if single_score and single_machine:
        if single_score.predicted_tok_per_sec >= best.predicted_tok_per_sec:
            verdict = (
                "Single machine is faster. Distributed split adds network "
                "overhead without benefit since the model fits on one machine."
            )
        else:
            speedup = best.predicted_tok_per_sec / single_score.predicted_tok_per_sec
            verdict = (
                f"Distributed plan is {speedup:.1f}x faster than the best single machine."
            )
        comparison = SingleMachineComparison(
            best_single_machine=single_machine.machine_id,
            predicted_tok_per_sec=single_score.predicted_tok_per_sec,
            verdict=verdict,
        )
    else:
        comparison = SingleMachineComparison(
            best_single_machine="none",
            predicted_tok_per_sec=0.0,
            verdict="Model does not fit on any single machine. Distributed split is required.",
        )

    return PlanResult(plan=best, single_machine_comparison=comparison)
