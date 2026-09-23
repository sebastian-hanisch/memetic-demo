"""Vehikel (Reproduzierbarkeit, bitidentisch zu hill-climbing-demo/genetic-algorithm-demo) und Auswertung
(Brute-Force, Sweep, Kopfexperiment, p_LS-Experiment) - schnelle Parameter über Funktionsargumente."""

import numpy as np
import pytest

import mem_algorithm as A
import mem_constants as C
import mem_evaluation as E
import mem_scenario as S


def test_generate_is_reproducible_and_shaped():
    a = S.generate(20, 0, seed=7)
    b = S.generate(20, 0, seed=7)
    assert np.array_equal(a.xy, b.xy)
    assert a.xy.shape == (21, 2)


def test_generate_matches_hill_climbing_demo_bit_for_bit_on_the_comparison_instance():
    """Die kleine Vergleichsinstanz (n=8, Seed 19) muss xy bitidentisch zu hill-climbing-demo/genetic-algorithm-demo
    liefern - reproduziert deren generate()/generate_perm() hier lokal (kein Cross-Repo-Import, wie überall im Portfolio)."""
    def ref_generate(n, cluster_share, seed):
        rng = np.random.default_rng(seed)
        n_grouped = int(round(n * cluster_share / 100))
        uniform = rng.random((n - n_grouped, 2)) * C.AREA
        centres = C.CLUSTER_MARGIN + rng.random((C.N_CLUSTERS, 2)) * (C.AREA - 2 * C.CLUSTER_MARGIN)
        which = rng.integers(0, C.N_CLUSTERS, size=n_grouped)
        grouped = np.clip(centres[which] + rng.normal(0.0, C.CLUSTER_SIGMA, size=(n_grouped, 2)), 0.0, C.AREA)
        depot = np.array([[C.AREA / 2, C.AREA / 2]])
        return np.vstack([depot, uniform, grouped])

    xy_ref = ref_generate(C.COMPARISON_N, 0, C.COMPARISON_VEHICLE_SEED)
    inst = S.generate(C.COMPARISON_N, 0, C.COMPARISON_VEHICLE_SEED)
    assert np.array_equal(inst.xy, xy_ref)


# --- Auswertung --------------------------------------------------------------------------------------------------------------------------------


def test_run_returns_expected_shapes():
    s = E.Settings(n=10, gens=8, pop=10)
    r = E.run(s, keep_history=True)
    assert r.best_history.shape == (9,)
    assert len(r.generations) == 9


def test_brute_force_matches_manual_computation_on_a_tiny_instance():
    inst, D = E.instance(5, 0, 3)
    optimum = E.brute_force(D)
    from itertools import permutations
    best = min(A.tour_length((0,) + p, D) for p in permutations(range(1, 6)))
    assert optimum == pytest.approx(best)


def test_analyse_computes_gap_only_within_brute_force_range():
    a_small = E.analyse(E.Settings(n=C.COMPARISON_N, seed=C.COMPARISON_VEHICLE_SEED, pop=C.COMPARISON_POP, gens=C.COMPARISON_GENS, p_ls=1.0))
    assert np.isfinite(a_small.brute_force_optimum)
    assert np.isfinite(a_small.gap)

    a_large = E.analyse(E.Settings(n=50, gens=5, pop=10))
    assert not np.isfinite(a_large.brute_force_optimum)
    assert not np.isfinite(a_large.gap)


def test_gap_is_never_negative():
    a = E.analyse(E.Settings(n=C.COMPARISON_N, seed=C.COMPARISON_VEHICLE_SEED, pop=C.COMPARISON_POP, gens=C.COMPARISON_GENS, p_ls=1.0))
    assert a.gap >= 0.0


def test_sweep_smoke():
    rows = E.sweep("pop", base=E.Settings(n=C.COMPARISON_N, seed=C.COMPARISON_VEHICLE_SEED, pop=10, gens=6, p_ls=0.0), values=(10, 20))
    assert len(rows) == 2
    assert all(np.isfinite(r["gap"]) for r in rows)


def test_head_experiment_smoke_small():
    report = E.head_experiment(n=10, seed=1, pop=8, gens=6, seeds=(1, 2))
    assert set(report) == {"ga", "memetic"}
    assert report["ga"]["ls_evaluations_mean"] == 0.0
    assert report["memetic"]["best_median"] > 0.0


def test_p_ls_experiment_smoke_small():
    rows = E.p_ls_experiment(n=10, seed=1, values=(0.0, 1.0), seeds=(1, 2), pop=8, gens=6)
    assert len(rows) == 2
    assert rows[0]["ls_evaluations_mean"] == 0.0
    assert rows[1]["ls_evaluations_mean"] > 0.0
