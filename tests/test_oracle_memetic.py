"""Orakel-Tests (unabhängiger Rechenweg): 2-opt-Delta gegen volle Neubewertung der umgedrehten Tour, gültige Paare gegen
die Definition "nicht benachbarte Kanten", Politur (Zugbudget, Bewertungszähler, lokales Optimum) gegen eine
Schleifenfassung, Order-Crossover gegen eine Referenz, Lauf-Invarianten."""

import numpy as np

import mem_algorithm as A

EPS = 1e-9


def _len(t, D):
    n = len(t)
    return sum(D[t[i]][t[(i + 1) % n]] for i in range(n))


def _reversed(t, i, j):
    return list(t[:i + 1]) + list(t[i + 1:j + 1])[::-1] + list(t[j + 1:])


def _ref_first_improving(t, D):
    n, cur, evals = len(t), _len(t, D), 0
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            evals += 1
            new = _reversed(t, i, j)
            if _len(new, D) - cur < -EPS:
                return new, evals
    return None, evals


def _ref_polish(t, D, max_moves):
    t, ev = list(t), 0
    for _ in range(max_moves):
        new, e = _ref_first_improving(t, D)
        ev += e
        if new is None:
            break
        t = new
    return t, ev


def test_2opt_delta_valid_pairs_and_polish_match_reference():
    rng = np.random.default_rng(1)
    for k in range(40):
        n = int(rng.integers(4, 12))
        xy = rng.random((n, 2)) * 100
        if k % 5 == 0:
            xy = np.round(xy / 25) * 25            # Gleichstände und doppelte Punkte
        D = A.dist_matrix(xy)
        t = rng.permutation(n)
        delta, ok = A._delta_2opt(t, D)
        base = _len(list(t), D)
        assert ok.sum() == n * (n - 3) // 2
        for i in range(n):
            for j in range(n):
                valid = i < j and j != i + 1 and not (i == 0 and j == n - 1)
                assert bool(ok[i, j]) == valid
                if valid:
                    new = _reversed(list(t), i, j)
                    assert abs(_len(new, D) - base - delta[i, j]) < 1e-9
                    assert A._apply_2opt(t, i, j).tolist() == new
        for budget in (0, 1, 3, 30):
            pt, pe = A.polish(t, D, budget)
            rt, re_ = _ref_polish(list(t), D, budget)
            assert pt.tolist() == rt and pe == re_
            assert _len(list(pt), D) <= base + 1e-9
        pt, _ = A.polish(t, D, 1000)
        assert _ref_first_improving(list(pt), D)[0] is None


def _ref_ox(p1, p2, i, j):
    rest = iter(g for g in p2 if g not in p1[i:j + 1])
    return [p1[k] if i <= k <= j else next(rest) for k in range(len(p1))]


def test_order_crossover_matches_reference():
    for s in range(100):
        n = int(np.random.default_rng(s).integers(2, 12))
        p1 = np.random.default_rng(s + 1000).permutation(n)
        p2 = np.random.default_rng(s + 2000).permutation(n)
        child = A.order_crossover(p1, p2, np.random.default_rng(s))
        i, j = sorted(np.random.default_rng(s).integers(0, n, size=2))
        assert child.tolist() == _ref_ox(p1.tolist(), p2.tolist(), int(i), int(j))


def test_run_memetic_invariants_against_independent_recomputation():
    for s in range(12):
        n = 8 + s % 4
        D = A.dist_matrix(np.random.default_rng(s).random((n, 2)) * 100)
        r = A.run_memetic(D, 10, 6, 0.9, 0.3, 1, 3, 0.5, 30, s, keep_history=True)
        for g in r.generations:
            for t, f in zip(g.population, g.fitness):
                assert sorted(t.tolist()) == list(range(n))
                assert abs(_len(t.tolist(), D) - f) < 1e-8
        assert abs(_len(r.best_individual.tolist(), D) - r.best_fitness) < 1e-8
        assert np.all(np.diff(r.best_history) <= 1e-9)           # Elitismus >= 1
        assert np.all(np.diff(r.ls_evaluations_history) >= 0)
        assert r.ls_evaluations == r.ls_evaluations_history[-1]
