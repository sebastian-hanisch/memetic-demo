"""Handrechnungen für Crossover/Mutation/Turnier (wortgleiche Mechanik zu genetic-algorithm-demo) und die 2-opt-
Politur (schlanke Neuimplementierung des Delta-Musters aus hill-climbing-demo), plus Struktur-/Brute-Force-Tests
der Hauptschleife."""

from itertools import permutations

import numpy as np
import pytest

import mem_algorithm as A


class FixedRNG:
    """Minimaler Ersatz für np.random.default_rng mit festen Rückgabewerten - für Handrechnungs-Tests."""

    def __init__(self, integers_return=None, random_return=None, choice_return=None):
        self._integers = integers_return
        self._random = random_return
        self._choice = choice_return

    def integers(self, lo, hi, size=None):
        return np.array(self._integers)

    def random(self, size=None):
        return self._random if size is None else np.array(self._random)

    def choice(self, n, size, replace=True):
        return np.array(self._choice)


# --- Distanz / Tourlänge ------------------------------------------------------------------------------------------------------------------------


def test_dist_matrix_matches_hand_calculation():
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    D = A.dist_matrix(xy)
    assert D[0, 1] == pytest.approx(3.0)
    assert D[1, 2] == pytest.approx(4.0)
    assert D[0, 2] == pytest.approx(5.0)


def test_tour_length_matches_hand_calculation():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    D = A.dist_matrix(xy)
    assert A.tour_length([0, 1, 2, 3], D) == pytest.approx(4.0)      # Einheitsquadrat, Umfang 4


# --- Crossover / Mutation / Turnier (Handrechnung mit fester RNG) -----------------------------------------------------------------------------


def test_order_crossover_matches_hand_calculation():
    p1 = np.array([0, 1, 2, 3, 4])
    p2 = np.array([4, 3, 2, 1, 0])
    child = A.order_crossover(p1, p2, FixedRNG(integers_return=[1, 3]))
    # Stück p1[1:4]=[1,2,3] übernommen; Rest aus p2 ohne {1,2,3}: p2=[4,3,2,1,0] -> [4,0], an Position 0 und 4
    assert child.tolist() == [4, 1, 2, 3, 0]


def test_swap_mutation_swaps_when_below_threshold():
    ind = np.array([0, 1, 2, 3])
    out = A.swap_mutation(ind, p_mut=0.5, rng=FixedRNG(integers_return=[0, 2], random_return=0.1))
    assert out.tolist() == [2, 1, 0, 3]


def test_swap_mutation_leaves_unchanged_above_threshold():
    ind = np.array([0, 1, 2, 3])
    out = A.swap_mutation(ind, p_mut=0.5, rng=FixedRNG(random_return=0.9))
    assert out.tolist() == ind.tolist()


def test_tournament_select_picks_the_lowest_fitness_contender():
    fitness = np.array([5.0, 1.0, 9.0, 3.0])
    # 2 Auswahlen, je 2 Kandidaten: (0,1) -> 1 gewinnt (Index 1); (2,3) -> 3 gewinnt (Index 3)
    rng = FixedRNG(integers_return=[[0, 1], [2, 3]])
    picked = A.tournament_select(fitness, k=2, n_select=2, rng=rng)
    assert picked.tolist() == [1, 3]


# --- Kanten / Diversität -------------------------------------------------------------------------------------------------------------------------


def test_tour_edges_and_edge_share_matches_hand_calculation():
    a = np.array([0, 1, 2, 3])
    b = np.array([0, 2, 1, 3])
    assert A.tour_edges(a) == {(0, 1), (1, 2), (2, 3), (0, 3)}
    assert 0.0 <= A.edge_share(a, b) < 1.0


def test_diversity_is_zero_for_an_identical_population():
    pop = np.array([[0, 1, 2, 3]] * 5)
    assert A.diversity(pop) == pytest.approx(0.0)


# --- 2-opt-Politur (Handrechnung) ---------------------------------------------------------------------------------------------------------------


def test_delta_2opt_matches_hand_calculation_on_a_crossed_square():
    """Quadrat-Ecken, aber Tour 0-2-1-3 kreuzt sich (Diagonalen statt Kanten) - 2-opt zwischen Position 0 und 2
    (Umkehren von Position 1..2) repariert die Tour zum unverkreuzten Quadrat (Umfang 4)."""
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    D = A.dist_matrix(xy)
    t = np.array([0, 2, 1, 3])                     # Länge: 0-2 (sqrt2) + 2-1 (1) + 1-3 (sqrt2) + 3-0 (1) = 2+2sqrt2
    delta, ok = A._delta_2opt(t, D)
    assert ok[0, 2]                                 # (i=0, j=2) ist ein gültiges Paar (j >= i+2)
    assert delta[0, 2] == pytest.approx(2.0 - 2.0 * np.sqrt(2.0))     # entfernt zwei sqrt2-Kanten, fügt zwei Einheitskanten ein
    new_length = A.tour_length(t, D) + delta[0, 2]
    assert new_length == pytest.approx(4.0)          # unverkreuztes Quadrat


def test_first_improving_2opt_finds_and_apply_2opt_executes_the_uncrossing_move():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    D = A.dist_matrix(xy)
    t = np.array([0, 2, 1, 3])
    move, evaluations = A._first_improving_2opt(t, D)
    assert move is not None and evaluations > 0
    i, j, delta = move
    assert delta < 0.0
    new_t = A._apply_2opt(t, i, j)
    assert A.tour_length(new_t, D) == pytest.approx(4.0)


def test_polish_reaches_the_2opt_local_optimum_on_a_small_square():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    D = A.dist_matrix(xy)
    t = np.array([0, 2, 1, 3])
    polished, evaluations = A.polish(t, D, max_moves=10)
    assert A.tour_length(polished, D) == pytest.approx(4.0)
    assert evaluations > 0
    # am lokalen Optimum angekommen: kein weiterer verbessernder 2-opt-Zug mehr
    move, _ = A._first_improving_2opt(polished, D)
    assert move is None


def test_polish_respects_the_move_budget():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    D = A.dist_matrix(xy)
    t = np.array([0, 2, 1, 3])
    polished, evaluations = A.polish(t, D, max_moves=0)
    assert polished.tolist() == t.tolist()
    assert evaluations == 0


# --- Hauptschleife -----------------------------------------------------------------------------------------------------------------------------


def _brute_force(D):
    n = len(D)
    best = None
    for perm in permutations(range(1, n)):
        tour = (0,) + perm
        length = A.tour_length(tour, D)
        if best is None or length < best:
            best = length
    return best


def test_run_memetic_returns_expected_shapes():
    rng = np.random.default_rng(0)
    xy = rng.random((11, 2)) * 100.0
    D = A.dist_matrix(xy)
    r = A.run_memetic(D, pop_size=10, generations=8, cx_prob=0.9, mut_prob=0.2, elitism=2, tournament_k=3, p_ls=0.5, ls_max_moves=10, seed=1, keep_history=True)
    assert r.best_history.shape == (9,)
    assert r.ls_evaluations_history.shape == (9,)
    assert len(r.generations) == 9


def test_run_memetic_with_p_ls_zero_never_calls_local_search():
    rng = np.random.default_rng(0)
    xy = rng.random((9, 2)) * 100.0
    D = A.dist_matrix(xy)
    r = A.run_memetic(D, pop_size=10, generations=10, cx_prob=0.9, mut_prob=0.2, elitism=2, tournament_k=3, p_ls=0.0, ls_max_moves=10, seed=2)
    assert r.ls_evaluations == 0


def test_run_memetic_ls_evaluations_grow_with_p_ls():
    rng = np.random.default_rng(0)
    xy = rng.random((12, 2)) * 100.0
    D = A.dist_matrix(xy)
    r_low = A.run_memetic(D, pop_size=12, generations=10, cx_prob=0.9, mut_prob=0.2, elitism=2, tournament_k=3, p_ls=0.1, ls_max_moves=10, seed=3)
    r_high = A.run_memetic(D, pop_size=12, generations=10, cx_prob=0.9, mut_prob=0.2, elitism=2, tournament_k=3, p_ls=1.0, ls_max_moves=10, seed=3)
    assert r_high.ls_evaluations > r_low.ls_evaluations


def test_run_memetic_best_history_is_monotonically_non_increasing():
    rng = np.random.default_rng(0)
    xy = rng.random((15, 2)) * 100.0
    D = A.dist_matrix(xy)
    r = A.run_memetic(D, pop_size=15, generations=20, cx_prob=0.9, mut_prob=0.2, elitism=2, tournament_k=3, p_ls=0.3, ls_max_moves=10, seed=4)
    assert np.all(np.diff(r.best_history) <= 1e-9)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_run_memetic_finds_the_brute_force_optimum_on_a_tiny_instance(seed):
    rng = np.random.default_rng(100 + seed)
    xy = rng.random((6, 2)) * 100.0
    D = A.dist_matrix(xy)
    optimum = _brute_force(D)
    r = A.run_memetic(D, pop_size=20, generations=30, cx_prob=0.9, mut_prob=0.2, elitism=2, tournament_k=3, p_ls=1.0, ls_max_moves=20, seed=seed)
    assert r.best_fitness == pytest.approx(optimum, rel=1e-6)


def test_run_memetic_without_history_leaves_generations_empty():
    rng = np.random.default_rng(0)
    xy = rng.random((9, 2)) * 100.0
    D = A.dist_matrix(xy)
    r = A.run_memetic(D, pop_size=10, generations=5, cx_prob=0.9, mut_prob=0.2, elitism=2, tournament_k=3, p_ls=0.5, ls_max_moves=10, seed=5, keep_history=False)
    assert r.generations == []
