"""Memetischer Algorithmus, numpy von Grund auf: Genetischer Algorithmus (Order-Crossover, Tausch-Mutation,
Turnierselektion, Elitismus - wortgleiche Mechanik zu genetic-algorithm-demos perm-Zweig) + lokale Politur jedes
Kindes (2-opt, erste Verbesserung, festes Zugbudget - schlanke Neuimplementierung des Delta-Musters aus
hill-climbing-demo, NUR 2-opt statt dessen vier Nachbarschaften). Lamarckian: das Kind wird durch die lokal
optimierte Tour ersetzt. Mit Wahrscheinlichkeit p_LS=0 ist das reines GA (Kontrollfall), p_LS=1 ist "voll
memetisch". Jede Politur verbraucht Nachbarschafts-Bewertungen; `MemeticResult.ls_evaluations` summiert sie -
macht den Kosten-Nutzen-Vergleich im Kopfexperiment möglich."""

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9


# --- Distanz / Tourlänge (Vehikel "Lieferroute", wie hill-climbing-demo/genetic-algorithm-demo) --------------------------------------------


def dist_matrix(xy):
    xy = np.asarray(xy, dtype=float)
    d = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((d * d).sum(axis=2))


def tour_length(tour, D):
    t = np.asarray(tour)
    return float(D[t, np.roll(t, -1)].sum())


def tour_length_batch(pop, D):
    nxt = np.roll(pop, -1, axis=1)
    return D[pop, nxt].sum(axis=1)


def tour_edges(tour):
    t = np.asarray(tour)
    a, b = t, np.roll(t, -1)
    return {(int(min(x, y)), int(max(x, y))) for x, y in zip(a, b)}


def edge_share(tour, reference):
    return len(tour_edges(tour) & tour_edges(reference)) / len(tour)


# --- Population: Erzeugen -------------------------------------------------------------------------------------------------------------------


def init_population(pop_size, n_nodes, rng):
    return np.array([rng.permutation(n_nodes) for _ in range(pop_size)], dtype=np.int64)


# --- Selektion --------------------------------------------------------------------------------------------------------------------------------


def tournament_select(fitness, k, n_select, rng):
    if n_select == 0:
        return np.empty(0, dtype=np.int64)
    n = len(fitness)
    contenders = rng.integers(0, n, size=(n_select, k))
    best = np.argmin(fitness[contenders], axis=1)
    return contenders[np.arange(n_select), best]


# --- Crossover / Mutation (wortgleiche Mechanik zu genetic-algorithm-demos perm-Zweig) -------------------------------------------------------


def order_crossover(p1, p2, rng):
    """Order Crossover (OX): ein Stück aus `p1` wird wörtlich übernommen, der Rest in der Reihenfolge von `p2` aufgefüllt."""
    n = len(p1)
    i, j = sorted(rng.integers(0, n, size=2))
    child = -np.ones(n, dtype=np.int64)
    child[i:j + 1] = p1[i:j + 1]
    taken = set(child[i:j + 1].tolist())
    fill = [g for g in p2.tolist() if g not in taken]
    pos = [k for k in range(n) if not (i <= k <= j)]
    for k, g in zip(pos, fill):
        child[k] = g
    return child


def swap_mutation(ind, p_mut, rng):
    if rng.random() >= p_mut:
        return ind.copy()
    out = ind.copy()
    i, j = rng.integers(0, len(ind), size=2)
    out[i], out[j] = out[j], out[i]
    return out


def diversity(pop, sample=40, rng=None):
    """1 - mittlerer Kantenanteil zwischen Paaren der Population."""
    n = len(pop)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if not pairs:
        return 0.0
    if rng is not None and len(pairs) > sample:
        idx = rng.choice(len(pairs), size=sample, replace=False)
        pairs = [pairs[k] for k in idx]
    shares = [edge_share(pop[i], pop[j]) for i, j in pairs]
    return float(1.0 - np.mean(shares))


# --- Lokale Suche: 2-opt, erste Verbesserung, festes Zugbudget (eigene, schlanke Fassung des Delta-Musters aus ------------------------------
# hill-climbing-demo/hc_algorithm.py - NUR 2-opt, kein Or-opt/Tausch, damit die Kosten je Individuum je Generation vertretbar bleiben) -------


def _valid_2opt_pairs(n):
    i, j = np.indices((n, n))
    ok = j >= i + 2
    ok[0, n - 1] = False
    return ok


def _delta_2opt(t, D):
    n = len(t)
    nxt = np.roll(t, -1)
    e = D[t, nxt]
    delta = D[np.ix_(t, t)] + D[np.ix_(nxt, nxt)] - e[:, None] - e[None, :]
    return delta, _valid_2opt_pairs(n)


def _first_improving_2opt(t, D):
    """Erste verbessernde 2-opt-Kante (i, j) in fester Reihenfolge, oder None - dazu die Zahl bewerteter Nachbarn."""
    n = len(t)
    delta, ok = _delta_2opt(t, D)
    d = np.where(ok, delta, np.inf)
    flat = d.ravel()
    improving = flat < -EPS
    if not improving.any():
        return None, int(ok.sum())
    k = int(np.argmax(improving))
    evaluations = int(ok.ravel()[:k + 1].sum())
    i, j = divmod(k, n)
    return (i, j, float(delta[i, j])), evaluations


def _apply_2opt(t, i, j):
    out = t.copy()
    out[i + 1:j + 1] = t[i + 1:j + 1][::-1]
    return out


def polish(tour, D, max_moves):
    """2-opt-Abstieg mit erster Verbesserung, bis ein lokales Optimum, `max_moves` oder das Zugbudget erreicht ist.
    Rückgabe: (polierte Tour, Zahl bewerteter Nachbarn)."""
    t = np.asarray(tour, dtype=np.int64).copy()
    evaluations = 0
    for _ in range(max_moves):
        move, count = _first_improving_2opt(t, D)
        evaluations += count
        if move is None:
            break
        i, j, _ = move
        t = _apply_2opt(t, i, j)
    return t, evaluations


# --- Hauptschleife -----------------------------------------------------------------------------------------------------------------------------


@dataclass
class Generation:
    population: np.ndarray
    fitness: np.ndarray


@dataclass
class MemeticResult:
    best_individual: np.ndarray
    best_fitness: float
    best_history: np.ndarray           # (generations + 1,)
    mean_history: np.ndarray
    diversity_history: np.ndarray
    ls_evaluations: int                 # Summe aller Nachbarschafts-Bewertungen der lokalen Suche über den ganzen Lauf
    ls_evaluations_history: np.ndarray  # kumulativ je Generation (0 = Startpopulation)
    generations: list = field(default_factory=list)


def run_memetic(D, pop_size, generations, cx_prob, mut_prob, elitism, tournament_k, p_ls, ls_max_moves, seed, keep_history=False):
    """Ein Memetic-Algorithmus-Lauf: GA-Hauptschleife (Order-Crossover, Tausch-Mutation, Turnier, Elitismus) wie
    genetic-algorithm-demos perm-Zweig, plus lokale Politur jedes Kindes mit Wahrscheinlichkeit `p_ls`
    (Lamarckian - das Kind wird ersetzt), Zugbudget `ls_max_moves` je Politur. `p_ls=0` ist reines GA."""
    rng = np.random.default_rng(seed)
    n_nodes = D.shape[0]
    elitism = min(elitism, pop_size)

    pop = init_population(pop_size, n_nodes, rng)
    fitness = tour_length_batch(pop, D)
    best_hist = [float(fitness.min())]
    mean_hist = [float(fitness.mean())]
    div_hist = [diversity(pop, rng=rng)]
    ls_eval_total = 0
    ls_eval_hist = [0]
    gens = [Generation(pop.copy(), fitness.copy())] if keep_history else []
    best_idx = int(np.argmin(fitness))
    best_ind, best_fit = pop[best_idx].copy(), float(fitness[best_idx])

    for _ in range(generations):
        order = np.argsort(fitness)
        elite = pop[order[:elitism]]
        n_children = pop_size - elitism
        parents_a = tournament_select(fitness, tournament_k, n_children, rng)
        parents_b = tournament_select(fitness, tournament_k, n_children, rng)
        children = []
        for pa, pb in zip(parents_a, parents_b):
            child = order_crossover(pop[pa], pop[pb], rng) if rng.random() < cx_prob else pop[pa].copy()
            child = swap_mutation(child, mut_prob, rng)
            if rng.random() < p_ls:
                child, evaluations = polish(child, D, ls_max_moves)
                ls_eval_total += evaluations
            children.append(child)
        children_arr = np.array(children, dtype=pop.dtype) if children else np.empty((0,) + pop.shape[1:], dtype=pop.dtype)
        pop = np.concatenate([elite, children_arr], axis=0)
        fitness = tour_length_batch(pop, D)
        best_hist.append(float(fitness.min()))
        mean_hist.append(float(fitness.mean()))
        div_hist.append(diversity(pop, rng=rng))
        ls_eval_hist.append(ls_eval_total)
        idx = int(np.argmin(fitness))
        if fitness[idx] < best_fit:
            best_ind, best_fit = pop[idx].copy(), float(fitness[idx])
        if keep_history:
            gens.append(Generation(pop.copy(), fitness.copy()))

    return MemeticResult(best_ind, best_fit, np.array(best_hist), np.array(mean_hist), np.array(div_hist), ls_eval_total, np.array(ls_eval_hist), gens)
