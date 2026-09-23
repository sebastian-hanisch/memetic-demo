"""Auswertung der Memetic-Algorithmus-Demo: ein Lauf gegen eine Brute-Force-Referenz (kleine Instanzen), Sweeps,
und zwei Experimente: Kopfexperiment (validiert die Idee, mit Kosten - reines GA gegen voll memetisch auf einem
knappen eigenen Budget) und eigener Regler (Polierwahrscheinlichkeit p_LS)."""

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import permutations

import numpy as np

import mem_algorithm as A
import mem_constants as C
import mem_scenario as S

BRUTE_FORCE_MAX_N = 9      # (n-1)! Touren; bei 9 sind das 40 320 - noch < 1 s


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    ballung: int = C.DEFAULT_BALLUNG
    seed: int = C.DEFAULT_SEED
    pop: int = C.DEFAULT_POP
    gens: int = C.DEFAULT_GEN
    cx: float = C.DEFAULT_CX
    mut: float = C.DEFAULT_MUT
    elitism: int = C.DEFAULT_ELITE
    k: int = C.DEFAULT_K
    p_ls: float = C.DEFAULT_LS
    mem_seed: int = C.DEFAULT_MEM_SEED


@lru_cache(maxsize=256)
def instance(n, ballung, seed):
    inst = S.generate(n, ballung, seed)
    return inst, A.dist_matrix(inst.xy)


def run(settings, keep_history=False):
    inst, D = instance(settings.n, settings.ballung, settings.seed)
    return A.run_memetic(D, settings.pop, settings.gens, settings.cx, settings.mut, settings.elitism, settings.k, settings.p_ls, C.LS_MAX_MOVES, settings.mem_seed, keep_history=keep_history)


def brute_force(D):
    n = len(D)
    best = None
    for perm in permutations(range(1, n)):
        tour = (0,) + perm
        length = A.tour_length(tour, D)
        if best is None or length < best:
            best = length
    return best


@dataclass
class Analysis:
    settings: Settings
    result: object
    inst: object
    D: np.ndarray
    brute_force_optimum: float             # nur bei n <= BRUTE_FORCE_MAX_N berechnet, sonst NaN

    @property
    def gap(self):
        if not np.isfinite(self.brute_force_optimum) or self.brute_force_optimum == 0:
            return float("nan")
        return max(0.0, 100.0 * (self.result.best_fitness - self.brute_force_optimum) / self.brute_force_optimum)


def analyse(settings, keep_history=True):
    result = run(settings, keep_history=keep_history)
    inst, D = instance(settings.n, settings.ballung, settings.seed)
    optimum = brute_force(D) if settings.n + 1 <= BRUTE_FORCE_MAX_N else float("nan")
    return Analysis(settings, result, inst, D, optimum)


# --- Sweep (wie die Geschwister dieser Linie) --------------------------------------------------------------------------------------------------


def run_config(param, value, base, seeds=None):
    seeds = C.SWEEP_SEEDS if seeds is None else seeds
    s0 = replace(base, **{param: value})
    gaps, costs = [], []
    for mem_seed in seeds:
        a = analyse(replace(s0, mem_seed=mem_seed), keep_history=False)
        gaps.append(a.gap)
        costs.append(a.result.ls_evaluations)
    return {"gap": float(np.nanmean(gaps)), "ls_evaluations": float(np.mean(costs))}


def sweep(param, base=None, values=None):
    # eigenes knappes Budget auf der kleinen Vergleichsinstanz, p_LS=0 als Basis (reines GA) - sonst poliert schon
    # eine kleine Politurwahrscheinlichkeit die Populationsgröße-Sweep-Werte auf 0 % Abstand weg (Deckeneffekt,
    # gemessen: bei p_LS=0.5 zeigt der Populations-Sweep gar keine Variation mehr)
    base = Settings(n=C.COMPARISON_N, ballung=0, seed=C.COMPARISON_VEHICLE_SEED, pop=C.COMPARISON_POP, gens=C.COMPARISON_GENS, p_ls=0.0) if base is None else base
    values = C.SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(param, v, base)} for v in values]


# --- Experiment 1: Kopfexperiment - validiert die Idee, mit Kosten (eigenes knappes Budget) --------------------------------------------------


def head_experiment(n=None, ballung=None, seed=None, pop=None, gens=None, seeds=None):
    """Reines GA (p_LS=0) gegen voll memetisch (p_LS=1) auf einem knappen eigenen Budget - Tourqualität UND
    Rechenkosten (ls_evaluations) je Variante, gemittelt über mehrere Memetic-Seeds."""
    n = C.DEFAULT_N if n is None else n
    ballung = C.DEFAULT_BALLUNG if ballung is None else ballung
    seed = C.DEFAULT_SEED if seed is None else seed
    pop = C.HEAD_POP if pop is None else pop
    gens = C.HEAD_GENS if gens is None else gens
    seeds = C.HEAD_SEEDS if seeds is None else seeds

    rows = {}
    for label, p_ls in (("ga", 0.0), ("memetic", 1.0)):
        bests, costs = [], []
        for mem_seed in seeds:
            s = Settings(n=n, ballung=ballung, seed=seed, pop=pop, gens=gens, p_ls=p_ls, mem_seed=mem_seed)
            r = run(s, keep_history=False)
            bests.append(r.best_fitness)
            costs.append(r.ls_evaluations)
        rows[label] = {"best_median": float(np.median(bests)), "ls_evaluations_mean": float(np.mean(costs))}
    return rows


# --- Experiment 2: eigener Regler - Polierwahrscheinlichkeit p_LS -----------------------------------------------------------------------------


def p_ls_experiment(n=None, ballung=None, seed=None, values=None, seeds=None, pop=None, gens=None):
    n = C.DEFAULT_N if n is None else n
    ballung = C.DEFAULT_BALLUNG if ballung is None else ballung
    seed = C.DEFAULT_SEED if seed is None else seed
    values = C.P_LS_VALUES if values is None else values
    seeds = C.HEAD_SEEDS if seeds is None else seeds
    pop = C.HEAD_POP if pop is None else pop
    gens = C.HEAD_GENS if gens is None else gens

    rows = []
    for p_ls in values:
        bests, costs = [], []
        for mem_seed in seeds:
            s = Settings(n=n, ballung=ballung, seed=seed, pop=pop, gens=gens, p_ls=p_ls, mem_seed=mem_seed)
            r = run(s, keep_history=False)
            bests.append(r.best_fitness)
            costs.append(r.ls_evaluations)
        rows.append({"p_ls": p_ls, "best_median": float(np.median(bests)), "ls_evaluations_mean": float(np.mean(costs))})
    return rows
