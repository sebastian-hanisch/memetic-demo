"""Konstanten der Memetic-Algorithmus-Demo: Vehikel (wie hill-climbing-demo/genetic-algorithm-demo, nur xy), GA-
Regler, Regler der lokalen Suche, Presets (Presets folgen nach den Messungen)."""

# --- Vehikel: Lieferroute (wortgleich aus hill-climbing-demo/hc_scenario.py, wie ant-system-demo/mmas) ---------------------------------------

AREA = 100.0
N_CLUSTERS = 5
CLUSTER_SIGMA = 6.0                # Streuung einer Gruppe in km
CLUSTER_MARGIN = 12.0              # Gruppenmittelpunkte liegen mindestens so weit vom Rand entfernt
N_MIN, N_MAX, DEFAULT_N, N_STEP = 8, 100, 30, 2      # N_MIN=8, damit der Slider auch die kleine Vergleichsinstanz abbilden kann (wie ant-system-demo)
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG, BALLUNG_STEP = 0, 100, 0, 25

# --- Genetischer Teil (wie genetic-algorithm-demos perm-Zweig) --------------------------------------------------------------------------------

POP_MIN, POP_MAX, DEFAULT_POP, POP_STEP = 10, 150, 40, 10
GEN_MIN, GEN_MAX, DEFAULT_GEN, GEN_STEP = 10, 300, 80, 10
CX_MIN, CX_MAX, DEFAULT_CX, CX_STEP = 0.0, 1.0, 0.9, 0.05
MUT_MIN, MUT_MAX, DEFAULT_MUT, MUT_STEP = 0.0, 1.0, 0.2, 0.05
ELITE_MIN, ELITE_MAX, DEFAULT_ELITE, ELITE_STEP = 0, 10, 2, 1
K_MIN, K_MAX, DEFAULT_K, K_STEP = 2, 8, 3, 1

# --- Lokale Suche (2-opt, erste Verbesserung) - eigener Regler: Polierwahrscheinlichkeit p_LS ---------------------------------------------------

LS_MIN, LS_MAX, DEFAULT_LS, LS_STEP = 0.0, 1.0, 0.5, 0.05     # p_LS: Anteil der Kinder, die lokal poliert werden
LS_MAX_MOVES = 30              # festes Zugbudget je Politur (nicht als Regler exponiert, siehe README "Grenzen")

SEED_MAX = 999999
DEFAULT_SEED = 35              # Vehikel-Seed (wie hill-climbing-demo/genetic-algorithm-demo)
DEFAULT_MEM_SEED = 7           # Seed des Memetic-Laufs selbst (Population, Selektion, Crossover, Mutation, Politur)

# --- Kleine Vergleichsinstanz (Brute-Force) - wie jedes Geschwister der diskreten Linie ----------------------------------------------------

COMPARISON_N = 8
COMPARISON_VEHICLE_SEED = 19
COMPARISON_POP, COMPARISON_GENS = 20, 20

# --- Kopfexperiment - validiert die Idee, mit Kosten (eigenes knappes Budget, wie GAs OPERATOR_POP/GENS) -----------------------------------

HEAD_POP, HEAD_GENS = 20, 25
HEAD_SEEDS = tuple(range(600000, 600020))

# --- Eigener Regler: Polierwahrscheinlichkeit p_LS --------------------------------------------------------------------------------------------

P_LS_VALUES = (0.0, 0.1, 0.25, 0.5, 1.0)

# --- 📐 Sweep ------------------------------------------------------------------------------------------------------------------------------------

SWEEP_SEEDS = tuple(range(700000, 700005))
SWEEP_VALUES = {"p_ls": P_LS_VALUES, "pop": (10, 20, 40, 60, 100)}
SWEEP_LABELS = {"p_ls": "Polierwahrscheinlichkeit p_LS", "pop": "Populationsgröße"}


def _preset(n=DEFAULT_N, ballung=DEFAULT_BALLUNG, pop=DEFAULT_POP, gens=DEFAULT_GEN, cx=DEFAULT_CX, mut=DEFAULT_MUT, elitism=DEFAULT_ELITE, k=DEFAULT_K, p_ls=DEFAULT_LS, seed=DEFAULT_SEED, mem_seed=DEFAULT_MEM_SEED):
    return {"n": n, "ballung": ballung, "seed": seed, "pop": pop, "gens": gens, "cx": cx, "mut": mut, "elitism": elitism, "k": k, "p_ls": p_ls, "mem_seed": mem_seed}


PRESETS = {
    "Standardfall": _preset(),
    "Reines GA (p_LS=0)": _preset(p_ls=0.0),
    "Voll memetisch (p_LS=1)": _preset(p_ls=1.0),
    "Große Population": _preset(pop=100),
    "Kleine Instanz (Vergleich mit Brute-Force)": _preset(n=COMPARISON_N, seed=COMPARISON_VEHICLE_SEED, pop=COMPARISON_POP, gens=COMPARISON_GENS),
}
PRESET_HELP = {
    "Standardfall": "30 Stopps, Population 40, 80 Generationen, p_LS=0.5 - findet 485.2 km (Start: rund 1600 km).",
    "Reines GA (p_LS=0)": "Gleiches Budget ohne Politur - 717.2 km, deutlich schlechter als mit lokaler Suche (485.2 km).",
    "Voll memetisch (p_LS=1)": "Jedes Kind wird poliert - 485.2 km, auf diesem großzügigen Budget identisch zu p_LS=0.5 (die Kosten steigen trotzdem weiter, siehe eigener Regler).",
    "Große Population": "Population 100 statt 40 - ebenfalls 485.2 km, kein zusätzlicher Nutzen bei diesem Budget.",
    "Kleine Instanz (Vergleich mit Brute-Force)": "8 Stopps - der Memetische Algorithmus trifft mit 255.4 km exakt das Brute-Force-Optimum (0.0 % Abstand).",
}
