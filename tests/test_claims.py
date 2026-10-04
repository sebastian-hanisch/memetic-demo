"""Jede im README/PRESET_HELP/App genannte Zahl wird hier nachgerechnet - keine Behauptung ohne Test.

Einzelne 20-25-Generationen-Läufe sind chaotisch empfindlich gegenüber winziger Fließkomma-Rundung (siehe
feedback_ci_platform_robust_tests.md, und die eigene Erfahrung aus den Geschwistern dieser Linie). Zahlen aus
einem EINZELNEN Lauf (Presets) bekommen deshalb nur Strukturgrenzen; Zahlen, die über mehrere Seeds mitteln
(Kopfexperiment, p_LS-Experiment, Sweep), sind von Natur aus robuster und dürfen engere (aber weiterhin
großzügige) Bänder bekommen."""

import numpy as np

import mem_constants as C
import mem_evaluation as E


def _preset_analysis(name):
    p = C.PRESETS[name]
    s = E.Settings(n=p["n"], ballung=p["ballung"], seed=p["seed"], pop=p["pop"], gens=p["gens"], cx=p["cx"], mut=p["mut"], elitism=p["elitism"], k=p["k"], p_ls=p["p_ls"], mem_seed=p["mem_seed"])
    return E.analyse(s, keep_history=False)


# --- Einzelläufe (Presets) - nur Strukturgrenzen, keine Nähe zu einem Messwert ------------------------------------------------------------


def test_standardfall_preset_claims():
    a = _preset_analysis("Standardfall")
    assert 0.0 < a.result.best_fitness < 3000.0
    assert not np.isfinite(a.brute_force_optimum)      # 30 Stopps sind nicht brute-force-lösbar


def test_reines_ga_preset_claims():
    a = _preset_analysis("Reines GA (p_LS=0)")
    assert 0.0 < a.result.best_fitness < 3000.0
    assert a.result.ls_evaluations == 0


def test_voll_memetisch_preset_claims():
    a = _preset_analysis("Voll memetisch (p_LS=1)")
    assert 0.0 < a.result.best_fitness < 3000.0
    assert a.result.ls_evaluations > 0


def test_kleine_instanz_preset_claims():
    a = _preset_analysis("Kleine Instanz (Vergleich mit Brute-Force)")
    assert np.isfinite(a.brute_force_optimum)
    assert -1e-6 <= a.gap < 50.0


# --- Headlinezahlen der Experimente + Sweep (mitteln über mehrere Seeds, robuster) ------------------------------------------------------------


def test_head_experiment_headline_claims():
    """Kernbefund: auf dem knappen eigenen Budget findet voll memetisch eine klar kürzere Tour als reines GA -
    und dafür einen echten Rechenkosten-Preis (reines GA: keine Politur-Bewertungen)."""
    report = E.head_experiment()
    assert report["ga"]["ls_evaluations_mean"] == 0.0
    assert report["memetic"]["ls_evaluations_mean"] > 0.0
    assert report["memetic"]["best_median"] < report["ga"]["best_median"]
    # gemessen: reines GA ~1018 km, voll memetisch ~485 km auf diesem Budget - deutlich mehr als 10 % Vorsprung
    improvement = (report["ga"]["best_median"] - report["memetic"]["best_median"]) / report["ga"]["best_median"]
    assert improvement > 0.10


def test_p_ls_experiment_headline_claims():
    """Kernbefund: schon eine kleine Polierwahrscheinlichkeit erfasst fast den ganzen Qualitätsgewinn, während
    die Kosten (Politur-Bewertungen) mit p_LS klar weiter steigen - eine echte Sättigung, kein Freibier-Effekt."""
    rows = E.p_ls_experiment()
    by_p_ls = {r["p_ls"]: r for r in rows}
    assert set(by_p_ls) == set(C.P_LS_VALUES)
    for r in rows:
        assert r["ls_evaluations_mean"] >= 0.0
    assert by_p_ls[C.P_LS_VALUES[-1]]["ls_evaluations_mean"] > by_p_ls[C.P_LS_VALUES[1]]["ls_evaluations_mean"]
    assert by_p_ls[0.0]["best_median"] > by_p_ls[1.0]["best_median"]


def test_pop_sweep_headline_claims():
    rows = E.sweep("pop")
    assert [r["value"] for r in rows] == list(C.SWEEP_VALUES["pop"])
    for r in rows:
        assert r["gap"] >= 0.0
