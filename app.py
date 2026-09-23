"""Memetischer Algorithmus - lokale Suche poliert die Population - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Letztes Stück der Populations-Metaheuristiken-Linie der "Konzepte"-Reihe: die einzige Konvergenzkante der ganzen
Linie. Kombiniert den Genetischen Algorithmus (genetic-algorithm-demo, globale Exploration über eine Population)
mit Hill Climbings lokaler Suche (hill-climbing-demo, 2-opt-Politur) - jedes Kind wird nach Crossover/Mutation mit
Wahrscheinlichkeit p_LS zusätzlich lokal poliert, bevor es Teil der nächsten Generation wird. Vehikel ist dieselbe
diskrete Lieferroute wie beide Vorgänger (nur xy, kein CO2 - wie ant-system-demo/max-min-ant-system-demo).

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import mem_constants as C
from mem_evaluation import Settings, analyse, head_experiment, instance, p_ls_experiment, sweep
from mem_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_mem_seed, randomize_seed, sync_query_params
from mem_visualization import build_diversity_curve, build_fitness_curve, build_head_comparison, build_p_ls_experiment, build_route, build_sweep

st.set_page_config(page_title="Memetischer Algorithmus – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings, keep_history=True)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _head():
    return head_experiment()


@st.cache_data(show_spinner=False)
def _p_ls():
    return p_ls_experiment()


st.title("🧬🔧 Memetischer Algorithmus – lokale Suche poliert die Population")
st.markdown(
    """
Der Genetische Algorithmus durchsucht den Lösungsraum global (Population, Crossover, Mutation), lässt dabei aber
Verbesserungen liegen, die eine kurze lokale Suche sofort finden würde. Der **Memetische Algorithmus** kombiniert
beides: nach Crossover und Mutation wird jedes Kind zusätzlich mit Wahrscheinlichkeit **p_LS** lokal poliert
(**2-opt**, wie bei Hill Climbing) - erst danach wird es Teil der nächsten Generation.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - "
    "letztes Stück der Populations-Metaheuristiken-Linie der \"Konzepte\"-Reihe, die einzige **Konvergenzkante** der ganzen "
    "Linie - **ein** Verfahren an einem wachsenden Beispiel. Es kombiniert "
    "[genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/) (globale Exploration) mit "
    "[hill-climbing-demo](https://sebastianhanisch-hill-climbing-demo.streamlit.app/) (lokale Verfeinerung, 2-opt)."
)

with st.expander("So funktioniert der Memetische Algorithmus", expanded=True):
    st.markdown(
        """
1. **Population, Selektion, Crossover, Mutation.** Wie beim Genetischen Algorithmus: Turnierselektion, Order-
   Crossover, Tausch-Mutation, Elitismus.
2. **Politur (neu).** Mit Wahrscheinlichkeit **p_LS** wird jedes so erzeugte Kind zusätzlich lokal verbessert:
   ein 2-opt-Abstieg mit erster Verbesserung (wie bei Hill Climbing), begrenzt durch ein festes Zugbudget.
3. **Lamarckian.** Das polierte Kind ERSETZT das ursprüngliche - der nächste Generation-Durchlauf erbt die lokal
   verbesserte Tour, nicht nur ihre Bewertung.
4. **p_LS=0** ist reines GA (Kontrollfall), **p_LS=1** ist "voll memetisch" (jedes Kind wird poliert). Jede
   Politur kostet zusätzliche Rechenzeit (Nachbarschafts-Bewertungen) - dieser Preis wird unten offen mitgezählt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
cols = st.columns(len(preset_names))
for col, name in zip(cols, preset_names):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_stops = st.slider("Stopps", *bounds("n_slider"), key="n_slider", step=C.N_STEP, help="Anzahl der Kundenstopps (das Depot kommt dazu).")
    cluster_share = st.slider("Anteil der Stopps in Gruppen [%]", *bounds("ballung_slider"), key="ballung_slider", step=C.BALLUNG_STEP, help="Wie viele Stopps in fünf Gruppen (Städten) liegen statt gleichverteilt im Gebiet.")
    st.markdown("**Memetischer Algorithmus**")
    pop_size = st.slider("Populationsgröße", *bounds("pop_slider"), key="pop_slider", step=C.POP_STEP)
    generations = st.slider("Generationen", *bounds("gens_slider"), key="gens_slider", step=C.GEN_STEP)
    cx_prob = st.slider("Crossover-Rate", *bounds("cx_slider"), key="cx_slider", step=C.CX_STEP, format="%.2f")
    mut_prob = st.slider("Mutationsrate", *bounds("mut_slider"), key="mut_slider", step=C.MUT_STEP, format="%.2f")
    elitism = st.slider("Elitismus", *bounds("elitism_slider"), key="elitism_slider", step=C.ELITE_STEP)
    tournament_k = st.slider("Turniergröße k", *bounds("k_slider"), key="k_slider", step=C.K_STEP)
    p_ls = st.slider("Polierwahrscheinlichkeit p_LS", *bounds("p_ls_slider"), key="p_ls_slider", step=C.LS_STEP, format="%.2f", help="Anteil der Kinder, die zusätzlich lokal (2-opt) poliert werden. 0 = reines GA, 1 = jedes Kind wird poliert.")
    seed = st.number_input("Zufalls-Seed des Vehikels", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Vehikel generieren", width="stretch", on_click=randomize_seed)
    mem_seed = st.number_input("Zufalls-Seed des Laufs", *bounds("mem_seed_input"), key="mem_seed_input", step=1)
    st.button("🎲 Neuen Lauf würfeln", width="stretch", on_click=randomize_mem_seed)

sync_query_params({
    "n_slider": int(n_stops), "ballung_slider": int(cluster_share), "pop_slider": int(pop_size), "gens_slider": int(generations),
    "cx_slider": float(cx_prob), "mut_slider": float(mut_prob), "elitism_slider": int(elitism), "k_slider": int(tournament_k),
    "p_ls_slider": float(p_ls), "seed_input": int(seed), "mem_seed_input": int(mem_seed),
})

settings = Settings(int(n_stops), int(cluster_share), int(seed), int(pop_size), int(generations), float(cx_prob), float(mut_prob), int(elitism), int(tournament_k), float(p_ls), int(mem_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
result = a.result
n_gens_run = len(result.generations) - 1
data_key = settings
inst, D = instance(settings.n, settings.ballung, settings.seed)

# --- Der Memetische Algorithmus in Aktion -----------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Der Memetische Algorithmus in Aktion")
if "mem_gen" not in st.session_state or st.session_state.get("mem_gen_owner") != data_key:
    st.session_state["mem_gen"] = n_gens_run
    st.session_state["mem_gen_owner"] = data_key
gen_col, play_col = st.columns([5, 2])
with gen_col:
    gen = st.slider("Generation", 0, n_gens_run, key="mem_gen", help="0 = Startpopulation.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()


def _frames():
    if n_gens_run == 0:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n_gens_run, min(n_gens_run + 1, 40))})


def _render(g):
    gd = result.generations[g]
    best_idx = int(np.argmin(gd.fitness))
    with view_slot.container():
        c1, c2 = st.columns(2)
        c1.markdown(f"**Generation {g} von {n_gens_run} – bester Wert dieser Generation: {gd.fitness[best_idx]:,.1f}**".replace(",", "."))
        c1.plotly_chart(build_route(inst.xy, gd.population[best_idx]), width="stretch", key=f"g_map_{g}")
        c2.markdown("**Bester und mittlerer Wert je Generation**")
        c2.plotly_chart(build_fitness_curve(result.best_history[:g + 1], result.mean_history[:g + 1], reference=a.brute_force_optimum if np.isfinite(a.brute_force_optimum) else None), width="stretch", key=f"g_curve_{g}")


if auto_play:
    for f in _frames():
        _render(f)
        time.sleep(0.15)
else:
    _render(gen)

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was der Memetische Algorithmus gefunden hat")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Beste gefundene Tour", f"{result.best_fitness:,.1f} km".replace(",", "."), delta=f"Start {result.best_history[0]:,.1f}".replace(",", "."), delta_color="off")
if np.isfinite(a.brute_force_optimum):
    m2.metric("Abstand zum Brute-Force-Optimum", f"{a.gap:.1f} %")
else:
    m2.metric("Brute-Force-Optimum", "nicht berechenbar (zu viele Stopps)")
m3.metric("Diversität am Ende", f"{result.diversity_history[-1]:.2f}", delta=f"Start {result.diversity_history[0]:.2f}", delta_color="off")
m4.metric("Bewertungen der lokalen Suche", f"{result.ls_evaluations:,}".replace(",", "."), help="Summe aller Nachbarschafts-Bewertungen der 2-opt-Politur über den ganzen Lauf - der Preis von p_LS > 0.")

st.markdown("**Diversität über die Generationen**")
st.plotly_chart(build_diversity_curve(result.diversity_history), width="stretch", key="diversity_curve")

st.markdown("---")

# --- Sweep -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängen Abstand zum Optimum und Kosten von Populationsgröße bzw. p_LS ab?")
st.caption("Läuft auf der kleinen Vergleichsinstanz (8 Stopps) mit eigenem knappen Budget und p_LS=0 als Basis (sonst poliert schon wenig lokale Suche die Populationsgröße-Werte auf 0 % Abstand weg).")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(C.SWEEP_LABELS), format_func=lambda k: C.SWEEP_LABELS[k], key="sweep_select")
if st.button("Sweep über 5 feste Memetic-Seeds berechnen (dauert etwa 10 bis 30 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {sweep_param}
if sweep_param in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep..."):
        rows_sweep = _sweep(sweep_param, None)
    st.plotly_chart(build_sweep(rows_sweep, C.SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")

st.markdown("---")

# --- Experiment 1: Kopfexperiment - validiert die Idee, mit Kosten -----------------------------------------------------------------------------

st.subheader("🔬 Wie viel besser ist voll memetisch als reines GA - und wie viel kostet das?")
st.caption(f"Eigenes, knapp bemessenes Budget (Population {C.HEAD_POP}, {C.HEAD_GENS} Generationen, unabhängig von der Seitenleiste) - erst wenn der GA nicht ohnehin genug Zeit zum Konvergieren hat, zeigt sich der Unterschied deutlich.")
if st.button("Reines GA gegen voll memetisch rechnen (dauert etwa 5 Sekunden)", key="head_start"):
    st.session_state["head_on"] = True
if st.session_state.get("head_on"):
    with st.spinner("Rechne 20 Läufe je Variante..."):
        report = _head()
    st.plotly_chart(build_head_comparison(report), width="stretch", key="head_chart")
    improvement = 100.0 * (report["ga"]["best_median"] - report["memetic"]["best_median"]) / report["ga"]["best_median"]
    st.warning(
        f"**Ehrlicher Befund:** Auf diesem knappen Budget findet voll memetisch eine um {improvement:.0f} % kürzere Tour als reines GA - "
        f"aber dafür im Mittel {report['memetic']['ls_evaluations_mean']:,.0f} zusätzliche Nachbarschafts-Bewertungen der lokalen Suche "
        "(reines GA: keine). Die klassische \"memetisch gewinnt pro Generation\"-Behauptung stimmt hier - aber nur die halbe Geschichte "
        "ohne diesen Preis wäre geschönt.".replace(",", ".")
    )

st.markdown("---")

# --- Experiment 2: eigener Regler - Polierwahrscheinlichkeit p_LS ------------------------------------------------------------------------------

st.subheader("🔬 Wie stark hängen Qualität und Kosten von der Polierwahrscheinlichkeit p_LS ab?")
st.caption("Gleiches knappes Budget wie oben, 5 Werte von p_LS.")
if st.button("p_LS von 0 bis 1 durchfahren (dauert etwa 10 Sekunden)", key="p_ls_start"):
    st.session_state["p_ls_on"] = True
if st.session_state.get("p_ls_on"):
    with st.spinner("Rechne 5 Polierwahrscheinlichkeiten × 20 Läufe..."):
        rows_p_ls = _p_ls()
    st.plotly_chart(build_p_ls_experiment(rows_p_ls), width="stretch", key="p_ls_chart")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    r"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Nur Lamarckian** | Das polierte Kind ersetzt den Genotyp immer. Ein Baldwinian-Modus (lokale Güte nur zur Bewertung nutzen, Genotyp unverändert lassen) ist eine ernsthafte Alternative aus der Literatur - hier nicht umgesetzt. | Hier nicht umgesetzt |
| **Nur 2-opt als lokale Nachbarschaft** | hill-climbing-demo zeigt 2-opt + Or-opt zusammen als stärkste Kombination - hier bewusst nur 2-opt, um die Kosten je Individuum je Generation vertretbar zu halten. | Hier nicht umgesetzt |
| **Festes Zugbudget je Politur** | Nicht als eigener Regler exponiert - ein zusätzlicher, hier fest verdrahteter Freiheitsgrad. | Hier nicht umgesetzt |
| **p_LS ist gut gewählt** | Zu klein: kaum Nutzen. Zu groß: unnötig hohe Kosten, sobald der Nutzen schon gesättigt ist (siehe Experiment oben). | Muss von Hand eingestellt werden, wie bei jedem Regler dieser Linie |
"""
)
st.caption(
    "Letztes Stück der gesamten Populations-Metaheuristiken-Linie (11 Stücke) - kein Nachfolger. Vorgänger: "
    "[genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/) (globale Exploration) und "
    "[hill-climbing-demo](https://sebastianhanisch-hill-climbing-demo.streamlit.app/) (lokale Verfeinerung, aus der Trajektorien-Linie)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**GA-Teil** (wie genetic-algorithm-demo). Turnierselektion, Order-Crossover, Tausch-Mutation, Elitismus - siehe
dortige Formulierung.

**Politur.** Mit Wahrscheinlichkeit $p_{LS}$ wird ein Kind $x$ durch $\text{2opt}(x)$ ersetzt: solange eine Kante
$(i, j)$ mit $j \ge i+2$ existiert, deren Umkehrung die Tourlänge verringert (erste Verbesserung in fester
Reihenfolge), wird sie ausgeführt - bis kein solcher Zug mehr existiert oder das Zugbudget `LS_MAX_MOVES`
erreicht ist.

**Kosten.** Jede Politur verbraucht Nachbarschafts-Bewertungen (wie bei Hill Climbing); die Summe über den ganzen
Lauf macht den Kosten-Nutzen-Vergleich möglich.

Implementiert in `mem_algorithm.py` (GA-Operatoren, 2-opt-Politur, Hauptschleife), `mem_scenario.py` (Vehikel),
`mem_evaluation.py` (Kennzahlen, Sweep, Kopfexperiment, p_LS-Experiment).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
