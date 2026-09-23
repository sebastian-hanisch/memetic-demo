# 🧬🔧 Memetischer Algorithmus – lokale Suche poliert die Population

**[→ Demo live ausprobieren](https://sebastianhanisch-memetic-demo.streamlit.app/)**

Letztes Stück (11. von 11) der **Populations-Metaheuristiken-Linie** der "Konzepte"-Reihe im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Die einzige
**Konvergenzkante** der ganzen Linie: kombiniert [genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/)
(globale Exploration über eine Population) mit [hill-climbing-demo](https://sebastianhanisch-hill-climbing-demo.streamlit.app/)
(lokale Verfeinerung, 2-opt, aus der Trajektorien-Linie) - jedes Kind wird nach Crossover/Mutation zusätzlich mit
Wahrscheinlichkeit $p_{LS}$ lokal poliert, bevor es Teil der nächsten Generation wird.

## Warum dieses Problem

Der Genetische Algorithmus durchsucht den Lösungsraum global (Population, Crossover, Mutation), lässt dabei aber
Verbesserungen liegen, die eine kurze lokale Suche sofort finden würde - z. B. eine Tour, die sich an einer
Stelle unnötig kreuzt. Hill Climbing findet solche Verbesserungen zuverlässig, bleibt aber im ersten lokalen
Optimum stecken. Der **Memetische Algorithmus** kombiniert beides: die Population liefert globale Exploration,
die lokale Suche liefert lokale Verfeinerung jedes einzelnen Fundes.

## Modell

Dieselbe diskrete Lieferroute wie hill-climbing-demo/genetic-algorithm-demo/ant-system-demo/max-min-ant-system-demo:
ein Depot in der Mitte und *n* Kundenstopps in einem 100×100-km-Gebiet. **`mem_scenario.generate` reproduziert die
Vehikel-Erzeugung wortgleich** (nur die xy-Koordinaten, wie ant-system-demo/max-min-ant-system-demo) - bei
Standard-Vehikel-Seed 35 bzw. der geteilten kleinen Vergleichsinstanz (n=8, Seed 19) bitidentisch zu den
Vorgänger-Demos.

## Methodik

**GA-Teil** (`mem_algorithm.py`, eigenständig implementiert, gleiche Formeln wie genetic-algorithm-demos
perm-Zweig): Turnierselektion, Order-Crossover, Tausch-Mutation, Elitismus.

**Lokale-Suche-Teil** (Lamarckian - das Kind wird durch die lokal optimierte Tour ersetzt): nach Crossover und
Mutation wird jedes Kind mit Wahrscheinlichkeit $p_{LS}$ (eigener Regler) durch einen 2-opt-Abstieg mit erster
Verbesserung poliert, begrenzt durch ein festes Zugbudget (`LS_MAX_MOVES=30`, nicht als Regler exponiert). Eigene,
schlanke Neuimplementierung des Delta-Musters aus `hc_algorithm.py` - NUR 2-opt statt der vier Nachbarschaften von
hill-climbing-demo, um die Kosten je Individuum je Generation vertretbar zu halten. $p_{LS}=0$ ist reines GA,
$p_{LS}=1$ ist "voll memetisch".

**Rechenkosten werden mitgezählt**: jeder lokale Abstieg verbraucht Nachbarschafts-Bewertungen; `MemeticResult`
führt eine laufende Summe (`ls_evaluations`), sichtbar in jeder Auswertung dieser Demo - macht den ehrlichen
Kosten-Nutzen-Vergleich möglich.

**Kreuzprobe**: kein externes Referenzpaket für "Memetic Algorithm" verfügbar/sinnvoll konfigurierbar - Rigor über
Handrechnungen (Order-Crossover/Tausch-Mutation wie genetic-algorithm-demo, 2-opt-Delta wie hill-climbing-demo, an
kleinen Beispielen mit bekanntem lokalem Optimum) plus Brute-Force-Vergleich auf der kleinen Instanz.

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| Wie viel besser ist voll memetisch als reines GA - und wie viel kostet das? | Auf einem knappen eigenen Budget findet voll memetisch eine um 52 % kürzere Tour als reines GA (1009 km → 485 km, Median über 20 Läufe) - aber dafür im Mittel rund 586.000 zusätzliche Nachbarschafts-Bewertungen der lokalen Suche (reines GA: keine). | `test_head_experiment_headline_claims` |
| Wie stark hängen Qualität und Kosten von der Polierwahrscheinlichkeit $p_{LS}$ ab? | Schon $p_{LS}=0{,}1$ erfasst fast den gesamten Qualitätsgewinn (486,9 km, gegenüber 485,2 km bei $p_{LS}=1{,}0$) - die Kosten steigen dagegen über den ganzen Bereich weiter linear (rund 89.000 bei $p_{LS}=0{,}1$ bis rund 594.000 bei $p_{LS}=1{,}0$). Eine echte Sättigung, kein Freibier-Effekt. | `test_p_ls_experiment_headline_claims` |
| Wie nah kommt der Memetische Algorithmus ans echte Optimum? | Auf der kleinen Vergleichsinstanz (8 Stopps) trifft der Standardfall exakt das Brute-Force-Optimum (255,4 km, 0,0 % Abstand). | `test_kleine_instanz_preset_claims` |
| Wie stark hängt der Abstand zum Optimum von der Populationsgröße ab (ohne Politur)? | Bei reinem GA ($p_{LS}=0$) auf der kleinen Instanz sinkt der Abstand mit wachsender Population, aber nicht ganz glatt (9,4 % bei Population 10 bis 0,0 % ab Population 60, mit einer nicht-monotonen Zwischenstufe bei 40) - ehrlich so berichtet, nicht zu einer glatten Kurve geglättet. | `test_pop_sweep_headline_claims` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist |
|---|---|
| **Nur Lamarckian** | Das polierte Kind ersetzt den Genotyp immer. Ein Baldwinian-Modus (lokale Güte nur zur Bewertung nutzen, Genotyp unverändert lassen) ist eine ernsthafte Alternative aus der Literatur - hier nicht umgesetzt. |
| **Nur 2-opt als lokale Nachbarschaft** | hill-climbing-demo zeigt 2-opt + Or-opt zusammen als stärkste Kombination - hier bewusst nur 2-opt, um die Kosten je Individuum je Generation vertretbar zu halten. |
| **Festes Zugbudget je Politur** | Nicht als eigener Regler exponiert - ein zusätzlicher, hier fest verdrahteter Freiheitsgrad (`LS_MAX_MOVES=30`). |
| **$p_{LS}$ ist gut gewählt** | Zu klein: kaum Nutzen. Zu groß: unnötig hohe Kosten, sobald der Nutzen schon gesättigt ist (siehe Befunde oben). |

Letztes Stück der gesamten Populations-Metaheuristiken-Linie (11 Stücke) - kein Nachfolger.

## Tests

64 Tests (`pytest tests/ -v`): Order-Crossover/Tausch-Mutation/Turnierselektion und die 2-opt-Politur per
Handrechnung geprüft (inkl. eines Beispiels mit bekanntem lokalem Optimum), Brute-Force-Vergleich auf sehr kleinen
Instanzen, Szenario-Erzeugung bitidentisch zu hill-climbing-demo/genetic-algorithm-demo geprüft, AppTest-Rauchtests
(jedes Preset, Generation-Slider inkl. Abspielen, Permalink-Grenzen, Sweep + beide Experimente auf Abruf) und
`test_claims.py` (jede Zahl aus diesem README, mit CI-robusten Bändern für Einzellauf-Kennzahlen - siehe
`feedback_ci_platform_robust_tests.md`, von Anfang an angewendet).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `mem_constants.py` | Regler-Grenzen, Vehikel-Konstanten, Presets |
| `mem_presets.py` | Permalink/Presets-Mechanik |
| `mem_scenario.py` | Vehikel-Erzeuger (Lieferroute), wortgleich zu hill-climbing-demo/genetic-algorithm-demo |
| `mem_algorithm.py` | GA-Operatoren, 2-opt-Politur, Hauptschleife (mit Kostenzählung) |
| `mem_evaluation.py` | Kennzahlen, Brute-Force-Referenz, Kopfexperiment, p_LS-Experiment, Sweep |
| `mem_visualization.py` | Plotly-Abbildungen (Karte, Fitness-/Diversitätsverlauf, Qualität-vs-Kosten-Vergleiche) |

## Bewusst nicht umgesetzt

- Baldwinian-Modus (lokale Güte nur zur Bewertung, Genotyp unverändert).
- Or-opt oder weitere lokale Nachbarschaften - nur 2-opt.
- Adaptives $p_{LS}$ (z. B. nur die Elite polieren, oder $p_{LS}$ über die Generationen absenken).
- Ein PDF-Export - wie bei den anderen Konzepte-Demos dieses Portfolios nicht Teil der Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy.
