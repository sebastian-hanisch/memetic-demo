"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Generation-Slider (0=Startpopulation), Abspielen ohne
doppelte Schlüssel, Würfel-Knöpfe, Permalink-Grenzen, Sweep + beide Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import mem_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def test_default_run_has_no_exception_and_shows_metrics():
    at = _run()
    _ok(at)
    assert at.metric


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["n_slider"] == p["n"] and at.session_state["p_ls_slider"] == p["p_ls"]
    assert at.metric


def test_generation_slider_runs_at_various_positions():
    at = _run(gens_slider=20, pop_slider=15)
    _ok(at)
    gen_slider = next(s for s in at.slider if s.key == "mem_gen")
    gen_slider.set_value(0).run()
    _ok(at)
    assert at.get("plotly_chart")
    gen_slider.set_value(20).run()
    _ok(at)
    assert at.get("plotly_chart")


def test_play_runs_without_duplicate_keys():
    at = _run(gens_slider=20, pop_slider=15)
    next(b for b in at.button if b.label == "▶️ Abspielen").click().run()
    _ok(at)


def test_dice_buttons_change_the_seeds():
    at = _run()
    old_seed = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neues Vehikel generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old_seed
    old_mem_seed = at.session_state["mem_seed_input"]
    next(b for b in at.button if b.label == "🎲 Neuen Lauf würfeln").click().run()
    _ok(at)
    assert at.session_state["mem_seed_input"] != old_mem_seed


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["pls"] = "9999"
    at.query_params["n"] = "13"
    at.run()
    _ok(at)
    assert at.session_state["p_ls_slider"] == C.LS_MAX
    assert at.session_state["n_slider"] == 12


@pytest.mark.parametrize("kw", [dict(n_slider=C.N_MIN), dict(n_slider=C.N_MAX), dict(pop_slider=C.POP_MIN), dict(gens_slider=C.GEN_MIN), dict(mut_slider=1.0, cx_slider=0.0), dict(p_ls_slider=C.LS_MIN), dict(p_ls_slider=C.LS_MAX)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_sweep_runs_on_demand():
    at = _run()
    at.selectbox(key="sweep_select").set_value("p_ls").run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_head_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "HEAD_SEEDS", (1, 2))
    monkeypatch.setattr(C, "HEAD_POP", 8)
    monkeypatch.setattr(C, "HEAD_GENS", 6)
    at = _run()
    next(b for b in at.button if b.key == "head_start").click().run()
    _ok(at)
    assert at.session_state["head_on"] and at.get("plotly_chart")


def test_p_ls_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "HEAD_SEEDS", (1, 2))
    monkeypatch.setattr(C, "HEAD_POP", 8)
    monkeypatch.setattr(C, "HEAD_GENS", 6)
    monkeypatch.setattr(C, "P_LS_VALUES", (0.0, 1.0))
    at = _run()
    next(b for b in at.button if b.key == "p_ls_start").click().run()
    _ok(at)
    assert at.session_state["p_ls_on"] and at.get("plotly_chart")


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Nur Lamarckian" in m.value for m in at.markdown)
