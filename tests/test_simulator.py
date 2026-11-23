import numpy as np
import pandas as pd
import pytest

from agentic_trading.agent.baselines import (
    equal_weights,
    kelly_weights,
    top_n_weights,
    weekly_dates,
)
from agentic_trading.agent.simulator import (
    Market,
    Portfolio,
    Rules,
    agent_rules,
    max_stop_distance,
    performance,
    reference_rules,
    simulate,
)

DAYS = pd.bdate_range("2010-01-04", periods=6)


def frame(o, h=None, lo=None, c=None, days=DAYS):
    o = np.asarray(o, float)
    c = o if c is None else np.asarray(c, float)
    h = np.maximum(o, c) if h is None else np.asarray(h, float)
    lo = np.minimum(o, c) if lo is None else np.asarray(lo, float)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, index=days[:len(o)])


def once(weights: dict):
    return lambda state: pd.Series(weights, dtype=float)


def test_trades_at_next_open_with_costs_and_no_leverage():
    m = Market.from_frames({"X": frame([10, 11, 12, 12, 12, 12], c=[10, 12, 12, 12, 12, 12])})
    res = simulate(m, once({"X": 1.0}), [DAYS[0]], reference_rules(0.01))
    assert res.nav.iloc[0] == 1.0                            # el día de la decisión no se opera
    # compra en la apertura siguiente (11), con el 1 % de coste y sin pasar de la caja
    assert res.nav.iloc[1] == pytest.approx(12 / 11 / 1.01)
    assert res.cash.min() >= -1e-12 and res.costs.iloc[1] == pytest.approx(0.01 / 1.01)


def test_min_weight_and_own_calendars():
    x = frame([10] * 6)
    y = frame([20, 20, 20, 20, 20], days=DAYS.delete(1))      # Y no cotiza el segundo día
    m = Market.from_frames({"X": x, "Y": y, "Z": frame([5] * 6)})
    res = simulate(m, once({"X": 0.4, "Y": 0.4, "Z": 0.005}), [DAYS[0]], Rules(cost=0.0))
    assert res.positions.iloc[1] == 1 and res.positions.iloc[2] == 2   # Y, en su próxima vela
    assert res.cash.iloc[-1] == pytest.approx(0.2)            # Z (< 1 %) no se compra


def test_stop_at_price_or_at_open_with_gap():
    rules = Rules(cost=0.0, min_weight=0.01, stop_loss=0.03)
    for o2, lo2, expected in ((100, 60, 0.97), (50, 50, 0.95)):
        m = Market.from_frames({"X": frame([100, 100, o2, 50, 50, 50], lo=[100, 100, lo2, 50, 50,
                                                                         50])})
        res = simulate(m, once({"X": 0.10}), [DAYS[0]], rules)
        # stop = 100 · (1 − 0,03 / 0,10) = 70: salta a 70 (pierde el 3 %) o, si abre por debajo,
        # a la apertura
        assert res.nav.iloc[-1] == pytest.approx(expected)
        assert len(res.stops) == 1 and res.positions.iloc[-1] == 0


def test_jump_across_invalid_bars_is_spliced():
    # X salta ×50 a través de una vela no válida (otra entidad, split sin corroborar...)
    x = frame([10, 10, 999, 500, 500, 510], c=[10, 10, 999, 500, 500, 520])
    x["valid"] = [True, True, False, True, True, True]
    m = Market.from_frames({"X": x})
    assert m.breaks == 1 and DAYS[2] not in m.dates
    res = simulate(m, once({"X": 1.0}), [DAYS[0]], reference_rules(0.0))
    # compra a 10; en el corte (apertura 500 → 10) no gana nada; después, 500 → 520 (+4 %)
    assert res.nav.iloc[-1] == pytest.approx(1.04)
    assert Market.from_frames({"X": x.assign(valid=True)}).breaks == 0


def test_ineligible_position_is_kept():
    m = Market.from_frames({"X": frame([10] * 6), "Y": frame([10] * 6)})
    calls = []

    def policy(state):
        calls.append(state.weights.copy())
        return pd.Series({"X": 0.5, "Y": 0.5}) if len(calls) == 1 else pd.Series({"X": 1.0})
    res = simulate(m, policy, [DAYS[0], DAYS[2]], Rules(cost=0.0, stop_loss=None))
    assert calls[1]["Y"] == pytest.approx(0.5)               # Y sigue en cartera ...
    assert res.positions.iloc[-1] == 2 and res.cash.iloc[-1] == pytest.approx(0.0)  # ... y X cabe


def test_cash_never_negative_with_gaps():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2010-01-04", periods=120)
    frames = {}
    for i in range(30):
        c = 50 * np.exp(np.cumsum(rng.normal(0, 0.03, len(days))))
        o = c * np.exp(rng.normal(0.01, 0.02, len(days)))   # aperturas por encima del cierre
        frames[f"S{i}"] = frame(o, np.maximum(o, c) * 1.01, np.minimum(o, c) * 0.99, c, days)
    m = Market.from_frames(frames)
    policy = lambda state: pd.Series(rng.dirichlet(np.ones(30)), index=list(frames))
    res = simulate(m, policy, days[::5], Rules(cost=0.005))
    assert res.cash.min() >= -1e-9 and (res.nav > 0).all()


def test_kelly_weights_keep_top_positions_above_one_percent():
    mu = pd.Series(np.r_[0.05, 0.03, np.full(500, 0.001), -0.02], index=[f"S{i}" for i in
                                                                        range(503)])
    w = kelly_weights(mu, pd.Series(0.05, index=mu.index))
    assert w.sum() == pytest.approx(1.0) and w[w > 0].min() >= 0.01
    assert w["S0"] > w["S1"] > 0 and w.iloc[-1] == 0
    flat = kelly_weights(pd.Series(0.01, index=range(500)), pd.Series(0.05, index=range(500)))
    assert (flat > 0).sum() == 100 and flat.max() == pytest.approx(0.01)
    assert kelly_weights(pd.Series([-0.1, 0.0]), pd.Series([0.1, 0.1])).sum() == 0


def test_top_n_with_exit_rank():
    mu = pd.Series(-np.arange(100.0), index=[f"S{i}" for i in range(100)])  # S0 el mejor
    w = top_n_weights(mu, held=["S50", "S70"], n=20, exit_rank=60)
    assert w["S50"] == pytest.approx(0.05) and w["S70"] == 0     # S50 se queda, S70 sale
    assert (w > 0).sum() == 20 and w[[f"S{i}" for i in range(19)]].gt(0).all()
    w2 = top_n_weights(mu.drop("S3"), held=["S3"], n=20)          # sin predicción: ocupa hueco
    assert (w2 > 0).sum() == 19
    assert equal_weights(mu).sum() == pytest.approx(1.0)


def test_weekly_dates_skip_partial_days():
    days = pd.bdate_range("2010-01-04", "2010-01-15")
    dates = np.concatenate([np.repeat(d, 3 if d == pd.Timestamp("2010-01-15") else 100)
                            for d in days])                  # el viernes 15 cotiza poco
    wd = weekly_dates(pd.Series(dates))
    assert list(wd) == [pd.Timestamp("2010-01-08"), pd.Timestamp("2010-01-14")]


def test_performance_metrics():
    days = pd.bdate_range("2010-01-01", periods=505)
    nav = pd.Series(1.1 ** (np.arange(505) / 252), index=days)
    nav.iloc[300:] *= 0.8
    from agentic_trading.agent.simulator import Result
    res = Result(nav, pd.Series(0.0, index=days), pd.Series(0.0, index=days),
                 pd.Series(0.0, index=days), pd.Series(0, index=days), pd.DataFrame())
    p = performance(res)
    assert p["caída máxima"] == pytest.approx(-0.2, abs=1e-3) and p["rotación anual"] == 0
    years = (days[-1] - days[0]).days / 365.25
    assert p["rentabilidad anual"] == pytest.approx((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)


# ----------------------------------------------------------------------------- reglas del agente


def run(pf, days, orders):
    """Avanza la cartera; orders: {día: (je, w, stop_d)} se ordena al cierre de ese día."""
    navs = []
    for t in range(days):
        nav, _ = pf.run_day(t)
        navs.append(nav)
        if t in orders:
            je, w, d = orders[t]
            N = len(pf.n)
            sd = np.full(N, np.nan)
            if d is not None:
                for j, v in d.items():
                    sd[j] = v
            pf.order(np.array(je, int), np.array(w, float), sd, eligible=np.arange(N))
    return navs


def test_agent_stop_chosen_and_clipped():
    rules = agent_rules(cost=0.0)
    # compra 10 % a 100; d = 0,2 → stop 80; el día 2 el mínimo toca 79: vende a 80 (−2 %)
    m = Market.from_frames({"X": frame([100, 100, 100, 90, 90, 90], lo=[100, 100, 79, 90, 90, 90])})
    pf = Portfolio(m, rules)
    navs = run(pf, 6, {0: ([0], [0.10], {0: 0.2})})
    assert navs[-1] == pytest.approx(1 - 0.10 * 0.2) and len(pf.stops) == 1
    # d = 0,5 pedido con w = 10 %: se recorta a 0,03 / 0,10 = 0,3 → stop 70 (no salta a 79)
    pf = Portfolio(m, rules)
    run(pf, 2, {0: ([0], [0.10], {0: 0.5})})
    assert pf.stop[0] == pytest.approx(70.0)
    # con w ≤ 3 %, tope del 50 %: w = 2 %, d = 0,9 → 0,5 → stop 50
    pf = Portfolio(m, rules)
    run(pf, 2, {0: ([0], [0.02], {0: 0.9})})
    assert pf.stop[0] == pytest.approx(50.0)
    # sin d: d_max
    pf = Portfolio(m, rules)
    run(pf, 2, {0: ([0], [0.10], None)})
    assert pf.stop[0] == pytest.approx(70.0)
    np.testing.assert_allclose(max_stop_distance([0.01, 0.06, 0.3], rules), [0.5, 0.5, 0.1])


def test_agent_stop_moves_on_held_position():
    # compra 10 % a 100; sube a 120 y la mantiene con d = 0,1 desde el cierre: stop 108
    # (w = 0,1·1,2/1,02 ≈ 0,1176: d_max ≈ 0,255, así que 0,1 vale). Luego cae: vende a 108.
    m = Market.from_frames({"X": frame([100, 100, 120, 120, 100, 100],
                                       lo=[100, 100, 120, 120, 100, 100])})
    pf = Portfolio(m, agent_rules(cost=0.0))
    navs = run(pf, 6, {0: ([0], [0.10], {0: 0.3}), 2: ([], [], {0: 0.1})})
    assert pf.stops[0][2:] == (100.0, "apertura")           # abre a 100 < 108: hueco
    assert navs[-1] == pytest.approx(0.9 + 0.1 * 1.0)        # vendió a la apertura (100)
    pf = Portfolio(m, agent_rules(cost=0.0))
    run(pf, 3, {0: ([0], [0.10], {0: 0.3}), 2: ([], [], {0: 0.1})})
    assert pf.stop[0] == pytest.approx(108.0) and pf.pend_n[0] != pf.pend_n[0]  # sin operar


def test_hold_keeps_drifted_weight_without_trading():
    m = Market.from_frames({"X": frame([10, 10, 20, 20, 20, 20]), "Y": frame([10] * 6)})
    pf = Portfolio(m, agent_rules(cost=0.01))
    traded = []
    for t in range(6):
        _, tr = pf.run_day(t)
        traded.append(tr)
        if t == 0:
            pf.order(np.array([0, 1]), np.array([0.5, 0.5]), eligible=[0, 1])
        elif t == 2:
            pf.order(np.array([], int), np.array([]), eligible=[0, 1])      # mantener las dos
    assert traded[3:] == [0.0, 0.0, 0.0]
    w = pf.weights()
    assert w[0] / w[1] == pytest.approx(2.0)                 # X conserva lo que le dan los precios


def test_min_trade_skips_small_orders():
    m = Market.from_frames({"X": frame([10] * 6)})
    pf = Portfolio(m, agent_rules(cost=0.0))
    traded = []
    for t in range(6):
        _, tr = pf.run_day(t)
        traded.append(tr)
        if t == 0:
            pf.order(np.array([0]), np.array([0.10]), eligible=[0])
        elif t == 2:
            pf.order(np.array([0]), np.array([0.104]), eligible=[0])    # Δ 0,4 % < 0,5 %: no
        elif t == 3:
            pf.order(np.array([0]), np.array([0.106]), eligible=[0])    # Δ 0,6 %: sí
    assert traded[1] == pytest.approx(0.10) and traded[3] == 0.0
    assert traded[4] == pytest.approx(0.006)
    # sin la regla (líneas base), la orden pequeña sí se ejecuta
    pf = Portfolio(m, Rules(cost=0.0))
    for t in range(4):
        _, tr = pf.run_day(t)
        if t == 0:
            pf.order(np.array([0]), np.array([0.10]))
        elif t == 2:
            pf.order(np.array([0]), np.array([0.104]))
    assert tr == pytest.approx(0.004)


@pytest.mark.parametrize("to, expected", [("cash", (0.6, 0.395, 0.0, 0.005)),
                                          ("spread", (0.6 / 0.995, 0.395 / 0.995, 0.0, 0.0))])
def test_min_weight_to_cash_or_spread(to, expected):
    # X 60 %, Y 39,5 %, Z 0,5 % → Z a 0; su peso va a la liquidez o se reparte entre X e Y
    m = Market.from_frames({s: frame([10] * 6) for s in "XYZ"})
    pf = Portfolio(m, agent_rules(cost=0.0, min_weight_to=to))
    run(pf, 3, {0: ([0, 1, 2], [0.6, 0.395, 0.005], None)})
    w = pf.weights()
    np.testing.assert_allclose([*w, pf.cash / pf.nav], expected, atol=1e-9)


def test_close_below_min_even_if_smaller_than_min_trade():
    # X cae al 0,4 % del capital al mantenerla: la regla del 1 % la cierra aunque Δ < 0,5 %
    # (sin stops, para que no salte antes)
    m = Market.from_frames({"X": frame([10, 10, 0.4, 0.4, 0.4, 0.4]), "Y": frame([10] * 6)})
    pf = Portfolio(m, agent_rules(cost=0.0, stop_loss=None))
    run(pf, 5, {0: ([0, 1], [0.1, 0.9], None), 2: ([], [], None)})
    assert pf.n[0] == 0 and pf.n[1] > 0
    # sin predicción ese día (no elegible), se mantiene
    pf = Portfolio(m, agent_rules(cost=0.0, stop_loss=None))
    for t in range(5):
        pf.run_day(t)
        if t == 0:
            pf.order(np.array([0, 1]), np.array([0.1, 0.9]), eligible=[0, 1])
        elif t == 2:
            pf.order(np.array([], int), np.array([]), eligible=[1])
    assert pf.n[0] > 0


def test_per_asset_costs_and_buy_tax():
    """Modelo de costes por activo (23-11-2026): c por activo en compras y ventas, impuesto solo
    en las compras; con c igual para todos y sin impuesto, lo mismo que el c único."""
    frames = {"X": frame([10, 10, 10, 10, 10, 10]), "Y": frame([20, 20, 20, 20, 20, 20])}
    m = Market.from_frames(frames)
    days = [DAYS[0], DAYS[2]]

    def pol(state):
        return pd.Series({"X": 0.5, "Y": 0.5} if state.date == DAYS[0] else {"X": 0.0, "Y": 0.0})

    base = simulate(m, pol, days, reference_rules(0.01))
    same = simulate(m, pol, days, Rules(cost=0.0, min_weight=None, stop_loss=None,
                                        cost_asset=np.array([0.01, 0.01])))
    np.testing.assert_allclose(same.nav, base.nav, rtol=1e-12)
    np.testing.assert_allclose(same.costs, base.costs, rtol=1e-12, atol=1e-15)
    r = Rules(cost=0.0, min_weight=None, stop_loss=None, cost_asset=np.array([0.01, 0.002]),
              buy_tax=np.array([0.005, 0.0]))
    res = simulate(m, pol, days, r)
    # compra: X paga 1 % + 0,5 %, Y 0,2 %; con 0,5 de peso cada una, sin pasar de la caja
    q = min(1.0, 1.0 / (0.5 * 1.015 + 0.5 * 1.002))
    buy = 0.5 * q * 0.015 + 0.5 * q * 0.002
    sell = 0.5 * q * 0.01 + 0.5 * q * 0.002              # ventas, sin impuesto
    assert res.costs.iloc[1] == pytest.approx(buy)
    assert res.costs.iloc[3] == pytest.approx(sell)
    assert res.nav.iloc[-1] == pytest.approx(1.0 - buy - sell)
    pf = Portfolio(m, r)
    assert pf.c_sell.tolist() == [0.01, 0.002] and pf.c_buy.tolist() == [0.015, 0.002]
    with pytest.raises(ValueError):
        Portfolio(m, Rules(cost_asset=np.array([0.01])))
