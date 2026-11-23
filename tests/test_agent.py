import numpy as np
import pandas as pd
import pytest
import torch

from agentic_trading.agent.data import Panel, equal_weight_logret
from agentic_trading.agent.env import TradingEnv
from agentic_trading.agent.policy import AgentNet, act, obs_tensors
from agentic_trading.agent.ppo import PPOConfig, Trainer, run_policy
from agentic_trading.agent.simulator import Market, agent_rules


def toy_panel(n_assets=40, n_days=400, seed=0) -> Panel:
    """Panel sintético: precios aleatorios y predicciones con algo de señal."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2005-01-03", periods=n_days)
    frames, rets = {}, rng.normal(0.0003, 0.02, (n_days, n_assets))
    for i in range(n_assets):
        c = 50 * np.exp(np.cumsum(rets[:, i]))
        o = c * np.exp(rng.normal(0, 0.005, n_days))
        frames[f"S{i}"] = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01,
                                        "low": np.minimum(o, c) * 0.99, "close": c}, index=days)
    m = Market.from_frames(frames)
    elig, feats, zs = [], [], []
    for _ in range(n_days):
        e = np.sort(rng.choice(n_assets, size=int(rng.integers(25, n_assets + 1)), replace=False))
        x = rng.normal(size=(len(e), 9)).astype(np.float32)
        elig.append(e.astype(np.int64))
        feats.append(x)
        zs.append(((x[:, 0] - x[:, 0].mean()) / x[:, 0].std()).astype(np.float32))
    cf = pd.DataFrame(m.close).ffill().to_numpy()
    dec_t = np.arange(n_days)
    return Panel(m, dec_t, days, elig, feats, zs, cf, {}, {},
                 equal_weight_logret(cf, dec_t, elig))


@pytest.fixture(scope="module")
def panel():
    return toy_panel()


def walk(panel, n=30, seed=0, explore=True):
    """Pasos con una red al azar; devuelve (obs, order, rec) de cada paso."""
    torch.manual_seed(seed)
    net, rules = AgentNet(), agent_rules()
    env = TradingEnv(panel, rules, n_steps=n)
    obs, rng, out = env.reset(5), np.random.default_rng(seed), []
    for _ in range(n):
        order, rec, _, _ = act(net, obs, rules, rng, explore)
        out.append((obs, order, rec))
        obs, _, done, _ = env.step_order(order)
        if done:
            break
    return out


def test_shapes(panel):
    net = AgentNet()
    env = TradingEnv(panel, agent_rules())
    o1, o2 = env.reset(3), env.reset(10)
    h = net.heads(**obs_tensors([o1, o2]))
    n = max(len(o1.elig), len(o2.elig))
    for k in ("score", "alpha", "beta", "trade_logit"):
        assert h[k].shape == (2, n)
    assert h["liq"].shape == (2,) and h["value"].shape == (2,)
    order, rec, lp, v = act(net, o1, agent_rules(), np.random.default_rng(0))
    assert len(order["je"]) == len(order["w"]) == order["n_trade"]
    assert len(order["cand"]) == len(order["d"]) and np.isfinite(lp) and np.isfinite(v)
    assert len(rec.cand) == len(rec.trade) == len(rec.s) == len(rec.frac) == len(rec.rel)


def test_weights_sum_to_one_and_stop_limit(panel):
    rules = agent_rules()
    for obs, order, rec in walk(panel, 40):
        held_c = obs.held[rec.cand]
        kept = obs.weight[rec.cand][held_c & ~rec.trade].sum() + obs.room_kept
        assert order["w"].sum() + order["cash"] + kept == pytest.approx(1.0, abs=1e-9)
        assert (order["w"] >= 0).all() and order["cash"] >= 0
        # w·d ≤ 3 % (w: el objetivo si se opera, el actual si se mantiene) y d ≤ 50 %
        w_new = np.where(rec.trade, 0.0, obs.weight[rec.cand])
        w_new[rec.trade] = order["w"]
        wd = w_new[rec.rel] * order["d"]
        assert (wd <= rules.stop_loss + 1e-12).all() and (order["d"] <= 0.5 + 1e-12).all()
        assert (order["d"] > 0).all()


def test_candidates_include_held_positions(panel):
    seen = 0
    for obs, order, rec in walk(panel, 40):
        held = np.flatnonzero(obs.held)
        assert set(held) <= set(rec.cand)
        seen += len(held)
    assert seen > 0                                   # hubo posiciones que comprobar


def test_episode_keeps_cash_and_units_nonnegative(panel):
    rules = agent_rules()
    torch.manual_seed(1)
    net = AgentNet()
    env = TradingEnv(panel, rules, n_steps=30)
    obs, rng = env.reset(0), np.random.default_rng(1)
    for _ in range(30):
        order, *_ = act(net, obs, rules, rng)
        # stops de las posiciones mantenidas: se fijan al decidir, con su peso actual
        obs, _, done, _ = env.step_order(order)
        pf = env.pf
        if done:
            break
    assert (pf.n >= 0).all() and pf.cash >= -1e-9


def test_permutation_invariance(panel):
    torch.manual_seed(0)
    net, rules = AgentNet(), agent_rules()
    env = TradingEnv(panel, rules)
    obs = env.reset(7)
    for _ in range(3):                                # con alguna posición abierta
        order, *_ = act(net, obs, rules, np.random.default_rng(0), explore=False)
        obs, *_ = env.step_order(order)
    a, *_ = act(net, obs, rules, np.random.default_rng(0), explore=False)
    perm = np.random.default_rng(3).permutation(len(obs.elig))
    o2 = type(obs)(obs.elig[perm], obs.x[perm], obs.z_mu5[perm], obs.held[perm],
                   obs.weight[perm], obs.port, obs.room_kept)
    b, *_ = act(net, o2, rules, np.random.default_rng(0), explore=False)
    wa, wb = dict(zip(a["je"], a["w"])), dict(zip(b["je"], b["w"]))
    assert wa.keys() == wb.keys() and len(wa) > 0
    for k, v in wa.items():
        assert v == pytest.approx(wb[k], rel=1e-5)
    da, db = dict(zip(a["cand"], a["d"])), dict(zip(b["cand"], b["d"]))
    assert da.keys() == db.keys()
    assert all(da[k] == pytest.approx(db[k], rel=1e-5) for k in da)


def test_deterministic_with_seed(panel):
    cfg = PPOConfig(n_envs=2, rollout=8, minibatch=8, epochs=1, episode=50, seed=3)

    def run():
        tr = Trainer(panel, cfg)
        rows = [tr.train_step() for _ in range(2)]
        return rows, [p.detach().clone() for p in tr.net.parameters()]
    r1, p1 = run()
    r2, p2 = run()
    for a, b in zip(r1, r2):
        for k in a:
            if not k.startswith("t_"):
                assert a[k] == b[k], k
    assert all(torch.equal(x, y) for x, y in zip(p1, p2))


def test_run_policy_deterministic_and_result(panel):
    torch.manual_seed(0)
    net = AgentNet()
    days = panel.dec_dates
    r1, d1 = run_policy(panel, net, days[100], days[200], agent_rules())
    r2, _ = run_policy(panel, net, days[100], days[200], agent_rules())
    assert r1.nav.equals(r2.nav) and r1.nav.iloc[0] == 1.0 and len(d1["npos"]) == 100


def test_equal_weight_logret_by_hand():
    close = np.array([[10.0, 20.0, 5.0], [11.0, 18.0, 5.0], [11.0, 18.0, 10.0]])
    elig = [np.array([0, 1]), np.array([0, 1, 2]), np.array([2])]
    r = equal_weight_logret(close, np.array([0, 1, 2]), elig)
    # k = 0: (1,1 + 0,9) / 2 = 1; k = 1: (1 + 1 + 2) / 3; el último, NaN
    np.testing.assert_allclose(r[:2], [0.0, np.log(4 / 3)])
    assert np.isnan(r[2])


def test_relative_reward_only_changes_reward(panel):
    """Mismas acciones con las dos recompensas en la primera recogida (misma semilla); la
    recompensa relativa es la absoluta menos ew_logret."""
    cfg = PPOConfig(n_envs=1, rollout=5, minibatch=5, epochs=1, episode=50, seed=1)
    a = Trainer(panel, cfg)
    b = Trainer(panel, PPOConfig(**{**cfg.__dict__, "reward": "relative"}))
    ra, rb = a.train_step(), b.train_step()
    assert ra["reward_mean"] == rb["reward_mean"]
    assert rb["reward_rel_mean"] == pytest.approx(ra["reward_rel_mean"])
    assert ra["turnover_day"] == rb["turnover_day"]


def test_trade_bias_sets_initial_trade_probability(panel):
    env = TradingEnv(panel, agent_rules())
    o = env.reset(3)
    for bias, p in ((0.0, 0.5), (-2.0, 1 / (1 + np.exp(2.0)))):
        torch.manual_seed(0)
        h = AgentNet(trade_bias=bias).heads(**obs_tensors([o]))
        assert torch.sigmoid(h["trade_logit"]).mean().item() == pytest.approx(p, abs=0.01)


def test_component_ratio_runs_and_is_deterministic(panel):
    cfg = PPOConfig(n_envs=2, rollout=8, minibatch=8, epochs=2, episode=50, seed=4,
                    ratio="component", reward="relative", trade_bias=-2.0)

    def run():
        tr = Trainer(panel, cfg)
        rows = [tr.train_step() for _ in range(2)]
        return rows, [p.detach().clone() for p in tr.net.parameters()]
    (r1, p1), (r2, p2) = run(), run()
    assert all(torch.equal(x, y) for x, y in zip(p1, p2))
    assert r1[0]["kl"] == pytest.approx(0.0, abs=1e-6) or r1[0]["kl"] == r2[0]["kl"]
    assert all(np.isfinite(r[k]) for r in r1 for k in ("loss_pi", "loss_v", "kl", "clipfrac"))


def test_old_components_match_joint_logprob(panel):
    """Las log-probabilidades por componente de partida suman la conjunta guardada al actuar."""
    tr = Trainer(panel, PPOConfig(n_envs=1, rollout=6, minibatch=4, episode=50, seed=2,
                                  ratio="component"))
    data, _ = tr.collect()
    old = tr._old_components(data)
    for d, (a, b, c, liq) in zip(data, old):
        assert float(a.sum() + b.sum() + c.sum() + liq) == pytest.approx(d[2], abs=1e-4)


def test_synthetic_signal_has_target_correlation():
    from agentic_trading.agent.data import synthetic_signal
    rng = np.random.default_rng(0)
    dates = np.repeat(np.arange(200), 300)
    y = rng.normal(size=len(dates)) + np.repeat(rng.normal(size=200), 300)   # + efecto del día
    s = synthetic_signal(y, dates, 0.1, seed=1)
    ic = np.mean([np.corrcoef(s[dates == d], y[dates == d])[0, 1] for d in range(200)])
    assert ic == pytest.approx(0.1, abs=0.01)
    assert np.array_equal(s, synthetic_signal(y, dates, 0.1, seed=1))


def test_rule_order_respects_rules(panel):
    from agentic_trading.agent.evaluation import rule_order, run_rule
    rules = agent_rules()
    env = TradingEnv(panel, rules, n_steps=30)
    obs = env.reset(5)
    for _ in range(30):
        order = rule_order(obs, rules, n=10, exit_rank=20)
        assert order["w"].sum() <= 1 + 1e-12 and (order["w"] >= 0).all()
        assert (order["w"][:len(order["cand"])] * order["d"] <= 0.03 + 1e-12).all()
        obs, _, done, _ = env.step_order(order)
        assert (env.pf.n > 0).sum() <= 10
        if done:
            break
    perf = run_rule(panel, panel.dec_dates[10], panel.dec_dates[200], rules, n=10, exit_rank=20)
    assert np.isfinite(perf["rentabilidad anual"])


def test_policy_runs_and_paired(panel):
    from agentic_trading.agent.evaluation import paired, policy_runs, summarize
    torch.manual_seed(0)
    net = AgentNet()
    a, b = panel.dec_dates[10], panel.dec_dates[120]
    r1 = policy_runs(panel, net, a, b, agent_rules(), 20, seeds=[1, 2, 3])
    r2 = policy_runs(panel, net, a, b, agent_rules(), 20, seeds=[1, 2, 3])
    assert r1 == r2                                   # determinista con las semillas
    assert r1["stochastic"][0] != r1["stochastic"][1]
    s = summarize(r1["stochastic"])
    assert s["rentabilidad anual"][1] > 0
    d, se, _ = paired(r1["stochastic"], r2["stochastic"], "Sharpe")
    assert d == 0 and se == 0


def test_rule_record_reproduces_rule_order(panel):
    """Clonado (agent/bc.py): con las acciones objetivo, el reparto de la política (softmax de las
    que se operan y la liquidez, como en act) da los pesos de la regla: ~1/20 en las que se
    quedan o se compran y < 1 % (la regla del 1 % las cierra) en las que se venden."""
    from agentic_trading.agent.bc import rule_record
    from agentic_trading.agent.evaluation import rule_order
    rules = agent_rules()
    env = TradingEnv(panel, rules, n_steps=40)
    obs = env.reset(5)
    for _ in range(40):
        order = rule_order(obs, rules)
        rec = rule_record(obs)
        held_c = obs.held[rec.cand]
        assert rec.trade[held_c].all()                      # todas las posiciones operan
        room = max(0.0, 1.0 - obs.room_kept)
        logits = np.r_[rec.s[rec.trade], rec.liq]
        e = np.exp(logits - logits.max())
        w = e / e.sum() * room
        w_by = dict(zip(obs.elig[rec.cand[rec.trade]], w[:-1]))
        keep = set(order["je"][order["w"] > 0])
        sold = set(order["je"][order["w"] == 0])
        assert keep == {j for j, x in w_by.items() if x >= rules.min_weight}
        # pesos casi iguales: κ·z con κ = 0,05; con 25–40 activos, el top 20 abarca más z que con
        # ~500, así que ±15 % del peso
        assert all(abs(w_by[j] - 1 / 20) < 0.0075 for j in keep)
        assert all(w_by[j] < rules.min_weight for j in sold)
        obs, _, done, _ = env.step_order(order)
        if done:
            break


def test_kl_to_reference_and_schedule(panel):
    import copy

    from agentic_trading.agent.policy import kl_to, record_tensors
    steps = walk(panel, n=6)
    torch.manual_seed(0)
    net = AgentNet()
    ob = obs_tensors([s[0] for s in steps])
    rc = record_tensors([s[2] for s in steps])
    ref = copy.deepcopy(net)
    assert float(kl_to(net, ref, ob, rc).detach().abs().max()) < 1e-6
    with torch.no_grad():
        ref.psi[-1].bias[0] += 0.3
    assert float(kl_to(net, ref, ob, rc).detach().min()) > 0
    cfg = PPOConfig(n_envs=2, rollout=8, minibatch=8, epochs=1, episode=50, seed=1,
                    ratio="component", kl_coef=1.0, kl_frac=0.5, kl_updates=4)
    tr = Trainer(panel, cfg, ref_net=copy.deepcopy(AgentNet()))
    rows = [tr.train_step() for _ in range(3)]
    assert [r["kl_coef"] for r in rows] == [1.0, 0.5, 0.0]
    assert np.isfinite(rows[0]["kl_ref"]) and "kl_ref" not in rows[2]
