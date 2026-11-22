"""¿Por qué la regla fija del entorno del agente da +21,7 % en 2016–19 y B2 (logs/baseline_b2.md)
+12,2 %? (22-11-2026). Las dos son la misma regla (las 20 de mayor μ_5 en unidades de σ_ventana, a
partes iguales; una sale al caer fuera de las 60 primeras; decisión semanal) y difieren en:

- predicciones: rel63 (B2) o final20 (agente);
- universo: sin cripto (B2) o con BTC y ETH (agente);
- reglas: las de la línea base (B2: stop fijo P_medio·(1 − 0,03/w), sin tamaño mínimo de
  operación) o las del agente (stop en d_max desde el último cierre, recalculado en cada decisión;
  órdenes < 0,5 % sin ejecutar; cierre de restos < 1 %);
- medición: tramo 2016–19 de una simulación de 2000–2019 (B2) o episodio que empieza en
  liquidez el primer día de 2016 (agente).

Corre la regla (agent/evaluation.rule_order) en el entorno del agente con todas las combinaciones,
con c = 0,1 %, y da rentabilidad anual, Sharpe y rotación en 2016–19 (y 2000–19 para la medición
larga). Después, sin simulador (solo acciones, días de baselines.weekly_dates): el top 20 de μ_5
de cada ejecución, su log-retorno a 5 velas sobre la media del día y su rango medio en σ_ventana,
momentum (12 meses menos el último) y reversión de 1 mes; la diferencia semanal entre los dos top
20, con su t (sin solapamiento: decisiones semanales y retorno a 5 velas). Escribe
logs/rule_gap.md.

Uso: python scripts/rule_gap.py
"""

import itertools

import numpy as np
import pandas as pd

from agentic_trading.agent.data import Panel, build_panel
from agentic_trading.agent.env import TradingEnv
from agentic_trading.agent.evaluation import rule_order
from agentic_trading.agent.simulator import Result, Rules, agent_rules, performance
from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import asset_info

A, B = "2016-01-01", "2019-12-31"


def without_crypto(p: Panel) -> Panel:
    info = asset_info(p.market.symbols, pd.read_parquet(DATA_DIR / "universe.parquet"))
    eq = info["asset_class"].to_numpy() == "equity"
    elig, feats, zs = [], [], []
    for e, f, z in zip(p.elig, p.feats, p.z_mu5):
        m = eq[e]
        elig.append(e[m])
        feats.append(f[m])
        zs.append(z[m])
    return Panel(p.market, p.dec_t, p.dec_dates, elig, feats, zs, p.close_ffill, p.norm,
                 p.fingerprint, p.ew_logret)


def crypto_share(p: Panel, log, a=A, b=B) -> float:
    info = asset_info(p.market.symbols, pd.read_parquet(DATA_DIR / "universe.parquet"))
    cr = info["asset_class"].to_numpy() != "equity"
    v = [w[cr].sum() for d, w in log if pd.Timestamp(a) <= d <= pd.Timestamp(b)]
    return float(np.mean(v)) if v else np.nan


def run(p: Panel, rules: Rules, start) -> tuple[Result, list]:
    k0, k1 = p.k_range(start, B)
    env = TradingEnv(p, rules, n_steps=k1 - 1 - k0)
    env.record = True
    obs = env.reset(k0)
    days, log, done = [(p.dec_t[k0], 1.0, 0.0)], [], False
    while not done:
        obs, _, done, info = env.step_order(rule_order(obs, rules))
        days += info["days"]
        log.append((p.market.dates[p.dec_t[env.k]], env.pf.weights()))
    t, nav, traded = map(np.array, zip(*days))
    idx = p.market.dates[t]
    z = pd.Series(0.0, index=idx)
    res = Result(pd.Series(nav, index=idx), pd.Series(traded, index=idx), z, z, z,
                 pd.DataFrame(columns=["date", "symbol", "price", "how"]))
    res.n_stops = env.pf.n_stops
    return res, log


def tails(runs=("rel63", "final20"), n: int = 20) -> list[str]:
    from agentic_trading.agent.baselines import weekly_dates
    from agentic_trading.data.dataset import WindowData
    d = WindowData(DATA_DIR / "dataset")
    info = asset_info(d.meta["symbols"], pd.read_parquet(DATA_DIR / "universe.parquet"))
    lc, sig = np.asarray(d.arr["log_close"]), np.asarray(d.arr["sigma"])
    m = None
    for run in runs:
        pr = pd.concat([pd.read_parquet(p, columns=["sample", "mu_5"]) for p in
                        sorted((DATA_DIR / "predictions" / run).glob("cv[0-9].parquet"))])
        pr = pr.rename(columns={"mu_5": run})
        m = pr if m is None else m.merge(pr, on="sample")
    smp = d.samples.iloc[m["sample"].to_numpy()]
    m["date"], m["asset"], row = smp["date"].to_numpy(), smp["asset"].to_numpy(), smp["row"].to_numpy()
    m = m[info["asset_class"].to_numpy()[m["asset"]] == "equity"].copy()
    row = d.samples["row"].to_numpy()[m["sample"].to_numpy()]
    m["y"] = np.asarray(d.arr["y_5"])[row] * sig[row]
    m["σ_ventana"], m["momentum"] = sig[row], lc[row - 21] - lc[row - 251]
    m["reversión"] = lc[row - 21] - lc[row]
    m = m[m["date"].isin(weekly_dates(m["date"]))]
    L = ["## Top 20 de μ_5 sin simulador (acciones, decisiones semanales)", "",
         ("| periodo | ejecución | log-retorno a 5 velas sobre la media (pb/semana, t) | "
         "rango σ_ventana | rango momentum | rango reversión |"), "|---|---|---|---|---|---|"]
    diffs = []
    for per, a, b in (("2000–15", "2000-01-01", "2015-12-31"), ("2016–19", A, B)):
        q = m[(m["date"] >= a) & (m["date"] <= b)].copy()
        g = q.groupby("date")
        q["ym"] = q["y"] - g["y"].transform("mean")
        for c in ("σ_ventana", "momentum", "reversión"):
            q["r_" + c] = g[c].rank(pct=True)
        wk = {}
        for run in runs:
            top = q[g[run].rank(ascending=False, method="first") <= n]
            wk[run] = top.groupby("date")["ym"].mean()
            se = wk[run].std() / np.sqrt(len(wk[run]))
            L.append(f"| {per} | {run} | {1e4 * wk[run].mean():+.1f} ({wk[run].mean() / se:.1f}) | "
                     f"{top['r_σ_ventana'].mean():.2f} | {top['r_momentum'].mean():.2f} | "
                     f"{top['r_reversión'].mean():.2f} |")
        dd = wk[runs[1]] - wk[runs[0]]
        diffs.append(f"- {per}: {runs[1]} − {runs[0]}, {1e4 * dd.mean():+.1f} pb/semana "
                     f"(t {dd.mean() / (dd.std() / np.sqrt(len(dd))):.2f}, {len(dd)} semanas).")
    return L + ["", "Diferencia semanal entre los dos top 20:", "", *diffs]


def main() -> None:
    rules = {"agente": agent_rules(0.001), "línea base": Rules(cost=0.001)}
    rows = []
    for pred in ("rel63", "final20"):
        full = build_panel(pred, freq="weekly")
        for crypto, (rname, r), meas in itertools.product(("con", "sin"), rules.items(),
                                                           ("episodio 2016", "tramo de 2000–19")):
            p = full if crypto == "con" else without_crypto(full)
            res, log = run(p, r, A if meas == "episodio 2016" else p.dec_dates[0])
            m = performance(res, A, B)
            mall = performance(res) if meas != "episodio 2016" else {}
            rows.append({"predicciones": pred, "cripto": crypto, "reglas": rname,
                         "medición": meas, "2016–19": m["rentabilidad anual"],
                         "Sharpe": m["Sharpe"], "rotación": m["rotación anual"],
                         "peso en cripto": crypto_share(p, log),
                         "2000–19": mall.get("rentabilidad anual", np.nan)})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    L = ["# Regla fija: B2 frente al entorno del agente (2016–19, c = 0,1 %)", "",
         "Ver el docstring de scripts/rule_gap.py.", "",
         ("| predicciones | cripto | reglas | medición | 2016–19 | Sharpe | rotación anual | "
         "peso medio en cripto | 2000–19 |"), "|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        L.append(f"| {r.predicciones} | {r.cripto} | {r.reglas} | {r.medición} | "
                 f"{100 * r._5:+.1f} % | {r.Sharpe:.2f} | {r.rotación:.0f} | "
                 f"{100 * r._8:.1f} % | "
                 + (f"{100 * r._9:+.1f} %" if np.isfinite(r._9) else "—") + " |")
    L += ["", *tails()]
    (ROOT / "logs" / "rule_gap.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
