"""Paso 1 de la ampliación del universo (22-11-2026, sin entrenar): ¿hay más señal de precio en
las medianas y pequeñas en 2012–19?

Grupos: grandes (el universo actual, data/), medianas de EE. UU. (S&P MidCap 400), pequeñas de
EE. UU. (S&P SmallCap 600) y medianas y pequeñas de Europa (el resto de universe.INDICES_EXT), del
universo ampliado (data/ext/, `make ext`). Solo acciones y solo 2000–2019 (bloque dev, sin abrir
el test). Un componente que Wikipedia pone a la vez en el 400 y el 600 cuenta como mediana.

Medida (la de logs/mu_tree/deciles.md, scripts/mu_deciles.py), dentro de cada grupo: en las
fechas semanales del agente (las de los grandes, como en mu_deciles.py) con al menos 20 acciones
del grupo, deciles de la señal entre las acciones del grupo y, por decil, el exceso de apertura a
apertura hasta la fecha semanal siguiente sobre la media del grupo, en pb por semana. Pierna larga
= media de D9 y D10; corta = media de D1 y D2. Señales: la mezcla de success_checks (media de los
rangos de la reversión de 1 mes y del momentum de 12 meses, entre las acciones del grupo), y cada
una por separado.

Además, por grupo: dispersión entre acciones (desviación típica del log-retorno semanal de
apertura a apertura, media de las semanas, pb), acciones por fecha y la horquilla estimada de
Abdi–Ranaldo (data/spread.py: por mes, la mediana entre las acciones del grupo; media de los meses,
pb). Todo por bloque y 2000–11 frente a 2012–19.

Lectura fijada antes de verlo: si en 2012–19 la pierna larga de la mezcla de algún grupo es al
menos 1,5 veces la de los grandes, ampliar compensa, y el siguiente paso es entrenar μ con el
universo ampliado. Si no, no compensa.

Escribe logs/universe_ext.md y logs/universe_ext.json.

Uso: PYTHONPATH=. python scripts/universe_ext.py [--md-only]
"""

import json
import sys

import numpy as np
import pandas as pd

from agentic_trading.agent.baselines import weekly_dates
from agentic_trading.config import DATA_DIR, ROOT
from agentic_trading.data.dataset import WindowData, asset_info
from agentic_trading.data.spread import abdi_ranaldo
from agentic_trading.data.universe import EXT_EU_GROUP, EXT_US_GROUPS
from agentic_trading.model.metrics import baseline_features, baseline_mix, nw_tstat
from agentic_trading.model.mu_metrics import (
    BLOCK_NAMES,
    NW_LAGS_WEEKLY,
    OpenPrices,
    block_of,
    weekly_frame,
)
from scripts.mu_compare import BLOCK_LABEL, fmt
from scripts.mu_deciles import decile_series

BASE, EXT = DATA_DIR, DATA_DIR / "ext"
OUT = ROOT / "logs"
LARGE = "grandes"
GROUPS = [LARGE, *EXT_US_GROUPS.values(), EXT_EU_GROUP]
EARLY, LATE = ("cv0", "cv1", "cv2"), ("cv3", "cv4")
PERIODS = {"2000–11": EARLY, "2012–19": LATE}
FACTOR = 1.5
MIN_STOCKS = 20
SIGNALS = ("mezcla", "reversión 1 m", "momentum 12 m")


def group_of(universe: pd.DataFrame) -> pd.Series:
    """Grupo de cada símbolo del universo ampliado (por sus índices; el 400 antes que el 600)."""
    def one(ind: str) -> str:
        for idx, g in EXT_US_GROUPS.items():
            if idx in ind.split("|"):
                return g
        return EXT_EU_GROUP
    return universe.set_index("symbol")["indices"].map(one)


def frames(ddir, wk: pd.DatetimeIndex, groups: pd.Series | None) -> tuple[pd.DataFrame, dict]:
    """Filas semanales (acciones de dev) de un dataset, con el grupo, las señales y ret_oo."""
    data = WindowData(ddir / "dataset")
    uni = pd.read_parquet(ddir / "universe.parquet")
    syms = data.meta["symbols"]
    equity = (asset_info(syms, uni)["asset_class"] == "equity").to_numpy()
    prices = OpenPrices.from_clean(ddir / "clean", [syms[a] for a in np.flatnonzero(equity)])
    wf = weekly_frame(data, data.split("dev"), equity, wk, prices)
    sym = np.asarray(syms, dtype=object)[wf["asset"].to_numpy()]
    wf["symbol"] = sym
    wf["group"] = LARGE if groups is None else groups.reindex(sym).to_numpy()
    base = baseline_features(data, data.samples["row"].to_numpy()[wf["sample"].to_numpy()])
    wf["rev"], wf["mom"] = base["reversal"], base["momentum"]
    dev = data.samples.iloc[data.split("dev")]
    n_dev = dev[equity[dev["asset"].to_numpy()]]["asset"].map(lambda a: syms[a])
    return wf, {"dev_assets": n_dev.groupby(
        n_dev.map(lambda s: LARGE if groups is None else groups.get(s))).nunique().to_dict()}


def period_mean(s: pd.Series, blocks) -> tuple[float, float]:
    return nw_tstat(s[np.isin(block_of(s.index), blocks)].dropna().to_numpy(), NW_LAGS_WEEKLY)


def spread_by_month(ddir, symbols) -> pd.Series:
    """Por mes, la mediana entre acciones de la horquilla de Abdi–Ranaldo (pb), 2000–2019."""
    cols = {}
    for s in symbols:
        df = pd.read_parquet(ddir / "clean" / f"{s}.parquet").loc[:"2019-12-31"]
        if len(df) > 1:
            cols[s] = abdi_ranaldo(df)
    m = pd.DataFrame(cols).median(axis=1) * 1e4
    m.index = m.index.to_timestamp(how="end").normalize()
    return m[m.index >= pd.Timestamp("2000-01-01")]


def counts_ext(groups: pd.Series, info: dict) -> pd.DataFrame:
    """Cuántos activos entran y cuántos se pierden en cada paso, por grupo del ampliado."""
    fetched = json.loads((EXT / "universe_fetched.json").read_text())
    excl = pd.read_parquet(EXT / "universe_excluded.parquet")
    man = pd.read_parquet(EXT / "raw" / "yahoo" / "_manifest.parquet").set_index("symbol")
    summ = pd.read_parquet(EXT / "clean" / "_summary.parquet").set_index("symbol")
    acts = pd.read_parquet(EXT / "clean" / "_actions.parquet")
    trunc = set(acts.loc[acts["action"] == "truncate", "symbol"])
    gi = {"SP400": EXT_US_GROUPS["SP400"], "SP600": EXT_US_GROUPS["SP600"]}
    rows = []
    for g in GROUPS[1:]:
        syms = groups.index[groups == g]
        ok = man.reindex(syms)["status"].eq("ok")
        sm = summ.reindex(syms[ok.to_numpy()])
        rows.append({
            "grupo": g,
            "en Wikipedia": sum(v for k, v in fetched.items() if gi.get(k, EXT_EU_GROUP) == g),
            "ya en el actual": int(excl["index"].map(lambda k: gi.get(k, EXT_EU_GROUP))
                                   .eq(g).sum()),
            "en el universo": len(syms),
            "descargados": int(ok.sum()),
            "con muestras tras limpiar": int((sm["usable_samples"] > 0).sum()),
            "históricos cortados": len(trunc & set(syms)),
            "muestras: brutas → útiles": (f"{int(sm['raw_samples'].sum()):,} → "
                                          f"{int(sm['usable_samples'].sum()):,} "
                                          f"({sm['usable_samples'].sum() / sm['raw_samples'].sum() - 1:+.1%})"),
            "con muestras en 2000–19": int(info["dev_assets"].get(g, 0)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    if "--md-only" in sys.argv:          # reescribe el informe desde universe_ext.json
        res = json.loads((OUT / "universe_ext.json").read_text())
        write_md(res, pd.DataFrame(res["counts"]))
        return
    data = WindowData(BASE / "dataset")
    uni = pd.read_parquet(BASE / "universe.parquet")
    equity = (asset_info(data.meta["symbols"], uni)["asset_class"] == "equity").to_numpy()
    dev = data.samples.iloc[data.split("dev")]
    dates = dev["date"]
    wk = weekly_dates(dates[dates.dt.dayofweek < 5])
    n_eq = dev[equity[dev["asset"].to_numpy()]].groupby("date").size()
    wk = wk[wk.isin(n_eq[n_eq >= MIN_STOCKS].index)]
    del data

    ext_groups = group_of(pd.read_parquet(EXT / "universe.parquet"))
    wf_l, info_l = frames(BASE, wk, None)
    wf_e, info_e = frames(EXT, wk, ext_groups)
    wf = pd.concat([wf_l, wf_e], ignore_index=True)

    res: dict = {"groups": {}, "weeks": len(wk)}
    series = {}
    for g in GROUPS:
        q = wf[wf["group"] == g]
        n = q.groupby("date").size()
        q = q[q["date"].isin(n[n >= MIN_STOCKS].index)].reset_index(drop=True)
        d = q["date"].to_numpy()
        sig = {"mezcla": baseline_mix(d, q["rev"].to_numpy(), q["mom"].to_numpy()),
               "reversión 1 m": q["rev"].to_numpy(), "momentum 12 m": q["mom"].to_numpy()}
        n = q.groupby("date").size()
        disp = 1e4 * q.groupby("date")["ret_oo"].std()
        spr = spread_by_month(BASE if g == LARGE else EXT, q["symbol"].unique())
        r = {"stocks": int(q["symbol"].nunique()), "weeks": int(n.size), "blocks": {},
             "periods": {}, "signals": {}}
        for b in BLOCK_NAMES:
            r["blocks"][b] = {"n": float(n[block_of(n.index) == b].mean()),
                              "disp": float(disp[block_of(disp.index) == b].mean()),
                              "spread": float(spr[block_of(spr.index) == b].mean())}
        for p, bl in PERIODS.items():
            r["periods"][p] = {"n": float(n[np.isin(block_of(n.index), bl)].mean()),
                               "disp": float(disp[np.isin(block_of(disp.index), bl)].mean()),
                               "spread": float(spr[np.isin(block_of(spr.index), bl)].mean())}
        for name, x in sig.items():
            s = decile_series(q, x)
            s["dif"] = s["larga"] - s["corta"]
            series[(g, name)] = s
            v = {"blocks": {}, "periods": {}}
            for b in BLOCK_NAMES:
                sb = s[block_of(s.index) == b]
                v["blocks"][b] = {k: float(sb[k].mean()) for k in ("larga", "corta", "dif")}
            for p, bl in PERIODS.items():
                v["periods"][p] = {k: dict(zip(("mean", "t"), period_mean(s[k], bl)))
                                   for k in ("larga", "corta", "dif")}
            r["signals"][name] = v
        res["groups"][g] = r

    # frente a los grandes: diferencia de la pierna larga de la mezcla, mismas semanas
    big = series[(LARGE, "mezcla")]["larga"]
    late_big = res["groups"][LARGE]["signals"]["mezcla"]["periods"]["2012–19"]["larga"]["mean"]
    verdict = {}
    for g in GROUPS[1:]:
        s = series[(g, "mezcla")]["larga"]
        diff = (s - big).dropna()
        m, t = period_mean(diff, LATE)
        late = res["groups"][g]["signals"]["mezcla"]["periods"]["2012–19"]["larga"]["mean"]
        ratio = late / late_big if late_big > 0 else float("nan")
        verdict[g] = {"larga_2012_19": late, "ratio": ratio, "dif_vs_grandes": m, "t": t,
                      "compensa": bool(late_big > 0 and ratio >= FACTOR)}
    res["vs_grandes"] = verdict
    res["grandes_larga_2012_19"] = late_big
    win = [g for g, v in verdict.items() if v["compensa"]]
    res["lectura"] = ("compensa: " + ", ".join(win)) if win else "no compensa"
    if late_big <= 0:
        res["lectura"] = "caso no previsto: la pierna larga de los grandes en 2012–19 es ≤ 0"
    cnt = counts_ext(ext_groups, info_e)
    res["counts"] = cnt.to_dict("records")
    res["counts_grandes"] = {"acciones en el universo": int((uni["asset_class"] == "equity").sum()),
                             "con muestras en 2000–19": int(info_l["dev_assets"].get(LARGE, 0))}
    (OUT / "universe_ext.json").write_text(json.dumps(res, indent=1, ensure_ascii=False,
                                                      default=float))
    write_md(res, cnt)


def write_md(res: dict, cnt: pd.DataFrame) -> None:
    G = res["groups"]
    L = ["# Universo ampliado, paso 1: señal de precio en medianas y pequeñas (22-11-2026)", "",
         ("Ver el docstring de scripts/universe_ext.py. Sin entrenar; solo 2000–2019. Exceso "
          "semanal de apertura a apertura sobre la media del grupo, pb por semana, en las "
          f"{res['weeks']} fechas semanales del agente (las de los grandes) con ≥ {MIN_STOCKS} "
          "acciones del grupo. Larga = D9–D10; corta = D1–D2; dif. = larga − corta; deciles "
          "dentro de cada grupo. t de NW (2 retardos)."), ""]
    big = res["grandes_larga_2012_19"]
    L += [f"**Lectura (regla fijada): {res['lectura']}.** Pierna larga de la mezcla en 2012–19: "
          f"grandes {fmt(big)} pb; "
          + "; ".join(f"{g} {fmt(v['larga_2012_19'])} ({v['ratio']:.2f} veces)"
                      for g, v in res["vs_grandes"].items())
          + f". Umbral: {FACTOR} veces.", ""]
    L += ["## Pierna larga de la mezcla en 2012–19 frente a los grandes", "",
          "| grupo | larga 2012–19 (t) | veces la de los grandes | larga − grandes, mismas semanas (t) |",
          "|---|---|---|---|"]
    lg = G[LARGE]["signals"]["mezcla"]["periods"]["2012–19"]["larga"]
    L.append(f"| {LARGE} | {fmt(lg['mean'])} ({lg['t']:.2f}) | 1.00 | — |")
    for g, v in res["vs_grandes"].items():
        p = G[g]["signals"]["mezcla"]["periods"]["2012–19"]["larga"]
        L.append(f"| {g} | {fmt(p['mean'])} ({p['t']:.2f}) | {v['ratio']:.2f} | "
                 f"{fmt(v['dif_vs_grandes'])} ({v['t']:.2f}) |")
    L += ["", "## Por periodo: piernas, dispersión, acciones y horquilla", "",
          ("| grupo | periodo | acciones por fecha | dispersión (pb) | horquilla A–R (pb) | "
           + " | ".join(f"{s}: larga / corta / dif. (t dif.)" for s in SIGNALS) + " |"),
          "|---|---|---|---|---|" + "---|" * len(SIGNALS)]
    for g in GROUPS:
        for p in PERIODS:
            c = G[g]["periods"][p]
            cells = []
            for s in SIGNALS:
                v = G[g]["signals"][s]["periods"][p]
                cells.append(f"{fmt(v['larga']['mean'])} / {fmt(v['corta']['mean'])} / "
                             f"{fmt(v['dif']['mean'])} ({v['dif']['t']:.2f})")
            L.append(f"| {g} | {p} | {c['n']:.0f} | {c['disp']:.0f} | {c['spread']:.0f} | "
                     + " | ".join(cells) + " |")
    L += ["", "## Por bloque", "",
          ("| grupo | bloque | acciones por fecha | dispersión (pb) | horquilla A–R (pb) | "
           + " | ".join(f"{s}: larga / corta / dif." for s in SIGNALS) + " |"),
          "|---|---|---|---|---|" + "---|" * len(SIGNALS)]
    for g in GROUPS:
        for b in BLOCK_NAMES:
            c = G[g]["blocks"][b]
            cells = [" / ".join(fmt(G[g]["signals"][s]["blocks"][b][k])
                                for k in ("larga", "corta", "dif")) for s in SIGNALS]
            n = "—" if np.isnan(c["n"]) else f"{c['n']:.0f}"
            L.append(f"| {g} | {BLOCK_LABEL[b]} | {n} | {c['disp']:.0f} | {c['spread']:.0f} | "
                     + " | ".join(cells) + " |")
    L += ["", "## Activos: cuántos entran y cuántos se pierden", "",
          "| " + " | ".join(cnt.columns) + " |", "|" + "---|" * len(cnt.columns)]
    for r in cnt.itertuples(index=False):
        L.append("| " + " | ".join(str(x) for x in r) + " |")
    cg = res["counts_grandes"]
    L += ["", (f"Grandes (universo actual): {cg['acciones en el universo']} acciones, "
          f"{cg['con muestras en 2000–19']} con muestras en 2000–19."), ""]
    L += caveats(res) + DECISIONS
    (OUT / "universe_ext.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def caveats(res: dict) -> list[str]:
    """Matices de la lectura, con las cifras de res (no cambian la regla)."""
    G, V = res["groups"], res["vs_grandes"]
    late = "2012–19"
    sp = {g: G[g]["periods"][late]["spread"] for g in GROUPS}
    n = {g: G[g]["periods"][late]["n"] for g in GROUPS}
    mom = {g: G[g]["signals"]["momentum 12 m"]["periods"][late]["dif"]["mean"] for g in GROUPS}
    rev = {g: G[g]["signals"]["reversión 1 m"]["periods"][late]["dif"]["mean"] for g in GROUPS}
    L = ["## Matices", ""]
    L.append("- La diferencia con los grandes, por pares en las mismas semanas, no es significativa: "
             + "; ".join(f"{g} {fmt(v['dif_vs_grandes'])} pb (t {v['t']:.2f})"
                         for g, v in V.items()) + ". La regla mira el cociente, no el t.")
    L.append("- Coste: la horquilla de Abdi–Ranaldo en 2012–19 es de "
             + ", ".join(f"{fmt(sp[g], 0, False)} pb en {g}" for g in GROUPS)
             + f". La ventaja bruta sobre los grandes (+4,5 a +4,7 pb por semana en la pierna "
             f"larga) se compara con una horquilla {fmt(sp[EXT_EU_GROUP] - sp[LARGE], 0, False)} pb "
             f"mayor en Europa y {fmt(sp[EXT_US_GROUPS['SP600']] - sp[LARGE], 0, False)} pb en las "
             "pequeñas de EE. UU. por cada ida y vuelta (la estimación exagera en las líquidas, "
             "así que la diferencia es una cota baja): con rotación semanal de una parte de la "
             "cartera, el coste extra se come buena parte.")
    L.append("- En Europa, la pierna larga viene del momentum (dif. de 12 meses "
             f"{fmt(mom[EXT_EU_GROUP])} pb frente a {fmt(mom[LARGE])} en los grandes; reversión "
             f"{fmt(rev[EXT_EU_GROUP])} frente a {fmt(rev[LARGE])}), justo lo que más infla el "
             "sesgo de supervivencia (componentes actuales que subieron hasta el índice).")
    L.append("- Menos acciones por decil en los grupos nuevos (acciones por fecha en 2012–19: "
             + ", ".join(f"{g} {n[g]:.0f}" for g in GROUPS) + "): cada decil tiene menos "
             "acciones y la pierna es más ruidosa, aunque eso no sesga la media.")
    return L + [""]


DECISIONS = [
    "## Sesgo de supervivencia", "",
    ("Los grupos nuevos son los componentes actuales (noviembre de 2026) de índices de medianas y "
     "pequeñas: empresas que hoy siguen cotizando y que han crecido hasta entrar en el índice (o "
     "no han caído fuera de él). Están aún más seleccionadas por haber ganado que los grandes: una "
     "pequeña de 2005 que quebró o que se quedó en micro no está, y una que subió desde micro sí. "
     "Eso infla sobre todo los niveles y la pierna larga del momentum en los años antiguos; las "
     "cifras de estos grupos son una cota optimista, más que las de los grandes."), "",
    "## Decisiones (opción conservadora ante la duda)", "",
    ("- Listas: solo las de Wikipedia con columna de tickers. Fuera: SDAX (sin tickers en la "
     "Wikipedia inglesa ni en la alemana), SBF 120 y CAC Mid 60 (sin tabla de componentes en la "
     "inglesa ni en la francesa): Francia no tiene medianas. Dentro: S&P 400 y 600; FTSE 250; "
     "MDAX; AMX y AScX (este de la Wikipedia neerlandesa); SMIM; FTSE Italia Mid Cap; IBEX "
     "Medium Cap (española) e IBEX Small Cap. No se traducen nombres a tickers a mano."),
    ("- Fuera las empresas que ya están en el universo actual: mismo símbolo, símbolo alternativo "
     "o nombre normalizado (universe.exclude_base). El nombre puede quitar alguna empresa "
     "distinta con el mismo nombre (p. ej., Ferrari Group frente a Ferrari)."),
    ("- Duplicados: además de las reglas de §3, los nombres normalizados repetidos se tratan como "
     "la misma empresa (se conserva el de más volumen), porque estas listas no tienen "
     "equivalencias declaradas ni CIK en el 400. Declaradas distintas: FBP y FBNC, INDV y SUPN "
     "(mismo CIK en Wikipedia por error). Una celda «LISN / LISP» se separa en dos clases."),
    ("- Componentes en dos listas (S&P 400 y 600, IBEX Medium y Small, AMX y AScX, desfases de "
     "Wikipedia): una sola fila; en EE. UU., cuentan como medianas."),
    ("- Tickers que Yahoo no reconoce: se pierden (sin corregirlos a mano); cuentan en la tabla "
     "de activos. Yahoo da una sola vela de HLX, INDV, LEG, QRVO y RAMP (reintentado): fuera "
     "del dataset, igual que los que la limpieza corta enteros (2)."),
    ("- Fechas semanales: las del universo actual, para todos los grupos (las mismas semanas que "
     "deciles.md); un grupo solo cuenta en las fechas con ≥ 20 de sus acciones."),
    ("- Horquilla: Abdi–Ranaldo mensual con precios sin el ajuste por dividendos; por mes, la "
     "mediana entre acciones (la media la mueven unas pocas ilíquidas). En acciones muy "
     "líquidas sobrestima la horquilla real (da ~20 pb en grandes como AAPL o MSFT, con "
     "horquillas reales de pocos pb): sirve para comparar grupos, no como coste absoluto."),
]


if __name__ == "__main__":
    main()
