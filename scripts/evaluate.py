"""Evalúa las predicciones fuera de muestra de una ejecución de validación cruzada.

Junta los bloques de data/predictions/<run>/cv*.parquet (2000–2019 fuera de muestra), ajusta la
línea base HAR en el entrenamiento de cada bloque y calcula las métricas del handoff §9, en total,
por bloque y por región.

--members (p. ej. 0-9) recalcula el ensemble con esos miembros de <fold>_members.parquet y
--folds (p. ej. cv4) se queda con esos bloques; con cualquiera de las dos, no escribe metrics.json.

Uso: python scripts/evaluate.py --run base [--members 0-9] [--folds cv4]
"""

import argparse
import json

import numpy as np
import pandas as pd

from agentic_trading.config import DATA_DIR
from agentic_trading.data.dataset import WindowData, asset_info
from agentic_trading.model.compare import load_members
from agentic_trading.model.folds import cv_folds
from agentic_trading.model.metrics import (
    baseline_features,
    evaluate,
    fit_har,
    predict_har,
    success_checks,
)
from agentic_trading.model.train import ensemble

H = (5, 10, 20)


def targets(data, idx):
    rows = data.samples["row"].to_numpy()[idx]
    v = np.stack([np.asarray(data.arr[f"v_{h}"])[rows] for h in H], 1).astype(float)
    m = data.samples.iloc[idx][[f"mask_sigma_{h}" for h in H]].to_numpy()
    return rows, v, m


def summary(rep: dict) -> pd.DataFrame:
    rows = []
    for h in H:
        r = rep[h]
        rows.append({"h": h, "IC": r["IC modelo"], "t": r["t IC modelo"],
                     "IC mom": r["IC momentum"], "IC rev": r["IC reversión"],
                     "IC mezcla": r["IC mezcla"], "dif. mezcla": r["dif. IC con la mezcla"],
                     "t dif.": r["t dif. IC mezcla"],
                     "dir ρ": r["dirección"]["rho"], "Q5−Q1": r["buckets"]["spread"],
                     "t Q5−Q1": r["buckets"]["t"], "QLIKE": r["QLIKE modelo"],
                     "QLIKE HAR": r.get("QLIKE HAR", np.nan), "QLIKE σ_v": r["QLIKE σ_ventana"],
                     "var real/pred": r["calibración σ"]["ratio"]})
    return pd.DataFrame(rows).set_index("h")


def parse_members(spec: str | None) -> list[int] | None:
    """'0-9' o '0,1,5' → lista de miembros; None → el ensemble guardado (todos)."""
    if spec is None:
        return None
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def print_checks(checks: dict) -> None:
    print("\nCriterio de éxito (orientativo en desarrollo):")
    mu, sig = checks["μ"], checks["σ"]
    print(f"μ: {'sí' if mu['ok'] else 'no'} ({mu['criterio']})")
    for h, r in mu["por horizonte"].items():
        print(f"  h={h}: IC μ {r['IC μ']:.4f} · IC mezcla {r['IC mezcla']:.4f} · "
              f"dif. {r['dif.']:+.4f} (t NW {r['t NW']:.2f})")
    print(f"σ: {'sí' if sig['ok'] else 'no'} ({sig['criterio']})")
    for h, r in sig["por horizonte"].items():
        print(f"  h={h}: QLIKE {r['QLIKE']:.4f} · HAR {r['QLIKE HAR']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--members", default=None, help="p. ej. 0-9; por defecto, el ensemble entero")
    ap.add_argument("--folds", nargs="+", default=None, help="p. ej. cv4; por defecto, todos")
    args = ap.parse_args()
    members = parse_members(args.members)
    pr = DATA_DIR / "predictions" / args.run
    data = WindowData(DATA_DIR / "dataset")
    info = asset_info(data.meta["symbols"], pd.read_parquet(DATA_DIR / "universe.parquet"))
    region = info["region"].to_numpy()             # por símbolo del dataset (meta.json)
    if (region == "?").any():
        print(f"AVISO: {(region == '?').sum()} activos del dataset no están en universe.parquet "
              "(región '?')")

    parts, hars, blocks = [], [], []
    for f in cv_folds(data.samples):
        if args.folds is not None and f["name"] not in args.folds:
            continue
        path = pr / f"{f['name']}.parquet"
        if not path.exists():
            continue
        if members is None:
            p = pd.read_parquet(path)
        else:
            mem = load_members(pr, f["name"], members)
            if len(mem) != len(members):
                raise SystemExit(f"{f['name']}: {len(mem)} miembros de {len(members)} pedidos")
            p = ensemble(list(mem.values()))
        tr_rows, tr_v, tr_m = targets(data, f["train"])
        rng = np.random.default_rng(0)
        sub = rng.choice(len(tr_rows), min(len(tr_rows), 500_000), replace=False)
        har = fit_har(baseline_features(data, tr_rows[sub])["har_x"], tr_v[sub], tr_m[sub])
        rows, _, _ = targets(data, p["sample"].to_numpy())
        hp = predict_har(har, baseline_features(data, rows)["har_x"])
        parts.append(p), hars.append(hp), blocks.append(f["name"])

    if not parts:
        raise SystemExit(f"No hay predicciones en {pr}")
    pred, har = pd.concat(parts, ignore_index=True), np.concatenate(hars)
    pd.set_option("display.width", 200, "display.precision", 4)

    report = {"total": evaluate(data, pred, har)}
    sel = f" (miembros {args.members})" if members is not None else ""
    print(f"== {args.run}{sel}: {len(pred):,} predicciones fuera de muestra ({', '.join(blocks)})\n")
    print(summary(report["total"]).to_string())
    print("dif. mezcla: diferencia diaria de IC con la mezcla fija de las dos líneas base "
          "(t de NW, retardos = h)")
    for h in H:
        r = report["total"][h]
        print(f"h={h}: quintiles de μ {r['buckets']['buckets']} · signo por confianza "
              f"(diagnóstico, de menor a mayor s_h) {r['signo por confianza']}")
    print("consistencia:", {k: round(v, 3) for k, v in report["total"]["consistencia"].items()})

    asset = data.samples["asset"].to_numpy()[pred["sample"].to_numpy()]
    reg = region[asset]
    for name, mask in [(b, np.repeat(blocks, [len(p) for p in parts]) == b) for b in blocks] + \
                      [(g, reg == g) for g in ("US", "Europa")]:
        report[name] = evaluate(data, pred[mask].reset_index(drop=True), har[mask])
        s = summary(report[name])
        print(f"\n{name}: IC h=5/10/20 {s.IC.round(4).tolist()} (t {s.t.round(1).tolist()}) · "
              f"QLIKE {s.QLIKE.round(3).tolist()} vs HAR {s['QLIKE HAR'].round(3).tolist()}")

    print_checks(success_checks(report["total"]))
    if members is None and args.folds is None:
        (pr / "metrics.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
