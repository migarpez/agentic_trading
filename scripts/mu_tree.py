"""Orquestador del árbol de pruebas sobre μ (mu_tree, 22-11-2026). Reglas, árbol y tiempos en
logs/mu_tree/README.md (fijados antes de lanzar; no se cambian).

Protocolo de cada paso: la configuración de final20 (target relativo, ventana de 63, velas
mensuales, MSE, λ = 1, lr = 0,001, 5 épocas, 20 % por época) más las opciones de su base y la
que se prueba; 2 miembros por bloque (semillas 1000·k + m, m = 0, 1, las de los miembros 0 y 1
de final20) en los 5 bloques, en data/predictions/mu_<paso>/. Se compara con su base
(scripts/mu_compare.py) y, si gana, pasa a ser la base de los siguientes. La base inicial, R0,
son los miembros 0 y 1 de final20.

  E1 = base + market;  E2 = base + target_rank;  E3 = base + xsec (solo si E1 ganó);
  E4 = base + tail_weight.
  Si ganó alguno: F1 (ensemble de 20 con la base final, data/predictions/mu_final20/, frente a
  final20) y F2 (agente: plan B con μ real como w_real, con esas predicciones, y su evaluación).
  Si no: F3 (plan B con final20, las 3 semillas de w_real, hasta 1.000 actualizaciones).

Reanudable: no repite lo hecho (miembros ya entrenados, veredictos ya escritos, agentes ya
entrenados o evaluados). Cada paso guarda su huella (dataset, vela del mercado si la usa,
configuración, miembros, base) en data/predictions/mu_<paso>/tree_fingerprint.json; si ya existe
y no coincide, el paso falla y no se sobrescribe. Si un paso falla, se registra y se sigue con lo
que no dependa de él (un paso que falla cuenta como que no gana). Estado en
logs/mu_tree/state.json; log en logs/mu_tree/tree.log.

Uso: setsid nohup .venv/bin/python -u scripts/mu_tree.py > logs/mu_tree/tree.log 2>&1 < /dev/null &
     python scripts/mu_tree.py --dry-run     (solo enseña el plan)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd

from agentic_trading.config import DATA_DIR, ROOT

PY = sys.executable
OUT = ROOT / "logs" / "mu_tree"
PRED = DATA_DIR / "predictions"
FOLDS = range(5)
CFG = ["--lr", "0.001", "--lam", "1", "--target", "rel", "--window", "63", "--coarse", "month",
       "--epochs", "5", "--epoch-frac", "0.2"]
OPTS = {"market": ["--market"], "target_rank": ["--target-rank"], "xsec": ["--xsec"],
        "tail_weight": ["--tail-weight", "3"]}
STEPS = [("E1", "market"), ("E2", "target_rank"), ("E3", "xsec"), ("E4", "tail_weight")]
R0 = {"name": "R0", "spec": "final20:0,1", "opts": []}
MEMBERS_STEP, MEMBERS_FINAL = 2, 20
# Agente (la configuración de w_real, scripts/planb.py)
AGENT = ["--reward", "relative", "--ratio", "component", "--trade-bias", "-2",
         "--freq", "weekly", "--gamma", "0.8"]
AGENT_SEEDS = (0, 1, 2)


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def load_state() -> dict:
    p = OUT / "state.json"
    return json.loads(p.read_text()) if p.exists() else {"steps": {}}


def save_state(state: dict) -> None:
    (OUT / "state.json").write_text(json.dumps(state, indent=1, ensure_ascii=False))


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                                   # noqa: BLE001
        return None


def run(cmd: list[str], out_path, env: dict | None = None) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out_path, "a") as fh:
        fh.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] $ {' '.join(cmd[1:])}\n")
        fh.flush()
        rc = subprocess.run(cmd, check=False, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                            env={**os.environ, **(env or {})}).returncode
    log(f"{'OK' if rc == 0 else f'FALLO ({rc})'} · {(time.time() - t0) / 60:.1f} min · "
        f"{' '.join(cmd[2:6])}… > {out_path.relative_to(ROOT)}")
    return rc == 0


# ----------------------------------------------------------------------------- modelo

def fingerprint(run_name: str, opts: list[str], members: int, base: str | None) -> dict:
    from agentic_trading.data.dataset import WindowData, dataset_id
    data = WindowData(DATA_DIR / "dataset")
    fp = {"dataset": dataset_id(data.meta), "cfg": CFG, "opts": opts,
          "flags": [f for o in opts for f in OPTS[o]], "members": members,
          "seeds": "1000·k + m", "base": base}
    if "market" in opts:
        fp["market"] = json.loads((DATA_DIR / "dataset" / "market" / "meta.json").read_text())
    return json.loads(json.dumps(fp, default=str))


def check_fingerprint(run_name: str, fp: dict) -> None:
    d = PRED / run_name
    d.mkdir(parents=True, exist_ok=True)
    p = d / "tree_fingerprint.json"
    if p.exists():
        if json.loads(p.read_text()) != fp:
            raise RuntimeError(f"{p}: la huella no coincide; no se sobrescribe")
    else:
        if any(d.glob("cv*.parquet")):
            raise RuntimeError(f"{d}: predicciones sin huella; no se sobrescribe")
        p.write_text(json.dumps(fp, indent=1))


def merge_fold(run_name: str, k: int, members: int) -> None:
    """Junta <bloque>_members y <bloque>.parquet si faltan (lo hace train_cv; por si acaso)."""
    from agentic_trading.model.train import ensemble
    d = PRED / run_name
    if (d / f"cv{k}.parquet").exists():
        return
    preds = [pd.read_parquet(d / f"cv{k}_m{m}.parquet") for m in range(members)]
    pd.concat(preds).to_parquet(d / f"cv{k}_members.parquet", index=False)
    ensemble(preds).to_parquet(d / f"cv{k}.parquet", index=False)


def train_run(run_name: str, opts: list[str], members: int, workers: int, threads: int) -> bool:
    flags = [f for o in opts for f in OPTS[o]]
    jobs = [(k, m) for m in range(members) for k in FOLDS
            if not (PRED / run_name / f"cv{k}_m{m}.parquet").exists()]
    log(f"{run_name}: {len(jobs)} entrenamientos pendientes de {members * len(FOLDS)} "
        f"({workers} a la vez × {threads} hilos) · opciones {opts}")

    def job(km):
        k, m = km
        cmd = [PY, "-u", str(ROOT / "scripts" / "train_cv.py"), "--run", run_name, "--folds",
               str(k), "--members", str(members), "--member-ids", str(m), "--threads",
               str(threads), *CFG, *flags]
        out = OUT / "train" / f"{run_name}_cv{k}_m{m}.log"
        env = {"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
        return run(cmd, out, env) or run(cmd, out, env)        # un reintento

    with ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(job, jobs))
    if not all(ok):
        log(f"{run_name}: fallaron {sum(not x for x in ok)} entrenamientos")
        return False
    for k in FOLDS:
        merge_fold(run_name, k, members)
    return True


def step_model(state: dict, step: str, run_name: str, opts: list[str], members: int,
               base_spec: str, workers: int, threads: int, full: bool = False) -> dict:
    """Entrena (si falta) y compara con la base. Devuelve el veredicto (o el fallo)."""
    st = state["steps"].get(step, {})
    vpath = OUT / f"{step}.json"
    fp = fingerprint(run_name, opts, members, base_spec)
    if st.get("status") == "done" and vpath.exists() and st.get("fingerprint") == fp:
        log(f"{step}: ya hecho ({'gana' if st['wins'] else 'no gana'})")
        return st
    st = {"run": run_name, "opts": opts, "base": base_spec, "fingerprint": fp,
          "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
    try:
        check_fingerprint(run_name, fp)
        t0 = time.time()
        if not train_run(run_name, opts, members, workers, threads):
            raise RuntimeError("entrenamiento incompleto")
        st["train_minutes"] = (time.time() - t0) / 60
        if full and not run([PY, "-u", str(ROOT / "scripts" / "evaluate.py"), "--run", run_name],
                            OUT / f"{step}_evaluate.log"):
            raise RuntimeError("evaluate.py falló")
        cmd = [PY, "-u", str(ROOT / "scripts" / "mu_compare.py"), "--step", step, "--run",
               run_name, "--base", base_spec] + (["--full"] if full else [])
        if not run(cmd, OUT / f"{step}_compare.log"):
            raise RuntimeError("mu_compare.py falló")
        v = json.loads(vpath.read_text())["verdict"]
        st.update(status="done", wins=bool(v["wins"]), verdict=v)
    except Exception as e:                              # noqa: BLE001 — se registra y se sigue
        st.update(status="failed", wins=False, error=repr(e))
        log(f"{step}: FALLO · {e!r}")
    st["finished"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    state["steps"][step] = st
    save_state(state)
    return st


# ----------------------------------------------------------------------------- agente

def agent_done(name: str, updates: int) -> bool:
    p = DATA_DIR / "agent" / name / "ckpt.pt"
    if not p.exists():
        return False
    import torch
    return torch.load(p, weights_only=False)["updates"] >= updates


def train_agents(names: list[str], extra: dict[str, list[str]], updates: int) -> bool:
    def job(name):
        if agent_done(name, updates):
            log(f"{name}: ya entrenado ({updates} actualizaciones)")
            return True
        cmd = [PY, "-u", str(ROOT / "scripts" / "train_agent.py"), "--name", name, "--updates",
               str(updates), *AGENT, *extra[name]]
        return run(cmd, ROOT / "logs" / "agent" / f"{name}.log", {"OMP_NUM_THREADS": "1"})

    with ThreadPoolExecutor(len(names)) as ex:
        return all(ex.map(job, names))


def eval_agents(step: str, names: list[str], refs: list[str]) -> bool:
    fresh = all((ROOT / "logs" / "agent" / f"{n}_eval2.json").exists()
                and (ROOT / "logs" / "agent" / f"{n}_eval2.json").stat().st_mtime
                > (DATA_DIR / "agent" / n / "ckpt.pt").stat().st_mtime for n in names)
    if not fresh and not run([PY, "-u", str(ROOT / "scripts" / "agent_eval.py"), "--names",
                              *names, "--seeds", "20", "--summary",
                              f"logs/mu_tree/{step}_agent_eval.md"],
                             OUT / f"{step}_agent_eval.log"):
        return False
    return run([PY, "-u", str(ROOT / "scripts" / "mu_compare.py"), "--step", step, "--agent",
                "--names", *names, "--ref", *refs], OUT / f"{step}_compare.log")


def step_agent(state: dict, step: str, fn) -> None:
    st = state["steps"].get(step, {})
    if st.get("status") == "done" and (OUT / f"{step}.json").exists():
        log(f"{step}: ya hecho")
        return
    st = {"started": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
    try:
        t0 = time.time()
        fn(st)
        st.update(status="done", minutes=(time.time() - t0) / 60)
    except Exception as e:                              # noqa: BLE001
        st.update(status="failed", error=repr(e))
        log(f"{step}: FALLO · {e!r}")
    st["finished"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    state["steps"][step] = st
    save_state(state)


def f2(st: dict) -> None:
    names = [f"mf_real_s{s}" for s in AGENT_SEEDS]
    extra = {n: ["--run", "mu_final20", "--seed", str(s)] for n, s in zip(names, AGENT_SEEDS)}
    st["names"] = names
    if not train_agents(names, extra, 400):
        raise RuntimeError("entrenamiento del agente incompleto")
    if not eval_agents("F2", names, [f"w_real_s{s}" for s in AGENT_SEEDS]):
        raise RuntimeError("evaluación del agente fallida")


def f3(st: dict) -> None:
    """w_real_s* seguidos hasta 1.000 actualizaciones: se copian su checkpoint, su huella y su
    log (400 actualizaciones) a w_real1000_s* y se reanuda (train_agent.py comprueba la huella).
    Si la copia no puede reanudarse, se entrena desde cero con el mismo nombre."""
    names = [f"w_real1000_s{s}" for s in AGENT_SEEDS]
    st["names"] = names
    for s, n in zip(AGENT_SEEDS, names):
        src, dst = DATA_DIR / "agent" / f"w_real_s{s}", DATA_DIR / "agent" / n
        if not dst.exists():
            shutil.copytree(src, dst)
            shutil.copy(ROOT / "logs" / "agent" / f"w_real_s{s}.jsonl",
                        ROOT / "logs" / "agent" / f"{n}.jsonl")
            log(f"{n}: copiado de w_real_s{s} (reanuda en la actualización 400)")
    extra = {n: ["--seed", str(s)] for n, s in zip(names, AGENT_SEEDS)}
    if not train_agents(names, extra, 1000):
        raise RuntimeError("entrenamiento del agente incompleto")
    if not eval_agents("F3", names, [f"w_real_s{s}" for s in AGENT_SEEDS]):
        raise RuntimeError("evaluación del agente fallida")


# ----------------------------------------------------------------------------- árbol

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None, help="entrenamientos a la vez")
    ap.add_argument("--threads", type=int, default=None, help="hilos por entrenamiento")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    par = json.loads((OUT / "parallel.json").read_text())
    workers = args.workers or par["workers"]
    threads = args.threads or par["threads"]
    state = load_state()
    state.setdefault("runs", []).append({"start": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                                         "git": git_head(), "workers": workers,
                                         "threads": threads})
    if args.dry_run:
        print(json.dumps({"workers": workers, "threads": threads, "steps": STEPS}, indent=1))
        return
    save_state(state)
    log(f"Inicio · versión {state['runs'][-1]['git']} · {workers} × {threads} hilos")

    if not (DATA_DIR / "dataset" / "market" / "meta.json").exists():
        log("construyendo la vela del mercado")
        from agentic_trading.data.dataset import WindowData
        from agentic_trading.data.market import build_market
        build_market(WindowData(DATA_DIR / "dataset"), DATA_DIR / "clean",
                     pd.read_parquet(DATA_DIR / "universe.parquet"))

    base = dict(R0)
    won = {}
    for step, opt in STEPS:
        if step == "E3" and not won.get("E1"):
            log("E3: no se prueba (E1 no ganó)")
            state["steps"]["E3"] = {"status": "skipped", "wins": False,
                                    "reason": "E1 no ganó"}
            save_state(state)
            won["E3"] = False
            continue
        opts = [*base["opts"], opt]
        log(f"== {step}: base {base['name']} ({base['spec']}) + {opt}")
        st = step_model(state, step, f"mu_{step}", opts, MEMBERS_STEP, base["spec"], workers,
                        threads)
        won[step] = st.get("wins", False)
        log(f"{step}: {st.get('status')} · {'GANA' if won[step] else 'no gana'}")
        if won[step]:
            base = {"name": step, "spec": f"mu_{step}", "opts": opts}
    state["final_base"] = base
    save_state(state)
    log(f"Base final: {base['name']} · opciones {base['opts']}")

    if base["name"] != "R0":
        log("== F1: ensemble de 20 con la base final")
        st = step_model(state, "F1", "mu_final20", base["opts"], MEMBERS_FINAL, "final20",
                        workers, threads, full=True)
        if st.get("status") == "done":
            log("== F2: agente con mu_final20")
            step_agent(state, "F2", f2)
        else:
            log("F2: no se hace (F1 falló)")
            state["steps"]["F2"] = {"status": "skipped", "reason": "F1 falló"}
            save_state(state)
    else:
        log("== F3: plan B con final20 hasta 1.000 actualizaciones")
        step_agent(state, "F3", f3)
    state["finished"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    save_state(state)
    log("Fin.")


if __name__ == "__main__":
    main()
