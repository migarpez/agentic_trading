import numpy as np
import pandas as pd
import pytest
import torch

from agentic_trading.model.folds import CV_BLOCKS, GAP, cv_folds, manual_fold, parse_years
from agentic_trading.model.tcn import TCN, joint_loss
from agentic_trading.model.train import ensemble, month_bootstrap


def test_tcn_shape_size_and_receptive_field():
    m = TCN()
    n = sum(p.numel() for p in m.parameters())
    assert 15_000 < n < 30_000 and m.receptive_field >= 252
    x, s = torch.randn(4, 252, 5), torch.randn(4, 1)
    mu, logv = m(x, s)
    assert mu.shape == logv.shape == (4, 3)
    # la salida depende de la primera vela de la ventana (el campo receptivo la cubre)
    x2 = x.clone()
    x2[:, 0] += 10
    assert not torch.allclose(m(x2, s)[0], mu)


def test_joint_loss_matches_formula():
    torch.manual_seed(0)
    mu, logv = torch.randn(64, 3), torch.randn(64, 3) * 0.3
    y, v = torch.randn(64, 3), torch.rand(64, 3) + 0.2
    mask = torch.rand(64, 3) > 0.2
    v[~mask] = 0.0                                   # σ_YZ = 0 enmascarada: no debe dar NaN
    loss, l_mu, l_sig = joint_loss(mu, logv, y, v, mask, lam=0.5)
    ref_mu = sum(((mu[:, j] - y[:, j]) ** 2).mean() / h for j, h in enumerate((5, 10, 20)))
    qs = []
    for j in range(3):
        r = v[mask[:, j], j] ** 2 / torch.exp(2 * logv[mask[:, j], j])
        qs.append((r - torch.log(r) - 1).mean())
    assert torch.isfinite(loss)
    assert l_mu == pytest.approx(ref_mu.item(), rel=1e-5)
    assert l_sig == pytest.approx(torch.stack(qs).mean().item(), rel=1e-5)
    assert loss == pytest.approx(ref_mu.item() + 0.5 * torch.stack(qs).mean().item(), rel=1e-5)


def test_cv_folds_gap_and_coverage():
    dates = pd.bdate_range("2000-12-01", "2019-12-31")
    s = pd.DataFrame({"date": dates, "target_end": dates + pd.offsets.BDay(20)})
    folds = cv_folds(s)
    dev = s[s.target_end <= "2019-12-31"]
    pred = np.concatenate([f["predict"] for f in folds])
    assert len(pred) == len(np.unique(pred)) == len(dev)      # cada muestra se predice una vez
    for f, (a, b) in zip(folds, CV_BLOCKS):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        tr = s.iloc[f["train"]]
        before, after = tr[tr.date < a], tr[tr.date > b]
        assert (before.target_end < a - GAP).all() and (after.date > b + GAP).all()
        assert len(before) + len(after) == len(tr)


def test_manual_fold_years_gap_and_subsample():
    dates = pd.bdate_range("2000-12-01", "2019-12-31")
    s = pd.DataFrame({"date": dates, "target_end": dates + pd.offsets.BDay(20)})
    std = cv_folds(s)[3]
    same = manual_fold(s, 3, [(2000, 2011), (2016, 2019)])        # los otros cuatro: como cv3
    assert same["name"] == "cv3" and np.array_equal(same["train"], std["train"])
    assert np.array_equal(same["predict"], std["predict"])
    a, b = pd.Timestamp("2012-01-01"), pd.Timestamp("2015-12-31")
    past = s.iloc[manual_fold(s, 3, [(2000, 2011)])["train"]]       # F: solo pasado
    assert (past.date.dt.year <= 2011).all() and (past.target_end < a - GAP).all()
    assert len(past) == (std["train"] < np.searchsorted(s.date, a)).sum()
    fut = manual_fold(s, 3, [(2000, 2007), (2016, 2019)])          # P: futuro en vez de pasado
    tr = s.iloc[fut["train"]]
    assert not tr.date.dt.year.between(2008, 2015).any() and (tr.date.dt.year >= 2016).any()
    assert (tr[tr.date > b].date > b + GAP).all()
    sub = manual_fold(s, 3, [(2000, 2007), (2016, 2019)], n=1000, seed=0)
    assert len(sub["train"]) == 1000 and np.isin(sub["train"], fut["train"]).all()
    assert (np.diff(sub["train"]) > 0).all() and sub["train_n"] == 1000
    again = manual_fold(s, 3, [(2000, 2007), (2016, 2019)], n=1000, seed=0)["train"]
    other = manual_fold(s, 3, [(2000, 2007), (2016, 2019)], n=1000, seed=1)["train"]
    assert np.array_equal(sub["train"], again) and not np.array_equal(sub["train"], other)
    with pytest.raises(ValueError):
        manual_fold(s, 3, [(2010, 2013)])                         # toca el bloque predicho
    with pytest.raises(ValueError):
        manual_fold(s, 3, [(2000, 2003)], n=10 ** 6)               # más de las que hay
    assert parse_years("2000-2011") == (2000, 2011) and parse_years("2005") == (2005, 2005)


def test_month_bootstrap():
    s = pd.DataFrame({"month": np.repeat(np.arange(10), 5)})
    out = month_bootstrap(s, np.arange(50), np.random.default_rng(0))
    assert len(out) == 50 and set(np.bincount(s.month[out])) <= {0, 5, 10, 15, 20, 25}


def test_ensemble_outputs():
    p = [pd.DataFrame({"sample": [0, 1], **{f"mu_{h}": [m, -m] for h in (5, 10, 20)},
                       **{f"logv_{h}": [0.0, np.log(2)] for h in (5, 10, 20)}})
         for m in (1.0, 3.0)]
    e = ensemble(p)
    assert e.mu_5.tolist() == [2.0, -2.0]
    np.testing.assert_allclose(e.sigma_5, [np.sqrt(5), np.sqrt(5) * 2])
    np.testing.assert_allclose(e.s_5, [np.sqrt(2), np.sqrt(2)])


def test_fast_forward_equals_full_convolution():
    torch.manual_seed(0)
    m = TCN()
    x = (torch.randn(8, 252, 5) * m.scale).transpose(1, 2)
    torch.testing.assert_close(m.features_last(x), m.features_full(x), rtol=1e-5, atol=1e-5)


def test_select_rule():
    from agentic_trading.model.tuning import extend_lambda, select
    folds = [f"cv{k}" for k in range(5)]
    def res(rows):
        return pd.DataFrame([{"config": c, "fold": f, "ic": ic, "qlike": q}
                             for c, ics, q in rows for f, ic in zip(folds, ics)])
    # B tiene mejor IC en 4/5 bloques y QLIKE solo un 1 % peor -> B
    r = res([("A", [0.02] * 5, 0.300), ("B", [0.03, 0.03, 0.03, 0.03, 0.01], 0.303)])
    assert select(r)["config"] == "B"
    # B gana solo en 3/5 -> A (mejor QLIKE)
    r = res([("A", [0.02] * 5, 0.300), ("B", [0.05, 0.05, 0.05, 0.01, 0.01], 0.301)])
    assert select(r)["config"] == "A"
    # B gana en todos pero su QLIKE es un 5 % peor -> A
    r = res([("A", [0.02] * 5, 0.300), ("B", [0.03] * 5, 0.315)])
    assert select(r)["config"] == "A"
    assert extend_lambda([0.3, 1, 3], 3) == 9 and extend_lambda([0.3, 1, 3], 0.3) == 0.1
    assert extend_lambda([0.3, 1, 3], 1) is None


def test_same_seed_same_bootstrap_and_epochs():
    from agentic_trading.model.train import month_bootstrap
    s = pd.DataFrame({"month": np.repeat(np.arange(30), 7)})
    a = month_bootstrap(s, np.arange(210), np.random.default_rng(5))
    b = month_bootstrap(s, np.arange(210), np.random.default_rng(5))
    assert (a == b).all()


def test_dilations_and_short_window():
    from agentic_trading.model.tcn import dilations_for
    from agentic_trading.model.train import TrainConfig, build_model
    assert dilations_for(252) == (1, 2, 4, 8, 16, 32, 64)
    assert dilations_for(63) == (1, 2, 4, 8, 16)
    m = build_model(TrainConfig(window=63))
    assert m.receptive_field >= 63 and len(m.layers) + 1 == 5 and m.window == 63
    torch.manual_seed(0)
    x, s = torch.randn(4, 252, 5), torch.randn(4, 1)
    # con campo receptivo 63, ver solo las 63 últimas velas da lo mismo que ver las 252
    torch.testing.assert_close(m(x[:, -63:], s)[0], m(x, s)[0])
    x2 = x.clone()
    x2[:, -63] += 10                                   # y la primera de esas 63 sí cuenta
    assert not torch.allclose(m(x2[:, -63:], s)[0], m(x[:, -63:], s)[0])
    # con la ventana por defecto, la misma red y la misma inicialización que TCN()
    torch.manual_seed(1)
    a = TCN()
    torch.manual_seed(1)
    b = build_model(TrainConfig())
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_huber_equals_mse_below_threshold_and_is_linear_above():
    torch.manual_seed(0)
    mu, logv = torch.randn(64, 3), torch.randn(64, 3) * 0.3
    y, v = torch.randn(64, 3), torch.rand(64, 3) + 0.2
    mask = torch.ones(64, 3, dtype=torch.bool)
    assert (mu - y).abs().max() < 3 * np.sqrt(5)
    _, mse, _ = joint_loss(mu, logv, y, v, mask)
    _, hub, _ = joint_loss(mu, logv, y, v, mask, huber=3.0)
    assert hub == pytest.approx(mse.item(), rel=1e-6)
    # error de 20 a h = 5, por encima del umbral d = 3·√5: 2·d·|e| − d²
    z, one = torch.zeros(1, 3), torch.ones(1, 3)
    _, big, _ = joint_loss(z, z, torch.tensor([[20.0, 0.0, 0.0]]), one, one.bool(), huber=3.0)
    d = 3 * np.sqrt(5)
    assert big == pytest.approx((2 * d * 20 - d ** 2) / 5, rel=1e-5)


def test_mask_mu_excludes_samples():
    torch.manual_seed(0)
    mu, logv = torch.randn(64, 3), torch.randn(64, 3) * 0.3
    y, v = torch.randn(64, 3), torch.rand(64, 3) + 0.2
    mask = torch.ones(64, 3, dtype=torch.bool)
    keep = torch.arange(64) < 40
    _, l_mu, l_sig = joint_loss(mu, logv, y, v, mask, mask_mu=keep)
    _, ref, _ = joint_loss(mu[:40], logv[:40], y[:40], v[:40], mask[:40])
    _, _, ref_sig = joint_loss(mu, logv, y, v, mask)
    assert l_mu == pytest.approx(ref.item(), rel=1e-6) and l_sig == pytest.approx(ref_sig.item())


def test_relative_target_training_drift_and_checkpoint(tmp_path):
    from agentic_trading.data.dataset import WindowData, build_dataset
    from agentic_trading.model.train import (
        TrainConfig,
        load_model,
        predict,
        save_model,
        train_model,
    )
    from tests.test_dataset import clean_frame
    frames = {"A": clean_frame(800, 1), "B": clean_frame(700, 2)}
    build_dataset(pd.DataFrame({"symbol": list(frames)}), frames.__getitem__, tmp_path / "ds")
    data = WindowData(tmp_path / "ds")
    idx = np.arange(len(data.samples))
    cfg = TrainConfig(epochs=1, epoch_frac=0.5, batch=128, target="rel", huber=3.0, window=63,
                      threads=1)
    model, hist = train_model(data, idx, cfg, seed=0, log=lambda *_: None)
    # todas las fechas tienen < 20 muestras: no cuentan en L_μ, sí en L_σ
    assert hist[0]["l_mu"] == 0.0 and hist[0]["l_sigma"] > 0
    np.testing.assert_allclose(model.drift, data.batch(idx)["y"].astype(float).mean(0), rtol=1e-5)
    p = predict(model, data, idx)
    save_model(model, tmp_path / "m.pt", cfg, 0, hist)
    m2, ck = load_model(tmp_path / "m.pt")
    assert ck["config"]["target"] == "rel" and m2.window == 63
    pd.testing.assert_frame_equal(predict(m2, data, idx), p)
    m2.drift = None                                    # μ_h = μ_rel + deriva_h
    raw = predict(m2, data, idx)
    for j, h in enumerate((5, 10, 20)):
        np.testing.assert_allclose(p[f"mu_{h}"] - raw[f"mu_{h}"], model.drift[j], rtol=1e-5)


def test_relative_target_ignores_samples_outside_training(tmp_path, monkeypatch):
    import agentic_trading.model.train as tr
    from agentic_trading.data.dataset import WindowData, build_dataset
    from tests.test_dataset import clean_frame
    monkeypatch.setattr(tr, "REL_MIN_SAMPLES", 1)        # que L_μ cuente con pocos activos
    frames = {"A": clean_frame(800, 1), "B": clean_frame(700, 2), "C": clean_frame(750, 3)}
    build_dataset(pd.DataFrame({"symbol": list(frames)}), frames.__getitem__, tmp_path)
    data = WindowData(tmp_path)
    asset = data.samples["asset"].to_numpy()
    train_idx = np.flatnonzero(asset != 2)
    cfg = tr.TrainConfig(epochs=1, epoch_frac=0.3, batch=128, target="rel", window=63, threads=1)
    m1, h1 = tr.train_model(data, train_idx, cfg, seed=0, log=lambda *_: None)
    rows_out = data.samples["row"].to_numpy()[asset == 2]
    for h in (5, 10, 20):                                # el activo que no entrena cambia ...
        y = np.array(data.arr[f"y_{h}"])
        y[rows_out] += 50.0
        data.arr[f"y_{h}"] = y
    m2, h2 = tr.train_model(data, train_idx, cfg, seed=0, log=lambda *_: None)
    assert h1[0]["l_mu"] > 0 and h1[0]["l_mu"] == h2[0]["l_mu"]   # ... y el modelo no
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)


def test_check_dataset_refuses_other_dataset(tmp_path):
    from agentic_trading.model.tuning import check_dataset
    ident = {"clean_fingerprint": "abc", "params": {"window": 252, "horizons": (5, 10, 20)},
             "n_samples": 10}
    check_dataset(tmp_path / "t", ident)                  # primera vez: guarda la huella
    check_dataset(tmp_path / "t", ident)                  # mismo dataset: reanuda
    with pytest.raises(SystemExit):
        check_dataset(tmp_path / "t", {**ident, "n_samples": 11})
    (tmp_path / "old").mkdir()
    pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "old" / "results.parquet")
    with pytest.raises(SystemExit):                       # resultados sin huella: no reanuda
        check_dataset(tmp_path / "old", ident)
