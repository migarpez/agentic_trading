"""Universo de activos: componentes actuales de los índices (Wikipedia) + cripto.

Produce una tabla con una fila por empresa/activo y su símbolo de Yahoo Finance. La tabla no
entra en el modelo (que no conoce la identidad del activo); sirve para descargar, auditar y
agrupar métricas.

Pasos:
1. Componentes de cada índice desde Wikipedia, con el ticker traducido a Yahoo.
2. Grupos de símbolos equivalentes (misma empresa): cotizaciones en varias bolsas declaradas en
   EQUIVALENT_LISTINGS y clases de acción del S&P 500 detectadas por CIK.
3. En cada grupo se conserva el símbolo con mayor volumen negociado (mediana de C·V en EUR del
   último año), calculado en cada ejecución.
4. Aviso de posibles duplicados no declarados (nombres que coinciden tras normalizar).

Universo ampliado (22-11-2026, `build_universe(INDICES_EXT, extra=[], exclude=..., by_name=True)`,
`make ext`): medianas y pequeñas de EE. UU. y de los mismos países europeos, aparte del actual (en
data/ext/). Se quitan las empresas que ya están en el universo actual (por símbolo, símbolo
alternativo o nombre normalizado) y, como no hay equivalencias declaradas para estas listas, los
nombres normalizados repetidos se tratan como la misma empresa (se conserva el de más volumen).
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass
from datetime import date

import pandas as pd
import requests
import yfinance as yf

WIKI = "https://en.wikipedia.org/wiki/"
HEADERS = {"User-Agent": "agentic-trading-research/0.1 (python-requests)"}

# Sufijo de Yahoo -> (país de la bolsa, divisa de cotización). Ojo: Londres cotiza en peniques.
EXCHANGES = {
    "": ("US", "USD"),
    ".L": ("GB", "GBp"),
    ".DE": ("DE", "EUR"),
    ".PA": ("FR", "EUR"),
    ".MC": ("ES", "EUR"),
    ".MI": ("IT", "EUR"),
    ".AS": ("NL", "EUR"),
    ".SW": ("CH", "CHF"),
}

# Divisa -> (par de Yahoo frente al EUR, factor de unidad). GBp son peniques: 1/100 de libra.
FX_TO_EUR = {
    "EUR": (None, 1.0),
    "USD": ("USDEUR=X", 1.0),
    "GBp": ("GBPEUR=X", 0.01),
    "CHF": ("CHFEUR=X", 1.0),
}


@dataclass(frozen=True)
class IndexSpec:
    name: str
    page: str
    ticker_col: str
    name_col: str
    sector_col: str
    suffix: str  # sufijo a añadir si el ticker de Wikipedia no lo trae
    expected: int
    wiki: str = WIKI  # Wikipedia de otro idioma cuando la inglesa no trae los tickers


INDICES = [
    IndexSpec("SP500", "List_of_S%26P_500_companies", "Symbol", "Security", "GICS Sector", "", 503),
    IndexSpec("FTSE100", "FTSE_100_Index", "Ticker", "Company",
              "FTSE industry classification benchmark sector", ".L", 100),
    IndexSpec("DAX", "DAX", "Ticker", "Company", "Prime Standard Sector", ".DE", 40),
    IndexSpec("CAC40", "CAC_40", "Ticker", "Company", "Sector", ".PA", 40),
    IndexSpec("IBEX35", "IBEX_35", "Ticker", "Company", "Sector", ".MC", 35),
    IndexSpec("FTSEMIB", "FTSE_MIB", "Ticker", "Company", "ICB Sector", ".MI", 40),
    IndexSpec("AEX", "AEX_index", "Ticker", "Company", "ICB Sector", ".AS", 25),
    IndexSpec("SMI", "Swiss_Market_Index", "Ticker", "Name", "Sector", ".SW", 20),
]

# Universo ampliado (22-11-2026): solo listas de Wikipedia con tickers (sin traducir nombres a
# tickers a mano). Fuera: SDAX (sin tickers en en/de), SBF 120 y CAC Mid 60 (sin tabla de
# componentes); ver logs/universe_ext.md.
INDICES_EXT = [
    IndexSpec("SP400", "List_of_S%26P_400_companies", "Symbol", "Security", "GICS Sector", "",
              400),
    IndexSpec("SP600", "List_of_S%26P_600_companies", "Symbol", "Security", "GICS Sector", "",
              603),
    IndexSpec("FTSE250", "FTSE_250_Index", "Ticker", "Company",
              "FTSE Industry Classification Benchmark sector", ".L", 250),
    IndexSpec("MDAX", "MDAX", "Symbol", "Name", "Industry", ".DE", 50),
    IndexSpec("AMX", "AMX_index", "Ticker symbol", "Company", "ICB Sector", ".AS", 25),
    IndexSpec("AScX", "AScX", "Ticker symbol", "Company", "Sector", ".AS", 20,
              "https://nl.wikipedia.org/wiki/"),
    IndexSpec("SMIM", "SMIM", "Ticker symbol", "Company", "Sector", ".SW", 30),
    IndexSpec("FTSEITMID", "FTSE_Italia_Mid_Cap", "Ticker", "Company", "ICB supersector", ".MI",
              60),
    IndexSpec("IBEXMC", "IBEX_Medium_Cap", "Ticker", "Empresa", "Sector", ".MC", 20,
              "https://es.wikipedia.org/wiki/"),
    IndexSpec("IBEXSC", "IBEX_Small_Cap", "Ticker", "Company", "Sector", ".MC", 30),
]
EXT_US_GROUPS = {"SP400": "medianas EE. UU.", "SP600": "pequeñas EE. UU."}
EXT_EU_GROUP = "medianas y pequeñas Europa"

# Correcciones de símbolos de Wikipedia que no coinciden con Yahoo (símbolo ya convertido).
SYMBOL_FIXES: dict[str, str] = {}

# Misma empresa cotizada en varias bolsas europeas (hecho sobre la empresa, no una elección:
# qué símbolo se conserva lo decide el volumen en build_universe).
EQUIVALENT_LISTINGS = [
    {"SHEL.L", "SHELL.AS"},       # Shell
    {"ULVR.L", "UNA.AS"},         # Unilever
    {"STLAM.MI", "STLAP.PA"},     # Stellantis
    {"STMPA.PA", "STMMI.MI"},     # STMicroelectronics
    {"IAG.L", "IAG.MC"},          # International Airlines Group
    {"MT.AS", "MTS.MC"},          # ArcelorMittal
    {"REL.L", "REN.AS"},          # RELX
]

# Pares con el mismo nombre normalizado que son empresas distintas (silencia el aviso).
KNOWN_DISTINCT = [
    {"MRK.DE", "MRK"},  # Merck KGaA (Alemania) frente a Merck & Co. (EE. UU.)
    # Universo ampliado (también para los grupos por CIK o por nombre de equivalent_groups):
    {"FBP", "FBNC"},    # First BanCorp (Puerto Rico) frente a First Bancorp (Carolina del Norte)
    {"INDV", "SUPN"},   # Indivior frente a Supernus: mismo CIK en Wikipedia por error
]

EXTRA_ASSETS = [
    # symbol, name, asset_class
    ("BTC-USD", "Bitcoin", "crypto"),
    ("ETH-USD", "Ethereum", "crypto"),
    # Commodities descartadas: los futuros continuos de Yahoo no tienen un volumen utilizable.
]

COLUMNS = ["symbol", "name", "asset_class", "indices", "country", "currency", "sector",
           "source_symbol", "alt_symbols", "snapshot_date"]


# --------------------------------------------------------------------------- símbolos

def to_yahoo(ticker: str, suffix: str) -> str:
    """Convierte un ticker de Wikipedia al formato de Yahoo Finance."""
    t = str(ticker).strip().upper()
    if suffix and any(t.endswith(s) for s in EXCHANGES if s):
        return SYMBOL_FIXES.get(t, t)  # ya trae sufijo (p. ej. AIR.PA en el DAX)
    t = t.rstrip(".").replace(".", "-")  # BRK.B -> BRK-B, BT.A -> BT-A
    return SYMBOL_FIXES.get(t + suffix, t + suffix)


def exchange_of(symbol: str) -> tuple[str, str]:
    for suffix, info in EXCHANGES.items():
        if suffix and symbol.endswith(suffix):
            return info
    return EXCHANGES[""]


def normalize_name(name: str) -> str:
    """Nombre de empresa sin acentos, puntuación, forma jurídica ni clase de acción."""
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"\(.*?\)", " ", n)
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    stop = {"plc", "ag", "sa", "se", "nv", "inc", "ltd", "limited", "corp", "corporation",
            "company", "co", "the", "holding", "holdings", "group", "spa", "kgaa", "class"}
    return " ".join(w for w in n.split() if w not in stop)


# --------------------------------------------------------------------------- Wikipedia

def _find_table(html: str, spec: IndexSpec) -> pd.DataFrame:
    """Busca la tabla de componentes: la que tiene la columna de ticker y el tamaño esperado."""
    for t in pd.read_html(io.StringIO(html)):
        cols = {str(c).split("[")[0].strip(): c for c in t.columns}
        if spec.ticker_col in cols and spec.name_col in cols:
            t = t.rename(columns={v: k for k, v in cols.items()})
            t = t.dropna(subset=[spec.ticker_col])
            if abs(len(t) - spec.expected) <= max(3, spec.expected // 10):
                return t
    raise ValueError(f"{spec.name}: no se encontró la tabla de componentes en {spec.page}")


def fetch_index(spec: IndexSpec, session: requests.Session | None = None) -> pd.DataFrame:
    s = session or requests
    r = s.get(spec.wiki + spec.page, headers=HEADERS, timeout=30)
    r.raise_for_status()
    t = _find_table(r.text, spec)
    out = pd.DataFrame({
        "source_symbol": t[spec.ticker_col].astype(str).str.strip(),
        "name": t[spec.name_col].astype(str).str.strip(),
        "sector": t[spec.sector_col].astype(str).str.strip() if spec.sector_col in t else None,
    })
    out["cik"] = t["CIK"].astype(str).str.strip().values if "CIK" in t else None
    # «LISN / LISP»: dos clases de la misma empresa en una celda; una fila por clase (con
    # by_name, build_universe conserva la de más volumen).
    out = (out.assign(source_symbol=out["source_symbol"].str.split(r"\s*/\s*"))
           .explode("source_symbol", ignore_index=True))
    out["symbol"] = [to_yahoo(x, spec.suffix) for x in out["source_symbol"]]
    out["index"] = spec.name
    return out


def fetch_sp500_added(session: requests.Session | None = None) -> pd.DataFrame:
    """Fecha de alta en el S&P 500 ("Date added") de cada componente actual, de la tabla de
    Wikipedia (23-11-2026; scripts/survivorship.py). Columnas: source_symbol, symbol (Yahoo),
    name, date_added (NaT si la celda no trae fecha) y fetched (fecha de descarga). Una fila por
    clase de acción, como fetch_index."""
    spec = next(x for x in INDICES if x.name == "SP500")
    s = session or requests
    r = s.get(spec.wiki + spec.page, headers=HEADERS, timeout=30)
    r.raise_for_status()
    t = _find_table(r.text, spec)
    if "Date added" not in t:
        raise ValueError("SP500: la tabla no trae la columna Date added")
    out = pd.DataFrame({
        "source_symbol": t[spec.ticker_col].astype(str).str.strip(),
        "name": t[spec.name_col].astype(str).str.strip(),
        "date_added": pd.to_datetime(t["Date added"].astype(str).str.extract(
            r"(\d{4}-\d{2}-\d{2})")[0], errors="coerce").to_numpy(),
    })
    out = (out.assign(source_symbol=out["source_symbol"].str.split(r"\s*/\s*"))
           .explode("source_symbol", ignore_index=True))
    out["symbol"] = [to_yahoo(x, spec.suffix) for x in out["source_symbol"]]
    out["fetched"] = pd.Timestamp(date.today())
    return out


# --------------------------------------------------------------------------- equivalencias

def equivalent_groups(raw: pd.DataFrame, by_name: bool = False) -> list[set[str]]:
    """Grupos de símbolos de la misma empresa presentes en raw: declarados + mismo CIK (+ mismo
    nombre normalizado con by_name). Los grupos que se solapan se unen."""
    present = set(raw["symbol"])
    groups = [g & present for g in EQUIVALENT_LISTINGS]
    cik = raw[raw["cik"].notna()].drop_duplicates("symbol")
    groups += [set(g) for g in cik.groupby("cik")["symbol"].agg(list) if len(g) > 1]
    if by_name:
        key = raw.drop_duplicates("symbol").assign(k=lambda d: d["name"].map(normalize_name))
        groups += [set(g) for g in key.groupby("k")["symbol"].agg(list) if len(g) > 1]
    merged: list[set[str]] = []
    for g in (g for g in groups if len(g) > 1 and g not in KNOWN_DISTINCT):
        for m in [m for m in merged if m & g]:
            merged.remove(m)
            g = g | m
        merged.append(g)
    return merged


def exclude_base(raw: pd.DataFrame, base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quita de raw las empresas que ya están en el universo base: mismo símbolo, uno de sus
    símbolos alternativos o el mismo nombre normalizado. Devuelve (raw sin ellas, quitadas)."""
    syms = set(base["symbol"])
    for a in base["alt_symbols"].dropna():
        syms |= set(str(a).split("|"))
    names = set(base.loc[base["asset_class"] == "equity", "name"].map(normalize_name))
    drop = raw["symbol"].isin(syms) | raw["name"].map(normalize_name).isin(names)
    return raw[~drop].reset_index(drop=True), raw[drop].reset_index(drop=True)


def traded_value_eur(symbols: list[str], period: str = "1y") -> pd.Series:
    """Mediana diaria de cierre × volumen en EUR durante `period`."""
    currencies = {s: exchange_of(s)[1] for s in symbols}
    pairs = sorted({FX_TO_EUR[c][0] for c in currencies.values() if FX_TO_EUR[c][0]})
    data = yf.download(list(symbols) + pairs, period=period, auto_adjust=False,
                       progress=False, threads=True)
    close, volume = data["Close"], data["Volume"]
    out = {}
    for s, cur in currencies.items():
        pair, unit = FX_TO_EUR[cur]
        fx = close[pair].ffill() if pair else 1.0
        out[s] = (close[s] * volume[s] * unit * fx).median()
    return pd.Series(out, name="traded_value_eur")


def resolve_equivalents(groups: list[set[str]]) -> tuple[dict[str, str], pd.DataFrame]:
    """Devuelve {descartado: conservado} y la tabla de volúmenes que justifica la elección."""
    if not groups:
        return {}, pd.DataFrame()
    tv = traded_value_eur(sorted(set().union(*groups)))
    rows, mapping = [], {}
    for gid, g in enumerate(groups):
        vals = tv.reindex(sorted(g))
        if vals.isna().all():
            raise ValueError(f"Sin datos de volumen para el grupo {sorted(g)}")
        keep = vals.idxmax()
        for s, v in vals.items():
            rows.append((gid, s, v, s == keep))
            if s != keep:
                mapping[s] = keep
    table = pd.DataFrame(rows, columns=["group", "symbol", "traded_value_eur", "kept"])
    return mapping, table


def possible_duplicates(uni: pd.DataFrame) -> pd.DataFrame:
    """Símbolos distintos con el mismo nombre normalizado que no están declarados como distintos."""
    eq = uni[uni["asset_class"] == "equity"].assign(key=lambda d: d["name"].map(normalize_name))
    dup = eq[eq.duplicated("key", keep=False)]
    keep = [k for k, g in dup.groupby("key") if set(g["symbol"]) not in KNOWN_DISTINCT]
    return dup[dup["key"].isin(keep)].sort_values("key")[["key", "symbol", "name", "indices"]]


# --------------------------------------------------------------------------- universo

def build_universe(snapshot: date | None = None, indices: list[IndexSpec] = INDICES,
                   extra: list[tuple] = EXTRA_ASSETS, exclude: pd.DataFrame | None = None,
                   by_name: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (universo, tabla de resolución de equivalencias). Con exclude (otro universo),
    las empresas que ya están en él quedan fuera; van en uni.attrs["excluded"], y los
    componentes leídos de cada índice en uni.attrs["fetched"]."""
    snapshot = snapshot or date.today()
    with requests.Session() as session:
        raw = pd.concat([fetch_index(spec, session) for spec in indices], ignore_index=True)
    fetched = raw.groupby("index").size().to_dict()
    excluded = raw.iloc[:0]
    if exclude is not None:
        raw, excluded = exclude_base(raw, exclude)

    mapping, listings = resolve_equivalents(equivalent_groups(raw, by_name))
    raw["orig_symbol"] = raw["symbol"]
    raw["symbol"] = raw["symbol"].replace(mapping)

    # Una fila por símbolo, registrando todos los índices y símbolos alternativos.
    alts = pd.Series(mapping).groupby(lambda s: mapping[s]).agg(lambda s: "|".join(sorted(s.index)))
    indices = raw.groupby("symbol")["index"].agg(lambda s: "|".join(dict.fromkeys(s)))
    # Nombre, sector y ticker de Wikipedia: los del listado conservado, no los del primero.
    own = raw["orig_symbol"] == raw["symbol"]
    eq = (raw.assign(_own=own).sort_values("_own", ascending=False, kind="stable")
          .drop_duplicates("symbol").set_index("symbol"))
    eq["indices"] = indices
    eq["alt_symbols"] = alts.reindex(eq.index)
    eq = eq.reset_index()
    eq["asset_class"] = "equity"
    eq[["country", "currency"]] = [exchange_of(s) for s in eq["symbol"]]

    extra = pd.DataFrame(extra, columns=["symbol", "name", "asset_class"])
    extra["source_symbol"] = extra["symbol"]
    extra["currency"] = "USD"

    uni = pd.concat([eq, extra], ignore_index=True)
    uni["snapshot_date"] = pd.Timestamp(snapshot)
    uni = uni[COLUMNS].sort_values(["asset_class", "country", "symbol"], ignore_index=True)
    if not uni["symbol"].is_unique:
        raise ValueError("Símbolos duplicados en el universo")
    uni.attrs = {"fetched": fetched, "excluded": excluded[["index", "symbol", "name"]]}
    return uni, listings
