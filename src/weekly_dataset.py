"""Build the weekly SKU-market (store x family) modelling panel.

Everything is stored as dense NumPy arrays indexed ``[series, week]`` so that lag / rolling
features can be built vectorised and - crucially - so that leakage is easy to audit: a feature
at forecast origin ``t`` may only touch columns ``<= t`` of any *historical* array.

Weeks start on Monday.  The first (partial) calendar week and the final partial week are
dropped from the history.  The panel additionally carries ``H`` future weeks of *known / assumed*
exogenous covariates (promotion plan, holidays) which are the only things allowed to look
beyond the origin.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Panel:
    weeks: pd.DatetimeIndex          # week-start dates, length Wh + H
    Wh: int                          # number of observed weeks
    H: int                           # forecast horizon (weeks)
    S: int
    n_store: int
    n_fam: int
    meta: pd.DataFrame               # one row per series
    stores: pd.DataFrame             # one row per store
    families: list
    Y_raw: np.ndarray                # [S, Wh] units incl. zeros (used for evaluation)
    Y: np.ndarray                    # [S, Wh] NaN before launch / during store closure
    L: np.ndarray                    # log1p(Y)
    Z: np.ndarray                    # [S, Wh] share of days with zero sales (NaN-masked)
    PROMO: np.ndarray                # [S, Wh+H] weekly sum of onpromotion; NaN before tracking
    TR: np.ndarray                   # [n_store, Wh] weekly transactions (NaN = no data)
    OIL: np.ndarray                  # [Wh+H] weekly mean oil price (history only is used as feature)
    CAL: dict = field(default_factory=dict)
    first_pos: np.ndarray = None     # [S] first week with sales > 0 (-1 = never)
    closed: np.ndarray = None        # [n_store, Wh] store closed whole week
    promo_start_week: int = 0
    promo_known_share: np.ndarray = None   # [H] share of days of each future week with a known plan
    last_date: pd.Timestamp = None
    first_week_start: pd.Timestamp = None
    gap_spells: pd.DataFrame = None  # mid-history zero spells (>=8w) after which sales resumed

    @property
    def store_idx(self) -> np.ndarray:
        return self.meta["store_idx"].values

    @property
    def closed_series(self) -> np.ndarray:
        return self.closed[self.store_idx]

    def gap_mask(self, upto: int) -> np.ndarray:
        """[S, Wh] True for weeks inside a *reporting-gap* spell that was already over (sales resumed)
        at week ``upto``.  Using ``upto`` = fold origin keeps this knowledge leakage-safe."""
        M = np.zeros((self.S, self.Wh), bool)
        sp = self.gap_spells
        for r in sp[sp["resume"] <= upto].itertuples(index=False):
            M[r.series, r.start:r.end + 1] = True
        return M

    def week_label(self, idx: int) -> pd.Timestamp:
        return self.weeks[idx]


def _bincount2(sidx: np.ndarray, widx: np.ndarray, vals: np.ndarray, S: int, W: int) -> np.ndarray:
    return np.bincount(sidx * W + widx, weights=vals, minlength=S * W).reshape(S, W)


def _holiday_arrays(holidays: pd.DataFrame, stores: pd.DataFrame, dates: pd.DatetimeIndex):
    """Return per-store-day holiday counts (national / regional / local), plus bridge / work-day /
    event indicators.  Transferred holidays are *not* holidays on their original date; the 'Transfer'
    row marks the day they were actually celebrated."""
    D = len(dates)
    pos = {d: i for i, d in enumerate(dates)}
    n = len(stores)
    nat = np.zeros((n, D), np.float32)
    reg = np.zeros((n, D), np.float32)
    loc = np.zeros((n, D), np.float32)
    bridge = np.zeros((n, D), np.float32)
    work = np.zeros((n, D), np.float32)
    event = np.zeros(D, np.float32)
    mothers = np.zeros(D, np.float32)
    blackfri = np.zeros(D, np.float32)
    cybermon = np.zeros(D, np.float32)
    quake = np.zeros(D, np.float32)
    city = stores["city"].values
    state = stores["state"].values
    for r in holidays.itertuples(index=False):
        i = pos.get(r.date)
        if i is None:
            continue
        typ, locale, name = r.type, r.locale, r.locale_name
        desc = str(r.description)
        if typ == "Event":
            event[i] += 1
            if "Madre" in desc:
                mothers[i] = 1
            if "Black Friday" in desc:
                blackfri[i] = 1
            if "Cyber Monday" in desc:
                cybermon[i] = 1
            if "Terremoto" in desc:
                quake[i] = 1
            continue
        if typ == "Work Day":
            work[:, i] += 1
            continue
        if bool(r.transferred):
            continue  # original date of a transferred holiday: a normal day
        if locale == "National":
            mask = np.ones(n, bool)
            target = nat
        elif locale == "Regional":
            mask = state == name
            target = reg
        else:
            mask = city == name
            target = loc
        target[mask, i] += 1
        if typ == "Bridge":
            bridge[mask, i] += 1
    return nat, reg, loc, bridge, work, event, mothers, blackfri, cybermon, quake


def find_gap_spells(Y_raw: np.ndarray, ok: np.ndarray, min_len: int = 8) -> pd.DataFrame:
    """Runs of >= ``min_len`` consecutive zero weeks inside a series' life (launched, store open)
    that are followed by positive sales again.  Whole product families vanish this way for months
    (e.g. PRODUCE in 2014/15), which looks like a reporting gap rather than zero demand."""
    rows = []
    S, W = Y_raw.shape
    for s in range(S):
        z = np.concatenate([[0], ((Y_raw[s] == 0) & ok[s]).astype(np.int8), [0]])
        d = np.diff(z)
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
        for a, b in zip(starts, ends):
            if b - a + 1 >= min_len and b + 1 < W and Y_raw[s, b + 1] > 0:
                rows.append((s, a, b, b + 1))
    return pd.DataFrame(rows, columns=["series", "start", "end", "resume"])


def build_panel(raw: dict, horizon: int = 13) -> Panel:
    train, test = raw["train"], raw["test"]
    stores = raw["stores"].sort_values("store_nbr").reset_index(drop=True)
    families = sorted(train["family"].astype(str).unique())
    n_store, n_fam = len(stores), len(families)
    S = n_store * n_fam
    st_map = {s: i for i, s in enumerate(stores["store_nbr"])}
    fam_map = {f: i for i, f in enumerate(families)}

    last_date = train["date"].max()
    first_date = train["date"].min()
    first_week_start = first_date + pd.Timedelta(days=(7 - first_date.weekday()) % 7)   # first Monday
    last_full_end = last_date - pd.Timedelta(days=(last_date.weekday() + 1) % 7)        # last Sunday <= last_date
    Wh = ((last_full_end - first_week_start).days + 1) // 7
    Wtot = Wh + horizon
    weeks = pd.date_range(first_week_start, periods=Wtot, freq="7D")

    tr = train[(train["date"] >= first_week_start) & (train["date"] <= last_full_end)]
    sidx = (tr["store_nbr"].map(st_map).values.astype(np.int64) * n_fam
            + tr["family"].astype(str).map(fam_map).values.astype(np.int64))
    widx = ((tr["date"] - first_week_start).dt.days.values // 7).astype(np.int64)
    sales = tr["sales"].values.astype(np.float64)
    Y_raw = _bincount2(sidx, widx, sales, S, Wh).astype(np.float32)
    zero_days = _bincount2(sidx, widx, (sales == 0).astype(np.float64), S, Wh)
    n_days = _bincount2(sidx, widx, np.ones(len(sales)), S, Wh)
    promo_h = _bincount2(sidx, widx, tr["onpromotion"].values.astype(np.float64), S, Wh).astype(np.float32)

    meta = pd.DataFrame({
        "series_id": np.arange(S),
        "store_nbr": np.repeat(stores["store_nbr"].values, n_fam),
        "family": np.tile(families, n_store),
        "store_idx": np.repeat(np.arange(n_store), n_fam),
        "fam_idx": np.tile(np.arange(n_fam), n_store),
    })
    for col in ("type", "cluster", "city", "state"):
        vals = np.repeat(stores[col].values, n_fam)
        meta[col] = vals
        meta[col + "_code"] = pd.factorize(pd.Series(vals).astype(str), sort=True)[0]

    # ---- launch (first sale) and store-closure masks ---------------------------------
    pos = Y_raw > 0
    first_pos = np.where(pos.any(1), pos.argmax(1), -1)
    store_tot = Y_raw.reshape(n_store, n_fam, Wh).sum(1)
    store_first = np.where((store_tot > 0).any(1), (store_tot > 0).argmax(1), 10**6)
    wk = np.arange(Wh)[None, :]
    closed = (store_tot == 0) & (wk >= store_first[:, None])
    launched = (first_pos[:, None] >= 0) & (wk >= first_pos[:, None])
    mask_ok = launched & ~closed[meta["store_idx"].values]
    Y = np.where(mask_ok, Y_raw, np.nan).astype(np.float32)
    L = np.log1p(Y)
    with np.errstate(invalid="ignore", divide="ignore"):
        Zs = zero_days / np.where(n_days > 0, n_days, np.nan)
    Z = np.where(mask_ok, Zs, np.nan).astype(np.float32)

    # ---- promotion: tracking only starts in 2014 ---------------------------------------
    promo_week_any = promo_h.sum(0)
    promo_start = int(np.argmax(promo_week_any > 0))
    PROMO = np.full((S, Wtot), np.nan, np.float32)
    PROMO[:, promo_start:Wh] = promo_h[:, promo_start:]

    # future promo: known plan (partial-week actuals + test file), then carry-forward assumption
    fut_start = last_full_end + pd.Timedelta(days=1)
    n_fut_days = horizon * 7
    fut_days = pd.date_range(fut_start, periods=n_fut_days, freq="D")
    daily = np.full((S, n_fut_days), np.nan, np.float32)
    known_df = pd.concat([
        train.loc[train["date"] >= fut_start, ["date", "store_nbr", "family", "onpromotion"]],
        test[["date", "store_nbr", "family", "onpromotion"]],
    ])
    if len(known_df):
        ks = (known_df["store_nbr"].map(st_map).values.astype(np.int64) * n_fam
              + known_df["family"].astype(str).map(fam_map).values.astype(np.int64))
        kd = (known_df["date"] - fut_start).dt.days.values
        ok = (kd >= 0) & (kd < n_fut_days)
        daily[ks[ok], kd[ok]] = known_df["onpromotion"].values[ok]
    known_mask_days = ~np.isnan(daily).all(axis=0)     # a day is 'known' if the plan exists for it
    # carry-forward: mean of the last 28 *known* days per series
    last_known_date = known_df["date"].max() if len(known_df) else last_date
    win_start = last_known_date - pd.Timedelta(days=27)
    recent = pd.concat([train.loc[train["date"] >= win_start, ["date", "store_nbr", "family", "onpromotion"]],
                        test[["date", "store_nbr", "family", "onpromotion"]]])
    recent = recent[(recent["date"] >= win_start) & (recent["date"] <= last_known_date)]
    rs = (recent["store_nbr"].map(st_map).values.astype(np.int64) * n_fam
          + recent["family"].astype(str).map(fam_map).values.astype(np.int64))
    carry = np.bincount(rs, weights=recent["onpromotion"].values.astype(np.float64), minlength=S)
    carry = carry / max(recent["date"].nunique(), 1)
    daily_filled = np.where(np.isnan(daily), carry[:, None], daily)
    PROMO[:, Wh:] = daily_filled.reshape(S, horizon, 7).sum(2)
    promo_known_share = known_mask_days.reshape(horizon, 7).mean(1)

    # ---- transactions (store traffic) -------------------------------------------------
    trn = raw["transactions"]
    trn = trn[(trn["date"] >= first_week_start) & (trn["date"] <= last_full_end)]
    ti = trn["store_nbr"].map(st_map).values.astype(np.int64)
    tw = ((trn["date"] - first_week_start).dt.days.values // 7).astype(np.int64)
    TR = _bincount2(ti, tw, trn["transactions"].values.astype(np.float64), n_store, Wh)
    TRn = _bincount2(ti, tw, np.ones(len(trn)), n_store, Wh)
    TR = np.where(TRn > 0, TR, np.nan).astype(np.float32)

    # ---- oil (external macro driver) -----------------------------------------------------
    oil = raw["oil"].set_index("date")["dcoilwtico"]
    all_days = pd.date_range(first_week_start, periods=Wtot * 7, freq="D")
    oil_d = oil.reindex(all_days).ffill().bfill()
    # the first weeks can be NaN before first observation -> bfill handles it
    OIL = oil_d.values.reshape(Wtot, 7).mean(1).astype(np.float32)
    OIL = pd.Series(OIL).ffill().values.astype(np.float32)

    # ---- calendar ----------------------------------------------------------------------------
    (nat, reg, loc, bridge, work, event, mothers, blackfri, cybermon, quake) = _holiday_arrays(
        raw["holidays"], stores, all_days)
    wk3 = lambda a: a.reshape(a.shape[0], Wtot, 7).sum(2)
    wk1 = lambda a: a.reshape(Wtot, 7).sum(1)
    mid = all_days.values.reshape(Wtot, 7)[:, 3]
    mid = pd.DatetimeIndex(mid)
    dts = pd.Series(all_days)
    is_pay = ((dts.dt.day == 15) | dts.dt.is_month_end).values.astype(np.float32)
    is_x = ((dts.dt.month == 12) & dts.dt.day.isin([24, 25, 26])).values.astype(np.float32)
    is_ny = (((dts.dt.month == 12) & (dts.dt.day == 31)) | ((dts.dt.month == 1) & dts.dt.day.isin([1, 2]))).values.astype(np.float32)
    dec25 = ((dts.dt.month == 12) & (dts.dt.day == 25)).values.astype(np.float32)
    CAL = dict(
        nat=wk3(nat), reg=wk3(reg), loc=wk3(loc), bridge=wk3(bridge), workday=wk3(work),
        event=wk1(event), mothers=wk1(mothers), blackfri=wk1(blackfri), cybermon=wk1(cybermon),
        quake=wk1(quake), payday=wk1(is_pay), xmas=wk1(is_x), ny=wk1(is_ny),
        ndays=7 - wk1(dec25),                      # Dec-25 never appears in the data
        woy=mid.isocalendar().week.values.astype(np.float32),
        month=mid.month.values.astype(np.float32),
    )
    return Panel(
        weeks=weeks, Wh=Wh, H=horizon, S=S, n_store=n_store, n_fam=n_fam, meta=meta, stores=stores,
        families=families, Y_raw=Y_raw, Y=Y, L=L, Z=Z, PROMO=PROMO, TR=TR, OIL=OIL, CAL=CAL,
        first_pos=first_pos, closed=closed, promo_start_week=promo_start,
        promo_known_share=promo_known_share, last_date=last_date, first_week_start=first_week_start,
        gap_spells=find_gap_spells(Y_raw, mask_ok),
    )
