import numpy as np
import pandas as pd


def load_invoices(src):
    """src: a CSV file path or URL with a date column and a price column."""
    df = pd.read_csv(src)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"date", "price"} <= set(df.columns):
        df = df.iloc[:, :2]
        df.columns = ["date", "price"]
    df = df[["date", "price"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"].astype(str).str.replace(",", ""), errors="coerce")
    df = df.dropna()
    return df.groupby("date", as_index=False)["price"].mean().sort_values("date").reset_index(drop=True)


def load_market(path):
    try:
        m = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    except FileNotFoundError:
        return None
    m["copper_bdt"] = m["copper_usd_lb"] * m["usdbdt"]
    return m.dropna()


def vol_band(df, h):
    p, d = df["price"].values, (df["date"] - df["date"].iloc[0]).dt.days.values.astype(float)
    gaps = np.maximum(np.diff(d), 1)
    dv = np.std(np.diff(p) / np.sqrt(gaps)) if len(p) > 3 else 0.0
    return 1.28 * dv * np.sqrt(h)  # ~80% range


def f_naive(df, mkt, h):
    return df["price"].iloc[-1]


def f_copper(df, mkt, h, lags=(0, 7, 14, 21, 28)):
    """Price follows copper (in BDT) with a lag; forecast = last price + copper-implied change."""
    if mkt is None or len(df) < 10:
        return None
    mkt = mkt[mkt.index <= df["date"].iloc[-1] + pd.Timedelta(days=h)]
    best = None
    for L in lags:
        c = mkt["copper_bdt"].reindex(df["date"] - pd.Timedelta(days=L), method="ffill").values
        ok = ~np.isnan(c)
        if ok.sum() < 10:
            continue
        x, y = np.log(c[ok]), np.log(df["price"].values[ok])
        b, a = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]
        if best is None or abs(r) > abs(best[0]):
            best = (r, L, a, b)
    if best is None:
        return None
    r, L, a, b = best
    last_t = df["date"].iloc[-1]
    c_now = mkt["copper_bdt"].asof(last_t - pd.Timedelta(days=L))
    c_fut = mkt["copper_bdt"].asof(last_t + pd.Timedelta(days=h - L))  # known if L >= h, else latest
    change = np.exp(b * (np.log(c_fut) - np.log(c_now)))
    f_copper.info = (L, r)
    return df["price"].iloc[-1] * change


def backtest(df, mkt, fn, h=14, start=10):
    errs = []
    for i in range(start, len(df)):
        hist = df.iloc[:i]
        t = hist["date"].iloc[-1] + pd.Timedelta(days=h)
        fut = df[df["date"] >= t]
        if fut.empty:
            break
        m = mkt[mkt.index <= hist["date"].iloc[-1]] if mkt is not None else None
        pred = fn(hist, m, h)
        if pred is None:
            return None
        errs.append(abs(pred - fut["price"].iloc[0]))
    return (float(np.mean(errs)), len(errs)) if errs else None


def decide(df, mkt, h):
    """Pick the forecaster that beat 'no change' in backtests; otherwise use 'no change'."""
    naive_bt = backtest(df, mkt, f_naive, min(h, 21))
    cop_bt = backtest(df, mkt, f_copper, min(h, 21))
    use_copper = bool(naive_bt and cop_bt and cop_bt[0] < 0.95 * naive_bt[0])
    mid = f_copper(df, mkt, h) if use_copper else f_naive(df, mkt, h)
    band = vol_band(df, h)
    info = getattr(f_copper, "info", None) if use_copper else None
    return dict(mid=mid, lo=mid - band, hi=mid + band, use_copper=use_copper,
                naive_bt=naive_bt, cop_bt=cop_bt, info=info)


def advice(last, mid, lo, hi):
    pct = (mid - last) / last * 100
    if lo > last:
        return "BUY NOW", "Even the low end of the range is above today's price.", "green"
    if hi < last:
        return "WAIT", "Even the high end of the range is below today's price.", "red"
    if pct > 1:
        return "BUY NOW (or most of it)", f"Expected to rise about {pct:.1f}%.", "green"
    if pct < -1:
        return "BUY ONLY WHAT YOU NEED, WAIT ON THE REST", f"Expected to fall about {abs(pct):.1f}%.", "orange"
    return "SPLIT THE ORDER", "No clear direction; spread purchases over 2-3 dates.", "orange"
