import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from model import load_invoices, decide, advice

st.set_page_config(page_title="Brass: buy or wait?", layout="centered")


@st.cache_data(ttl=6 * 3600)
def get_market():
    """Downloads copper and USD/BDT. Returns None if the download fails (app then still works)."""
    try:
        import yfinance as yf
        def get(sym):
            return yf.download(sym, start="2026-01-01", progress=False, auto_adjust=True)["Close"].squeeze()
        m = pd.concat([get("HG=F").rename("copper_usd_lb"), get("USDBDT=X").rename("usdbdt")], axis=1).ffill().dropna()
        if len(m) < 30:
            return None
        m.index = pd.to_datetime(m.index).tz_localize(None)
        m["copper_bdt"] = m["copper_usd_lb"] * m["usdbdt"]
        return m
    except Exception:
        return None


df = load_invoices("invoices.csv")
mkt = get_market()
st.title("Scrap brass: buy or wait?")
h = st.slider("Look ahead (days)", 7, 45, 21)
quote = st.number_input("Today's supplier quote (optional)", min_value=0.0, step=1.0)
if quote > 0:
    df = pd.concat([df, pd.DataFrame({"date": [pd.Timestamp.today().normalize()], "price": [quote]})], ignore_index=True)

last = df["price"].iloc[-1]
r = decide(df, mkt, h)
label, why, color = advice(last, r["mid"], r["lo"], r["hi"])
st.markdown(f"## :{color}[{label}]")
st.write(why)
c1, c2 = st.columns(2)
c1.metric("Last price", f"{last:.0f}")
c2.metric(f"Expected in {h} days", f"{r['mid']:.0f}", f"{r['mid'] - last:+.0f}")
st.write(f"Likely range (~80%): **{r['lo']:.0f} to {r['hi']:.0f}**")

fut = df["date"].iloc[-1] + pd.Timedelta(days=h)
fig = go.Figure()
fig.add_scatter(x=df["date"], y=df["price"], mode="lines+markers", name="Your purchases")
fig.add_scatter(x=[df["date"].iloc[-1], fut], y=[last, r["mid"]], mode="lines", line=dict(dash="dash"), name="Forecast")
fig.add_scatter(x=[fut, fut], y=[r["lo"], r["hi"]], mode="lines", line=dict(width=8), name="Range")
fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

st.subheader("How reliable is this?")
if r["use_copper"]:
    L, corr = r["info"]
    st.write(f"Using copper (in BDT). On past data your price followed copper with a ~{L}-day delay (correlation {corr:.2f}), "
             f"and it beat the 'no change' guess: error {r['cop_bt'][0]:.1f} vs {r['naive_bt'][0]:.1f}.")
else:
    st.write("Copper did not beat a simple 'price stays the same' guess on your history, so the app uses that and "
             "shows the typical range. This will update automatically as more data builds up.")
if r["naive_bt"]:
    st.caption(f"Tested on {r['naive_bt'][1]} past points. Small sample: treat as a guide, not a guarantee.")
if mkt is not None:
    st.caption(f"Market data last updated: {mkt.index[-1].date()}")
else:
    st.caption("Copper data could not be downloaded right now, so the simple range is shown.")
