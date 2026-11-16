"""
Frontera eficiente de Markowitz (app de Streamlit)
- Precios de cierre de Yahoo Finance
- Retornos anuales (último cierre de cada año)
- Hasta 10 activos y 1000 portafolios simulados
- Frontera eficiente optimizada + CML

Local:  pip install -r requirements.txt
        streamlit run frontera_eficiente.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

st.set_page_config(page_title="Frontera eficiente", layout="wide")
st.title("Frontera eficiente de Markowitz")

# ----------------------------- PARÁMETROS -----------------------------
with st.sidebar:
    st.header("Parámetros")
    tickers_txt = st.text_input(
        "Activos (máx. 10, separados por coma)",
        "AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, JNJ, XOM, PG, KO")
    inicio = st.date_input("Desde", pd.Timestamp("2010-01-01"))
    fin = st.date_input("Hasta", pd.Timestamp("2025-12-31"))
    n_port = st.number_input("Cantidad de portafolios", 100, 1000, 1000, 100)
    rf = st.number_input("Tasa libre de riesgo anual (%)", 0.0, 20.0, 4.0, 0.25) / 100
    mostrar_cml = st.checkbox("Mostrar CML", True)

tickers = [t.strip().upper() for t in tickers_txt.split(",") if t.strip()]
if len(tickers) < 2:
    st.error("Ingresá al menos 2 activos.")
    st.stop()
if len(tickers) > 10:
    st.warning("Se usan solo los primeros 10 activos.")
    tickers = tickers[:10]


# ----------------------------- DATOS -----------------------------
@st.cache_data(show_spinner="Descargando precios de Yahoo Finance...")
def precios_cierre(tickers, inicio, fin):
    datos = yf.download(list(tickers), start=str(inicio), end=str(fin),
                        auto_adjust=False, progress=False)
    cierre = datos["Close"]
    if isinstance(cierre, pd.Series):
        cierre = cierre.to_frame(tickers[0])
    return cierre.dropna(axis=1, how="all")


precios = precios_cierre(tuple(tickers), inicio, fin)
faltantes = [t for t in tickers if t not in precios.columns]
if faltantes:
    st.warning(f"Sin datos para: {', '.join(faltantes)}")
tickers = [t for t in tickers if t in precios.columns]
precios = precios[tickers].dropna()

# Retornos anuales: último cierre de cada año
cierres_anuales = precios.resample("YE").last()
retornos = cierres_anuales.pct_change().dropna()

if len(retornos) < 3 or len(tickers) < 2:
    st.error("No hay suficientes años de datos. Ampliá el rango de fechas.")
    st.stop()

mu = retornos.mean().values
cov = retornos.cov().values
n = len(tickers)

# ----------------------------- SIMULACIÓN -----------------------------
rng = np.random.default_rng(42)
pesos = rng.random((int(n_port), n))
pesos /= pesos.sum(axis=1, keepdims=True)
ret_p = pesos @ mu
vol_p = np.sqrt(np.einsum("ij,jk,ik->i", pesos, cov, pesos))
sharpe_p = (ret_p - rf) / vol_p

# ----------------------------- OPTIMIZACIÓN -----------------------------
limites = [(0, 1)] * n
suma_1 = {"type": "eq", "fun": lambda w: w.sum() - 1}
w0 = np.ones(n) / n

w_mv = minimize(lambda w: w @ cov @ w, w0, bounds=limites,
                constraints=[suma_1], method="SLSQP").x
w_tg = minimize(lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w), w0,
                bounds=limites, constraints=[suma_1], method="SLSQP").x

r_mv, v_mv = w_mv @ mu, np.sqrt(w_mv @ cov @ w_mv)
r_tg, v_tg = w_tg @ mu, np.sqrt(w_tg @ cov @ w_tg)
sharpe_tg = (r_tg - rf) / v_tg

frontera = []
for objetivo in np.linspace(r_mv, mu.max(), 60):
    res = minimize(lambda w: w @ cov @ w, w0, bounds=limites, method="SLSQP",
                   constraints=[suma_1,
                                {"type": "eq", "fun": lambda w, o=objetivo: w @ mu - o}])
    if res.success:
        frontera.append((np.sqrt(res.fun), objetivo))
frontera = np.array(frontera)

# ----------------------------- GRÁFICO -----------------------------
fig, ax = plt.subplots(figsize=(11, 6.5))
sc = ax.scatter(vol_p, ret_p, c=sharpe_p, cmap="viridis", s=12, alpha=0.7)
fig.colorbar(sc, label="Ratio de Sharpe")
ax.plot(frontera[:, 0], frontera[:, 1], "k-", lw=2.2, label="Frontera eficiente")

if mostrar_cml:
    x = np.linspace(0, frontera[:, 0].max() * 1.05, 100)
    ax.plot(x, rf + sharpe_tg * x, "r--", lw=2, label=f"CML (pendiente {sharpe_tg:.2f})")
    ax.scatter(0, rf, c="black", s=60, zorder=5, label=f"Libre de riesgo ({rf:.1%})")

ax.scatter(v_tg, r_tg, marker="*", s=400, c="red", edgecolor="k", zorder=6,
           label="Máximo Sharpe (tangente)")
ax.scatter(v_mv, r_mv, marker="D", s=120, c="orange", edgecolor="k", zorder=6,
           label="Mínima varianza")

vol_ind = np.sqrt(np.diag(cov))
ax.scatter(vol_ind, mu, marker="x", c="gray", s=60)
for t, xv, yv in zip(tickers, vol_ind, mu):
    ax.annotate(t, (xv, yv), xytext=(5, 3), textcoords="offset points",
                fontsize=8, color="dimgray")

ax.set_xlabel("Volatilidad anual (σ)")
ax.set_ylabel("Retorno esperado anual")
ax.set_title(f"{n} activos · {int(n_port)} portafolios · retornos anuales")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlim(left=0)
y_max = max(ret_p.max(), frontera[:, 1].max(), mu.max())
ax.set_ylim(min(0, mu.min()) - 0.02, y_max * 1.25)
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
st.pyplot(fig)

# ----------------------------- RESULTADOS -----------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Portafolios clave")
    resumen = pd.DataFrame(
        {"Retorno": [r_mv, r_tg], "Volatilidad": [v_mv, v_tg],
         "Sharpe": [(r_mv - rf) / v_mv, sharpe_tg]},
        index=["Mínima varianza", "Máximo Sharpe"])
    st.dataframe(resumen.style.format({"Retorno": "{:.2%}", "Volatilidad": "{:.2%}",
                                       "Sharpe": "{:.3f}"}))
    if mostrar_cml:
        st.latex(r"CML:\; E(R) = " + f"{rf * 100:.2f}" + r"\% + "
                 + f"{sharpe_tg:.3f}" + r"\,\sigma")
with c2:
    st.subheader("Pesos óptimos")
    pesos_opt = pd.DataFrame({"Mínima varianza": w_mv, "Máximo Sharpe": w_tg},
                             index=tickers)
    st.dataframe(pesos_opt.style.format("{:.2%}"))
    st.download_button("Descargar pesos (CSV)", pesos_opt.to_csv(),
                       "pesos_optimos.csv", "text/csv")

st.subheader("Retornos anuales (precios de cierre)")
st.dataframe(retornos.style.format("{:.2%}"))
