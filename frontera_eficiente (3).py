"""
Frontera eficiente de Markowitz con 10 activos, 1000 portafolios simulados y la CML.
Datos: precios de cierre de Yahoo Finance (yfinance). Retornos ANUALES.

Requisitos:  pip install -r requirements.txt
Uso:         python frontera_eficiente.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ------------------------- PARÁMETROS (editables) -------------------------
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
           "JPM", "JNJ", "XOM", "PG", "KO"]
INICIO = "2010-01-01"
FIN = "2025-12-31"          # último año completo
TASA_LIBRE_RIESGO = 0.04    # anual (ej. T-Bill 4%)
N_PORTAFOLIOS = 1000
SEMILLA = 42
# --------------------------------------------------------------------------


def descargar_precios(tickers, inicio, fin):
    import yfinance as yf
    datos = yf.download(tickers, start=inicio, end=fin,
                        auto_adjust=False, progress=False)
    precios = datos["Close"][tickers]          # precios de cierre
    return precios.dropna()


def retornos_anuales(precios):
    # Último cierre de cada año -> variación porcentual año contra año
    cierres_anuales = precios.resample("YE").last()
    return cierres_anuales.pct_change().dropna()


def stats_portafolio(w, mu, cov):
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    return ret, vol


def simular_portafolios(mu, cov, rf, n, semilla):
    rng = np.random.default_rng(semilla)
    pesos = rng.random((n, len(mu)))
    pesos /= pesos.sum(axis=1, keepdims=True)  # suman 1, sin ventas en corto
    rets = pesos @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", pesos, cov, pesos))
    sharpe = (rets - rf) / vols
    return pesos, rets, vols, sharpe


def optimizar(mu, cov, rf):
    n = len(mu)
    bounds = [(0, 1)] * n
    suma1 = {"type": "eq", "fun": lambda w: w.sum() - 1}
    w0 = np.ones(n) / n

    # Mínima varianza
    min_var = minimize(lambda w: w @ cov @ w, w0, bounds=bounds,
                       constraints=[suma1], method="SLSQP").x
    # Tangente (máximo Sharpe)
    tang = minimize(lambda w: -((w @ mu - rf) / np.sqrt(w @ cov @ w)), w0,
                    bounds=bounds, constraints=[suma1], method="SLSQP").x

    # Frontera eficiente: mínima volatilidad para cada retorno objetivo
    r_min = min_var @ mu
    r_max = mu.max()
    frontera = []
    for r_obj in np.linspace(r_min, r_max, 60):
        cons = [suma1, {"type": "eq", "fun": lambda w, r=r_obj: w @ mu - r}]
        res = minimize(lambda w: w @ cov @ w, w0, bounds=bounds,
                       constraints=cons, method="SLSQP")
        if res.success:
            frontera.append((np.sqrt(res.fun), r_obj))
    return min_var, tang, np.array(frontera)


def main(retornos=None):
    if retornos is None:
        precios = descargar_precios(TICKERS, INICIO, FIN)
        retornos = retornos_anuales(precios)

    print("Retornos anuales (precios de cierre):")
    print(retornos.round(4), "\n")

    mu = retornos.mean().values          # retorno esperado anual
    cov = retornos.cov().values          # covarianza anual
    tickers = list(retornos.columns)
    rf = TASA_LIBRE_RIESGO

    pesos, rets, vols, sharpe = simular_portafolios(mu, cov, rf,
                                                    N_PORTAFOLIOS, SEMILLA)
    w_mv, w_tg, frontera = optimizar(mu, cov, rf)
    r_mv, v_mv = stats_portafolio(w_mv, mu, cov)
    r_tg, v_tg = stats_portafolio(w_tg, mu, cov)
    sharpe_tg = (r_tg - rf) / v_tg

    # Mejor de los 1000 simulados
    i_best = sharpe.argmax()

    # ---- Resultados ----
    tabla = pd.DataFrame({"Mín. varianza": w_mv,
                          "Tangente (máx. Sharpe)": w_tg,
                          "Mejor simulado": pesos[i_best]},
                         index=tickers)
    print("Pesos óptimos:")
    print((tabla * 100).round(2).astype(str) + " %", "\n")
    print(f"Mín. varianza   -> Ret {r_mv:.2%} | Vol {v_mv:.2%} | "
          f"Sharpe {(r_mv - rf) / v_mv:.3f}")
    print(f"Tangente        -> Ret {r_tg:.2%} | Vol {v_tg:.2%} | "
          f"Sharpe {sharpe_tg:.3f}")
    print(f"Mejor simulado  -> Ret {rets[i_best]:.2%} | Vol {vols[i_best]:.2%} | "
          f"Sharpe {sharpe[i_best]:.3f}")
    print(f"CML: E(R) = {rf:.2%} + {sharpe_tg:.3f} · σ")

    # ---- Gráfico ----
    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(vols, rets, c=sharpe, cmap="viridis", s=12, alpha=0.7)
    plt.colorbar(sc, label="Ratio de Sharpe")
    ax.plot(frontera[:, 0], frontera[:, 1], "k-", lw=2.2,
            label="Frontera eficiente")

    x_cml = np.linspace(0, max(vols.max(), frontera[:, 0].max()) * 1.1, 100)
    ax.plot(x_cml, rf + sharpe_tg * x_cml, "r--", lw=2,
            label=f"CML (pendiente = {sharpe_tg:.2f})")

    ax.scatter(v_tg, r_tg, marker="*", s=400, c="red", edgecolor="k",
               zorder=5, label="Portafolio tangente")
    ax.scatter(v_mv, r_mv, marker="D", s=120, c="orange", edgecolor="k",
               zorder=5, label="Mínima varianza")
    ax.scatter(0, rf, marker="o", s=80, c="black", zorder=5,
               label=f"Tasa libre de riesgo ({rf:.1%})")

    vol_ind = np.sqrt(np.diag(cov))
    ax.scatter(vol_ind, mu, marker="x", s=70, c="gray")
    for t, x, y in zip(tickers, vol_ind, mu):
        ax.annotate(t, (x, y), xytext=(5, 3), textcoords="offset points",
                    fontsize=8, color="dimgray")

    ax.set_xlabel("Volatilidad anual (σ)")
    ax.set_ylabel("Retorno esperado anual")
    ax.set_title(f"Frontera eficiente – {len(tickers)} activos, "
                 f"{N_PORTAFOLIOS} portafolios y CML")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(left=0)
    y_top = max(rets.max(), frontera[:, 1].max(), mu.max())
    ax.set_ylim(min(0, mu.min()) - 0.02, y_top * 1.25)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("frontera_eficiente.png", dpi=150)
    tabla.to_csv("pesos_optimos.csv")
    plt.show()


if __name__ == "__main__":
    main()
