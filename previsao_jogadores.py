# =============================================================
# TDE 3 — Aprendizado de Máquina Supervisionado
# Tema: Previsão de Nota Geral e Gols de Jogadores de Futebol
# Modelos: Regressão Linear + Random Forest
#
# ARQUIVOS NECESSÁRIOS (na mesma pasta que este script):
#   - jogadores_notas.csv  → usado na Parte 1
#   - jogadores_gols.json  → usado na Parte 2
# =============================================================

import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_absolute_error, r2_score

# ── utilitário ────────────────────────────────────────────────
def carregar_dados(caminho):
    if not os.path.exists(caminho):
        print(f"\n❌ Arquivo não encontrado → {caminho}"); sys.exit(1)
    ext = caminho.lower().split(".")[-1]
    if ext == "csv":
        df = pd.read_csv(caminho)
    elif ext == "json":
        df = pd.read_json(caminho)
    else:
        print(f"❌ Formato '{ext}' não suportado."); sys.exit(1)
    print(f"✅ {caminho} carregado ({len(df)} registros)")
    return df

def avaliar(y_real, y_prev, unidade=""):
    mae = mean_absolute_error(y_real, y_prev)
    r2  = r2_score(y_real, y_prev)
    print(f"  Erro Médio Absoluto (MAE): {mae:.2f}{unidade}")
    print(f"  Coeficiente R²           : {r2:.4f}")
    status = "✅ Boa capacidade!" if r2 >= 0.8 else ("⚠️  Desempenho razoável." if r2 >= 0.5 else "❌ Precisa de mais dados.")
    print(f"  {status}")
    return mae, r2

def comparar_modelos(nome, X, y, features, unidade, random_state=42):
    """Treina Regressão Linear e Random Forest, imprime comparação."""
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, np.arange(len(y)), test_size=0.3, random_state=random_state)

    print(f"\n  Dados de treino: {len(X_tr)} | Dados de teste: {len(X_te)}")

    # ── Regressão Linear ──
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    y_lr = lr.predict(X_te)

    # ── Random Forest ──
    rf = RandomForestRegressor(n_estimators=200, max_depth=6,
                               min_samples_leaf=3, random_state=random_state)
    rf.fit(X_tr, y_tr)
    y_rf = rf.predict(X_te)

    print("\n  [Regressão Linear]")
    mae_lr, r2_lr = avaliar(y_te, y_lr, unidade)

    print("\n  [Random Forest]")
    mae_rf, r2_rf = avaliar(y_te, y_rf, unidade)

    # ── Importância das features (Random Forest) ──
    print("\n  Importância das features (Random Forest):")
    imp = sorted(zip(features, rf.feature_importances_), key=lambda x: -x[1])
    for feat, v in imp:
        barra = "█" * int(v * 40)
        print(f"    {feat:<22} {barra} {v*100:.1f}%")

    # ── Melhor modelo ──
    melhor = "Random Forest" if r2_rf > r2_lr else "Regressão Linear"
    print(f"\n  🏆 Melhor modelo: {melhor}")

    return rf if r2_rf > r2_lr else lr, X_te, y_te, idx_te, (y_rf if r2_rf > r2_lr else y_lr)


# =============================================================
# PARTE 1 — PREVISÃO DE NOTA GERAL  (CSV)
# Features: velocidade, chute, passe, resistencia, drible
# =============================================================
print("=" * 62)
print("   PARTE 1 — PREVISÃO DE NOTA GERAL DO JOGADOR")
print("=" * 62)

df = carregar_dados("jogadores_notas.csv")
print(f"  Faixa de notas: {df['nota_geral'].min()} – {df['nota_geral'].max()}")
print(df.head(5).to_string(index=False))

features1 = ["velocidade", "chute", "passe", "resistencia", "drible"]
X1 = df[features1].values
y1 = df["nota_geral"].values

modelo1, X1_te, y1_te, idx1_te, y1_prev = comparar_modelos(
    "Nota Geral", X1, y1, features1, " pts")

# previsão detalhada
print("\n  Resultados (amostra — 10 jogadores):")
print(f"  {'Jogador':<22} {'Real':>5} {'Previsto':>9} {'Erro':>6}")
print("  " + "-" * 46)
for i in range(min(10, len(idx1_te))):
    nome = df.iloc[idx1_te[i]]["jogador"]
    r, p = y1_te[i], round(y1_prev[i], 1)
    print(f"  {nome:<22} {r:>5} {p:>9} {abs(p-r):>6.1f}")

# novo jogador
print("\n  PREVISÃO — NOVO JOGADOR:")
nj = {"velocidade":85,"chute":80,"passe":88,"resistencia":83,"drible":82}
print(f"  Atributos: {nj}")
Xnj = np.array([[nj[f] for f in features1]])
print(f"  → Nota geral prevista: {modelo1.predict(Xnj)[0]:.1f}")

print("\n" + "=" * 62)
print("Fim da Parte 1.")


# =============================================================
# PARTE 2 — PREVISÃO DE GOLS NA TEMPORADA  (JSON)
# Features: posição, nota_geral, chute, jogos, assistências
# =============================================================
print("\n" + "=" * 62)
print("   PARTE 2 — PREVISÃO DE GOLS NA TEMPORADA")
print("=" * 62)

posicao_map = {"Atacante": 3, "Meia": 2, "Defensor": 1, "Goleiro": 0}

df2 = carregar_dados("jogadores_gols.json")
df2["posicao_cod"] = df2["posicao"].map(posicao_map)

if df2["posicao_cod"].isnull().any():
    inv = df2[df2["posicao_cod"].isnull()]["posicao"].unique()
    print(f"❌ Posições inválidas: {inv}"); sys.exit(1)

print(f"  Faixa de gols: {df2['gols'].min()} – {df2['gols'].max()}")
print(df2[["jogador","posicao","nota_geral","chute","jogos","assistencias","gols"]].head(5).to_string(index=False))

features2 = ["posicao_cod", "nota_geral", "chute", "jogos", "assistencias"]
X2 = df2[features2].values
y2 = df2["gols"].values

modelo2, X2_te, y2_te, idx2_te, y2_prev = comparar_modelos(
    "Gols", X2, y2, features2, " gols")

# previsão detalhada
print("\n  Resultados (amostra — 10 jogadores):")
print(f"  {'Jogador':<22} {'Posição':<10} {'Real':>5} {'Previsto':>9} {'Erro':>6}")
print("  " + "-" * 56)
for i in range(min(10, len(idx2_te))):
    nome = df2.iloc[idx2_te[i]]["jogador"]
    pos  = df2.iloc[idx2_te[i]]["posicao"]
    r, p = y2_te[i], round(max(0, y2_prev[i]), 1)
    print(f"  {nome:<22} {pos:<10} {r:>5} {p:>9} {abs(p-r):>6.1f}")

# novo jogador
print("\n  PREVISÃO — NOVO JOGADOR:")
nj2 = {"posicao":"Atacante","nota_geral":88,"chute":84,"jogos":30,"assistencias":8}
print(f"  Atributos: {nj2}")
Xnj2 = np.array([[posicao_map[nj2["posicao"]], nj2["nota_geral"],
                   nj2["chute"], nj2["jogos"], nj2["assistencias"]]])
gols = max(0, modelo2.predict(Xnj2)[0])
print(f"  → Gols previstos na temporada: {gols:.1f}")

print("\n" + "=" * 62)
print("Fim da execução.")
