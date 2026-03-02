"""
stats_page.py
───────────────────────────────────────────────────────────────────────
Page Statistiques Avancée — Audit Sémantique Loi de Finances Cameroun
───────────────────────────────────────────────────────────────────────
Analyses couvertes :
  1. Vue d'ensemble — KPIs clés & radar budgétaire
  2. Glissement sémantique — similarité cosinus, KS-test, articles extrêmes
  3. Tests de significativité — Chi², Mann-Whitney, Cohen's d
  4. Évolution thématique — Topics LDA & piliers SND30
  5. Clustering comparatif — K-Means / HDBSCAN, métriques, transitions

Intégration dans app_dash.py :
    from audit_semantique.dashboard.stats_page import (
        create_stats_page, register_stats_callbacks
    )
    # Dans display_page() :
    elif pathname == '/stats':
        return create_stats_page(DATA)
    # Après init de app :
    register_stats_callbacks(app, DATA)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import (
    chi2_contingency,
    mannwhitneyu,
    ks_2samp,
    shapiro,
    ttest_ind,
    spearmanr,
)
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ─── Constantes ────────────────────────────────────────────────────────────

PILIERS_SND30 = [
    "Transformation structurelle de l'économie pour accélérer la croissance economique",
    "Développement du capital humain et du bien-être social",
    "Promotion de l'emploi et de l'insertion socio-économique",
    "Gouvernance, décentralisation et gestion stratégique de l'État",
]
PILIERS_COURT = [
    "Transformation structurelle",
    "Capital humain",
    "Emploi et insertion",
    "Gouvernance",
]
PILIER_MAP = dict(zip(PILIERS_SND30, PILIERS_COURT))

PILIER_PALETTE = {
    "Transformation structurelle": "#f97316",
    "Capital humain": "#22c55e",
    "Emploi et insertion": "#a855f7",
    "Gouvernance": "#0ea5e9",
}

YEAR_COLORS = {"2024": "#3b82f6", "2025": "#f97316"}

# ─── Helpers statistiques ──────────────────────────────────────────────────

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d avec pooled SD (Welch-like)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    pooled = np.sqrt((x.std(ddof=1)**2 + y.std(ddof=1)**2) / 2)
    return float((x.mean() - y.mean()) / pooled) if pooled > 0 else float("nan")


def interpret_d(d: float) -> str:
    if np.isnan(d): return "N/A"
    ad = abs(d)
    if ad < 0.2:  return "négligeable"
    if ad < 0.5:  return "faible"
    if ad < 0.8:  return "moyen"
    return "fort"


def interpret_p(p: float) -> tuple[str, str]:
    """(niveau significativité, icône)"""
    if p < 0.001: return "***  p<0.001", "danger"
    if p < 0.01:  return "**   p<0.01",  "warning"
    if p < 0.05:  return "*    p<0.05",  "info"
    return "n.s.  p≥0.05", "secondary"


def herfindahl(shares: np.ndarray) -> float:
    """Indice Herfindahl-Hirschman (concentration budgétaire)."""
    s = np.asarray(shares, dtype=float)
    s = s / s.sum()
    return float((s**2).sum())


def _safe(val, fmt=".2f"):
    try:
        if np.isnan(val): return "N/A"
        return format(val, fmt)
    except Exception:
        return str(val)


def _map_pilier(val) -> str:
    return PILIER_MAP.get(str(val), str(val))


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convertit une série en numérique (NaN si impossible)."""
    return pd.to_numeric(series, errors="coerce")


def _gini(x: np.ndarray | pd.Series) -> float:
    """Coefficient de Gini (copié de la logique du dashboard principal)."""
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr) & (arr >= 0)]
    if arr.size == 0:
        return float("nan")
    if np.all(arr == 0):
        return 0.0
    arr_sorted = np.sort(arr)
    n = arr_sorted.size
    cum = np.cumsum(arr_sorted)
    return float((2.0 * np.sum((np.arange(1, n + 1) * arr_sorted)) / (n * cum[-1]) - (n + 1) / n))


def _lorenz_curve(x: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (x_lorenz, y_lorenz) pour la courbe de Lorenz normalisée."""
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr) & (arr >= 0)]
    if arr.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    s = np.sort(arr)
    cum = np.cumsum(s) / s.sum()
    y = np.concatenate([[0.0], cum])
    x_l = np.linspace(0.0, 1.0, len(y))
    return x_l, y


# ─── Chargement des données ────────────────────────────────────────────────

def _load_stats_data(DATA: dict) -> dict:
    """Extrait et prépare toutes les données nécessaires à l'onglet stats."""
    out = {}

    # --- Budgets bruts
    out["budget_2024"] = DATA.get("budget_2024", pd.DataFrame())
    out["budget_2025"] = DATA.get("budget_2025", pd.DataFrame())

    # --- Piliers agrégés
    out["piliers_2024"] = DATA.get("piliers_2024", pd.DataFrame())
    out["piliers_2025"] = DATA.get("piliers_2025", pd.DataFrame())
    out["comparison"]   = DATA.get("comparison",   pd.DataFrame())

    # --- Articles avec topics
    out["art_topics_24"] = DATA.get("articles_topics_2024", pd.DataFrame())
    out["art_topics_25"] = DATA.get("articles_topics_2025", pd.DataFrame())

    # --- Articles avec clusters
    out["art_clusters_24"] = DATA.get("articles_clusters_2024", pd.DataFrame())
    out["art_clusters_25"] = DATA.get("articles_clusters_2025", pd.DataFrame())

    # --- Topics LDA
    out["lda_24"] = DATA.get("lda_topics_2024", pd.DataFrame())
    out["lda_25"] = DATA.get("lda_topics_2025", pd.DataFrame())

    # --- Objectifs classifiés
    out["obj_24"] = DATA.get("objectifs_classifications_2024", pd.DataFrame())
    out["obj_25"] = DATA.get("objectifs_classifications_2025", pd.DataFrame())

    # --- Similarités embeddings (agrégé ou par année)
    sim_df = DATA.get("embeddings_similarities")
    if isinstance(sim_df, pd.DataFrame) and not sim_df.empty:
        out["sim_df"] = sim_df
    else:
        parts = []
        sim24 = DATA.get("similarities_2024")
        sim25 = DATA.get("similarities_2025")
        if isinstance(sim24, pd.DataFrame) and not sim24.empty:
            tmp = sim24.copy()
            if "annee" not in tmp.columns:
                tmp["annee"] = 2024
            parts.append(tmp)
        if isinstance(sim25, pd.DataFrame) and not sim25.empty:
            tmp = sim25.copy()
            if "annee" not in tmp.columns:
                tmp["annee"] = 2025
            parts.append(tmp)
        out["sim_df"] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # --- Distributions topics
    out["dist_24"] = DATA.get("topic_dists_2024")
    out["dist_25"] = DATA.get("topic_dists_2025")

    # --- Résultats de tests déjà calculés (Excel)
    out["chi2_results"] = DATA.get("chi2", pd.DataFrame())
    out["mw_results"] = DATA.get("mannwhitney", pd.DataFrame())

    return out


# ─── Composants visuels partagés ──────────────────────────────────────────

def _kpi_card(title: str, value: str, subtitle: str = "",
              color: str = "#1e3a5f", badge: str = "") -> dbc.Card:
    """Carte KPI compacte et percutante."""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.P(title, className="text-muted mb-1",
                       style={"fontSize": "0.75rem", "fontWeight": 600,
                              "textTransform": "uppercase", "letterSpacing": "0.08em"}),
                html.H3(value, style={"fontWeight": 800, "color": color,
                                       "fontSize": "1.6rem", "margin": 0}),
                html.P(subtitle, className="text-muted mt-1",
                       style={"fontSize": "0.8rem"}) if subtitle else html.Div(),
                dbc.Badge(badge, color="light", text_color="dark",
                          className="mt-1") if badge else html.Div(),
            ])
        ])
    ], className="h-100 shadow-sm",
       style={"borderRadius": "12px", "border": f"1.5px solid {color}22",
               "borderLeft": f"4px solid {color}"})


def _section_header(title: str, subtitle: str = "") -> html.Div:
    return html.Div([
        html.H4(title, style={"fontWeight": 700, "color": "#1e3a5f",
                               "borderBottom": "2px solid #2563eb",
                               "paddingBottom": "8px", "marginBottom": "4px"}),
        html.P(subtitle, className="text-muted",
               style={"fontSize": "0.85rem"}) if subtitle else html.Div(),
    ], className="mb-4 mt-4")


def _test_result_card(test_name: str, statistic: str, p_val: float,
                       interpretation: str, color: str = "primary") -> dbc.Card:
    sig_text, badge_color = interpret_p(p_val)
    return dbc.Card([
        dbc.CardBody([
            html.H6(test_name, className="mb-2",
                    style={"fontWeight": 700, "color": "#1e3a5f"}),
            html.Div([
                dbc.Badge(f"Stat = {statistic}", color=color, className="me-2"),
                dbc.Badge(f"p = {p_val:.4f}", color=badge_color, className="me-2"),
                dbc.Badge(sig_text.strip(), color="dark"),
            ], className="mb-2"),
            html.P(interpretation, style={"fontSize": "0.82rem",
                                           "color": "#374151", "margin": 0}),
        ])
    ], className="mb-3 shadow-sm",
       style={"borderRadius": "10px", "border": "1px solid #e2e8f0"})


# ═════════════════════════════════════════════════════════════════════════════=
# TABS — CONTENU
# ═════════════════════════════════════════════════════════════════════════════=

def _tab_overview(d: dict) -> html.Div:
    """Vue d'ensemble : KPIs clés et répartition par piliers."""
    art24 = d["art_topics_24"]
    art25 = d["art_topics_25"]
    piliers24 = d["piliers_2024"]
    piliers25 = d["piliers_2025"]
    budget24 = d.get("budget_2024", pd.DataFrame())
    budget25 = d.get("budget_2025", pd.DataFrame())

    n_art_24 = len(art24) if not art24.empty else 0
    n_art_25 = len(art25) if not art25.empty else 0

    # Budget total (priorité aux feuilles Budget_2024/2025, comme sur la page Budget)
    def _total_budget(df):
        if df is None or df.empty:
            return np.nan
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) == 0:
            return np.nan
        return df[num_cols[0]].sum()

    if not budget24.empty and "ae" in budget24.columns:
        tot24 = float(budget24["ae"].sum())
    else:
        tot24 = _total_budget(piliers24)

    if not budget25.empty and "ae" in budget25.columns:
        tot25 = float(budget25["ae"].sum())
    else:
        tot25 = _total_budget(piliers25)

    # Nombre de piliers utilisés
    def _n_piliers(df):
        if df is None or df.empty:
            return 0
        # chaque ligne correspond déjà à un pilier (index)
        return df.shape[0]

    n_pil_24 = _n_piliers(piliers24)
    n_pil_25 = _n_piliers(piliers25)

    cards = dbc.Row([
        dbc.Col(_kpi_card("Articles 2024", str(n_art_24),
                          "Nombre d'articles audités en 2024",
                          color=YEAR_COLORS["2024"]), md=3),
        dbc.Col(_kpi_card("Articles 2025", str(n_art_25),
                          "Nombre d'articles audités en 2025",
                          color=YEAR_COLORS["2025"]), md=3),
        dbc.Col(_kpi_card("Budget total 2024", _safe(tot24, ".1f"),
                          "Somme du premier indicateur budgétaire",
                          color=YEAR_COLORS["2024"]), md=3),
        dbc.Col(_kpi_card("Budget total 2025", _safe(tot25, ".1f"),
                          "Somme du premier indicateur budgétaire",
                          color=YEAR_COLORS["2025"]), md=3),
    ], className="gy-3")

    # Diagramme en barres par pilier si possible
    fig_piliers = None
    if not piliers24.empty and not piliers25.empty:
        def _prepare_piliers(df, year):
            out = df.copy()
            if out.empty:
                return out
            # dans analyse_budgetaire.xlsx, le pilier est l'index
            out = out.reset_index()
            pilier_col = out.columns[0]
            out["Pilier_court"] = out[pilier_col].astype(str)
            # on privilégie la colonne 'ae' si elle existe, sinon première numérique
            if "ae" in out.columns:
                mcol = "ae"
            else:
                num_cols = out.select_dtypes(include="number").columns
                if len(num_cols) == 0:
                    return pd.DataFrame()
                mcol = num_cols[0]
            g = out[["Pilier_court", mcol]].copy()
            g["Année"] = str(year)
            g.rename(columns={mcol: "Montant"}, inplace=True)
            return g

        g24 = _prepare_piliers(piliers24, 2024)
        g25 = _prepare_piliers(piliers25, 2025)
        if not g24.empty and not g25.empty:
            g = pd.concat([g24, g25], ignore_index=True)
            fig_piliers = px.bar(
                g, x="Pilier_court", y="Montant", color="Année",
                barmode="group", color_discrete_map=YEAR_COLORS,
                title="Répartition budgétaire par pilier"
            )
            fig_piliers.update_layout(
                legend_title_text="Année", template="plotly_white",
                margin=dict(t=60, l=10, r=10, b=10),
            )

    return html.Div([
        _section_header(
            "Vue d'ensemble",
            "Volume d'articles audités et grands équilibres budgétaires par pilier."
        ),
        cards,
        html.Hr(className="my-4"),
        html.Div([
            dcc.Graph(figure=fig_piliers)
            if fig_piliers is not None
            else dbc.Alert(
                "Impossible de construire la répartition par pilier (colonnes non détectées).",
                color="secondary",
            )
        ]),
    ])


def _tab_semantic(d: dict) -> html.Div:
    """Glissement sémantique basé sur les similarités d'embeddings si dispo."""
    sim_df = d["sim_df"]
    if sim_df is None or sim_df.empty:
        return html.Div([
            _section_header("Glissement sémantique"),
            dbc.Alert(
                "Aucune matrice de similarité d'embeddings n'est disponible dans les données chargées.",
                color="warning",
            ),
        ])

    # On privilégie la structure produite par main_pipeline :
    # id, score_max_similarite, annee
    score_col = None
    if "score_max_similarite" in sim_df.columns:
        score_col = "score_max_similarite"
    elif "similarity" in sim_df.columns:
        score_col = "similarity"
    else:
        num_cols = sim_df.select_dtypes(include="number").columns
        num_cols = [c for c in num_cols if c.lower() not in {"id", "annee", "year"}]
        if num_cols:
            score_col = num_cols[0]

    if not score_col:
        return html.Div([
            _section_header("Glissement sémantique"),
            dbc.Alert(
                "Aucune colonne de score de similarité détectée (score_max_similarite ou similaire).",
                color="warning",
            ),
        ])

    vals = sim_df[score_col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) == 0:
        return html.Div([
            _section_header("Glissement sémantique"),
            dbc.Alert("Les similarités sont vides après nettoyage.", color="warning"),
        ])

    # Histogramme global par année si la colonne annee est disponible
    if "annee" in sim_df.columns:
        fig = px.histogram(
            sim_df.dropna(subset=[score_col]),
            x=score_col,
            color="annee",
            nbins=40,
            barmode="overlay",
            labels={score_col: "Similarité cosinus"},
            title="Distribution des similarités cosinus par année (embeddings)",
        )
        fig.update_traces(opacity=0.65)
    else:
        fig = px.histogram(
            vals, nbins=40, labels={"value": "Similarité cosinus"},
            title="Distribution globale des similarités entre articles (embeddings)",
        )
    fig.update_layout(template="plotly_white", margin=dict(t=60, l=10, r=10, b=10))

    return html.Div([
        _section_header(
            "Glissement sémantique",
            "Analyse de la stabilité du langage budgétaire à partir des similarités d'embeddings."
        ),
        dbc.Row([
            dbc.Col(_kpi_card(
                "Similarité moyenne", _safe(vals.mean(), ".3f"),
                subtitle="Plus elle est élevée, plus le langage reste stable."
            ), md=3),
            dbc.Col(_kpi_card(
                "Similarité médiane", _safe(vals.median(), ".3f"),
                subtitle="Valeur centrale de la distribution des similarités."
            ), md=3),
            dbc.Col(_kpi_card(
                "Quantile 10%", _safe(vals.quantile(0.1), ".3f"),
                subtitle="Queue basse — zones de plus forte rupture potentielle."
            ), md=3),
            dbc.Col(_kpi_card(
                "Quantile 90%", _safe(vals.quantile(0.9), ".3f"),
                subtitle="Queue haute — continuité maximale du langage."
            ), md=3),
        ], className="gy-3"),
        html.Hr(className="my-4"),
        dcc.Graph(figure=fig),
    ])


def _tab_tests(d: dict) -> html.Div:
    """Tests de significativité sur les topics et les clusters."""
    dist24 = d["dist_24"]
    dist25 = d["dist_25"]
    art24 = d["art_topics_24"]
    art25 = d["art_topics_25"]
    cl24 = d["art_clusters_24"]
    cl25 = d["art_clusters_25"]
    chi2_df = d["chi2_results"]
    mw_df = d["mw_results"]

    children = [_section_header(
        "Tests de significativité",
        "Comparaisons robustes entre 2024 et 2025 : topics, dominance et clusters."
    )]

    # ── Résumé des tests pré-calculés (Excel) s'ils existent ───────────────
    if isinstance(chi2_df, pd.DataFrame) and not chi2_df.empty:
        row = chi2_df.iloc[0]
        chi2_val = row.get("chi2", float("nan"))
        p_val = row.get("p_value", float("nan"))
        dof = row.get("dof", "?")
        signif = row.get("significatif", False)
        interp = (
            f"Test du Chi² sur les piliers SND30 (ddl={dof}). "
            f"Conclusion : {'répartition modifiée de façon significative' if signif else 'pas de réallocation significative entre piliers'}."
        )
        children.insert(1, _test_result_card(
            "Chi² (piliers SND30)",
            f"{chi2_val:.2f}" if pd.notna(chi2_val) else "N/A",
            p_val if pd.notna(p_val) else 1.0,
            interp,
            color="info",
        ))

    if isinstance(mw_df, pd.DataFrame) and not mw_df.empty and "p_value" in mw_df.columns:
        # sujets les plus significatifs selon les tests pré-calculés
        top = mw_df.sort_values("p_value").head(5).copy()
        n_sig = int((mw_df.get("significatif") == True).sum()) if "significatif" in mw_df.columns else None
        subtitle = (
            f"Nombre de topics significatifs (p<{0.05}): {n_sig}" if n_sig is not None
            else "Synthèse des topics les plus significatifs (tests pré-calculés)."
        )
        table_mw = dbc.Table.from_dataframe(
            top,
            striped=True, bordered=False, hover=True, size="sm",
            class_name="mt-2",
        )
        children.extend([
            html.H5("Tests Mann-Whitney pré-calculés (top 5)", className="mt-3"),
            html.P(subtitle, className="text-muted", style={"fontSize": "0.85rem"}),
            table_mw,
            html.Hr(className="my-4"),
        ])

    # ── 1) Mann-Whitney et Cohen's d par topic (probabilités continues) ──
    #     (recalcul à partir des matrices de topics si disponibles)
    if dist24 is not None and dist25 is not None:
        try:
            arr24 = np.asarray(dist24, dtype=float)
            arr25 = np.asarray(dist25, dtype=float)
            k = min(arr24.shape[1], arr25.shape[1]) if arr24.ndim == 2 and arr25.ndim == 2 else 0
        except Exception:
            k = 0
        if k > 0:
            rows = []
            for t in range(k):
                x = arr24[:, t]
                y = arr25[:, t]
                x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
                if len(x) == 0 or len(y) == 0:
                    p = np.nan; d_eff = np.nan
                else:
                    try:
                        _, p = mannwhitneyu(x, y, alternative="two-sided")
                    except Exception:
                        p = np.nan
                    d_eff = cohens_d(x, y)
                rows.append({
                    "Topic": f"Topic {t}",
                    "Moyenne 2024": np.nanmean(x) if len(x) > 0 else np.nan,
                    "Moyenne 2025": np.nanmean(y) if len(y) > 0 else np.nan,
                    "Δ (2025-2024)": (np.nanmean(y) - np.nanmean(x)) if len(x) > 0 and len(y) > 0 else np.nan,
                    "p (Mann-Whitney)": p,
                    "Cohen d": d_eff,
                    "Interprétation d": interpret_d(d_eff),
                })
            df_topics = pd.DataFrame(rows)
            df_topics["Signif."] = df_topics["p (Mann-Whitney)"].apply(
                lambda p: interpret_p(p)[0] if pd.notna(p) else "N/A"
            )

            table = dbc.Table.from_dataframe(
                df_topics.round({"Moyenne 2024": 3, "Moyenne 2025": 3,
                                  "Δ (2025-2024)": 3, "p (Mann-Whitney)": 4,
                                  "Cohen d": 3}),
                striped=True, bordered=False, hover=True, size="sm",
                class_name="mt-2",
            )
            children.extend([
                html.H5("Tests Mann-Whitney par topic (probabilités continues)",
                        className="mt-3"),
                html.P(
                    "Lecture : on compare la distribution de probabilité de chaque topic entre 2024 et 2025.",
                    className="text-muted", style={"fontSize": "0.85rem"}
                ),
                table,
            ])

    # ── 2) Chi² sur topics dominants ──
    if not art24.empty and not art25.empty and "dominant_topic" in art24.columns and "dominant_topic" in art25.columns:
        c24 = art24["dominant_topic"].value_counts().sort_index()
        c25 = art25["dominant_topic"].value_counts().sort_index()
        idx = sorted(set(c24.index).union(set(c25.index)))
        c24 = c24.reindex(idx, fill_value=0)
        c25 = c25.reindex(idx, fill_value=0)
        contingency = np.vstack([c24.values, c25.values])
        try:
            chi2, p_chi, dof, _ = chi2_contingency(contingency)
            n = contingency.sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1))) if n > 0 else np.nan
        except Exception:
            chi2 = p_chi = cramers_v = np.nan
            dof = 0

        children.append(html.Hr(className="my-4"))
        children.append(html.H5("Répartition des topics dominants (Chi²)", className="mt-2"))
        children.append(_test_result_card(
            "Chi² sur la répartition des topics dominants",
            f"{chi2:.2f}" if pd.notna(chi2) else "N/A",
            p_chi if pd.notna(p_chi) else 1.0,
            f"Cramér V = {_safe(cramers_v, '.3f')} (intensité globale du rééquilibrage des thèmes).",
            color="info",
        ))

        df_dom = pd.DataFrame({
            "Topic": [f"Topic {i}" for i in idx],
            "2024": c24.values,
            "2025": c25.values,
        })
        df_melt = df_dom.melt(id_vars="Topic", value_vars=["2024", "2025"],
                              var_name="Année", value_name="Nombre d'articles")
        fig_dom = px.bar(
            df_melt, x="Topic", y="Nombre d'articles", color="Année",
            barmode="group", color_discrete_map=YEAR_COLORS,
            title="Évolution de la dominance thématique par topic",
        )
        fig_dom.update_layout(template="plotly_white", margin=dict(t=60, l=10, r=10, b=10))
        children.append(dcc.Graph(figure=fig_dom))

    # ── 3) Chi² sur clusters ──
    def _get_cluster_col(df: pd.DataFrame):
        for c in df.columns:
            if "cluster" in c.lower():
                return c
        return None

    if not cl24.empty and not cl25.empty:
        col24 = _get_cluster_col(cl24)
        col25 = _get_cluster_col(cl25)
        if col24 and col25:
            vc24 = cl24[col24].value_counts().sort_index()
            vc25 = cl25[col25].value_counts().sort_index()
            idxc = sorted(set(vc24.index).union(set(vc25.index)))
            vc24 = vc24.reindex(idxc, fill_value=0)
            vc25 = vc25.reindex(idxc, fill_value=0)
            cont = np.vstack([vc24.values, vc25.values])
            try:
                chi2_c, p_c, dof_c, _ = chi2_contingency(cont)
                n_c = cont.sum()
                cram_v_c = np.sqrt(chi2_c / (n_c * (min(cont.shape) - 1))) if n_c > 0 else np.nan
            except Exception:
                chi2_c = p_c = cram_v_c = np.nan
                dof_c = 0

            children.append(html.Hr(className="my-4"))
            children.append(html.H5("Structure des clusters (Chi²)", className="mt-2"))
            children.append(_test_result_card(
                "Chi² sur la répartition des clusters",
                f"{chi2_c:.2f}" if pd.notna(chi2_c) else "N/A",
                p_c if pd.notna(p_c) else 1.0,
                f"Cramér V = {_safe(cram_v_c, '.3f')} (réallocation des articles entre clusters).",
                color="primary",
            ))

    return html.Div(children)


def _tab_thematic(d: dict) -> html.Div:
    """Évolution thématique globale : focus sur les topics dominants."""
    art24 = d["art_topics_24"]
    art25 = d["art_topics_25"]

    if art24.empty or art25.empty or "dominant_topic" not in art24.columns or "dominant_topic" not in art25.columns:
        return html.Div([
            _section_header("Évolution thématique"),
            dbc.Alert(
                "Les colonnes de topics dominants ne sont pas disponibles pour les deux années.",
                color="warning",
            ),
        ])

    c24 = art24["dominant_topic"].value_counts(normalize=True).sort_index()
    c25 = art25["dominant_topic"].value_counts(normalize=True).sort_index()
    idx = sorted(set(c24.index).union(set(c25.index)))
    c24 = c24.reindex(idx, fill_value=0)
    c25 = c25.reindex(idx, fill_value=0)

    df = pd.DataFrame({
        "Topic": [f"Topic {i}" for i in idx],
        "Part 2024": c24.values,
        "Part 2025": c25.values,
        "Δ (points de %)": (c25.values - c24.values) * 100,
    })

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Part 2024", "Part 2025"))
    fig.add_trace(go.Bar(x=df["Topic"], y=df["Part 2024"] * 100,
                         name="2024", marker_color=YEAR_COLORS["2024"]), row=1, col=1)
    fig.add_trace(go.Bar(x=df["Topic"], y=df["Part 2025"] * 100,
                         name="2025", marker_color=YEAR_COLORS["2025"]), row=1, col=2)
    fig.update_yaxes(title_text="Part des articles (%)", row=1, col=1)
    fig.update_yaxes(title_text="Part des articles (%)", row=1, col=2)
    fig.update_layout(
        template="plotly_white", showlegend=False,
        title="Répartition relative des thèmes dominants",
        margin=dict(t=60, l=10, r=10, b=10),
    )

    table = dbc.Table.from_dataframe(
        df.round({"Part 2024": 3, "Part 2025": 3, "Δ (points de %)": 1}),
        striped=True, bordered=False, hover=True, size="sm",
        class_name="mt-3",
    )

    return html.Div([
        _section_header(
            "Évolution thématique",
            "Lecture des grands gagnants et perdants parmi les thèmes budgétaires."
        ),
        dcc.Graph(figure=fig),
        html.H5("Tableau de synthèse (parts relatives)", className="mt-3"),
        table,
    ])


def _tab_clustering(d: dict) -> html.Div:
    """Analyse croisée de la structure de clusters entre 2024 et 2025."""
    cl24 = d["art_clusters_24"]
    cl25 = d["art_clusters_25"]

    if cl24.empty or cl25.empty:
        return html.Div([
            _section_header("Clustering"),
            dbc.Alert(
                "Les fichiers d'articles clusterisés ne sont pas disponibles pour les deux années.",
                color="warning",
            ),
        ])

    def _get_cluster_col(df: pd.DataFrame):
        for c in df.columns:
            if "cluster" in c.lower():
                return c
        return None

    col24 = _get_cluster_col(cl24)
    col25 = _get_cluster_col(cl25)
    if not col24 or not col25:
        return html.Div([
            _section_header("Clustering"),
            dbc.Alert(
                "Impossible d'identifier la colonne de cluster (nom contenant 'cluster').",
                color="warning",
            ),
        ])

    vc24 = cl24[col24].value_counts(normalize=True).sort_index()
    vc25 = cl25[col25].value_counts(normalize=True).sort_index()
    idx = sorted(set(vc24.index).union(set(vc25.index)))
    vc24 = vc24.reindex(idx, fill_value=0)
    vc25 = vc25.reindex(idx, fill_value=0)

    df = pd.DataFrame({
        "Cluster": [str(i) for i in idx],
        "Part 2024": vc24.values,
        "Part 2025": vc25.values,
        "Δ (points de %)": (vc25.values - vc24.values) * 100,
    })

    fig = px.bar(
        df.melt(id_vars="Cluster", value_vars=["Part 2024", "Part 2025"],
                var_name="Année", value_name="Part"),
        x="Cluster", y="Part", color="Année", barmode="group",
        color_discrete_map={
            "Part 2024": YEAR_COLORS["2024"],
            "Part 2025": YEAR_COLORS["2025"],
        },
        title="Répartition relative des clusters (2024 vs 2025)",
    )
    fig.update_layout(template="plotly_white", margin=dict(t=60, l=10, r=10, b=10))

    table = dbc.Table.from_dataframe(
        df.round({"Part 2024": 3, "Part 2025": 3, "Δ (points de %)": 1}),
        striped=True, bordered=False, hover=True, size="sm",
        class_name="mt-3",
    )

    return html.Div([
        _section_header(
            "Clustering",
            "Structure des groupes d'articles et déplacements entre clusters."
        ),
        dcc.Graph(figure=fig),
        html.H5("Tableau de synthèse (parts relatives)", className="mt-3"),
        table,
    ])


def _tab_adequation(d: dict) -> html.Div:
    """Onglet d'adéquation budgétaire / sémantique 2024–2025.

    Synthèse compacte de plusieurs résultats du notebook :
      - Portrait budgétaire global (AE / CP et par pilier)
      - Adéquation SND30 des objectifs (nombre, AE, scores)
      - Indicateurs de glissement sémantique et concentration budgétaire
    """
    budget24 = d.get("budget_2024", pd.DataFrame())
    budget25 = d.get("budget_2025", pd.DataFrame())
    piliers24 = d.get("piliers_2024", pd.DataFrame())
    piliers25 = d.get("piliers_2025", pd.DataFrame())
    comparison = d.get("comparison", pd.DataFrame())
    obj24 = d.get("obj_24", pd.DataFrame())
    obj25 = d.get("obj_25", pd.DataFrame())
    sim_df = d.get("sim_df", pd.DataFrame())
    chi2_df = d.get("chi2_results", pd.DataFrame())
    mw_df = d.get("mw_results", pd.DataFrame())

    seuil_sim = 0.70

    # ── 1. Portrait budgétaire global ─────────────────────────────────────
    ae24 = ae25 = cp24 = cp25 = np.nan
    if not budget24.empty and {"ae", "cp"}.issubset(budget24.columns):
        ae24 = float(budget24["ae"].sum()) / 1e9
        cp24 = float(budget24["cp"].sum()) / 1e9
    if not budget25.empty and {"ae", "cp"}.issubset(budget25.columns):
        ae25 = float(budget25["ae"].sum()) / 1e9
        cp25 = float(budget25["cp"].sum()) / 1e9

    fig_macro = go.Figure()
    if not np.isnan(ae24) and not np.isnan(ae25) and not np.isnan(cp24) and not np.isnan(cp25):
        x_labels = ["AE", "CP"]
        fig_macro = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("AE vs CP totaux", "Δ AE par pilier"),
            specs=[[{"type": "bar"}, {"type": "bar"}]],
        )

        fig_macro.add_trace(
            go.Bar(
                x=x_labels,
                y=[ae24, cp24],
                name="2024",
                marker_color=YEAR_COLORS["2024"],
            ),
            row=1,
            col=1,
        )
        fig_macro.add_trace(
            go.Bar(
                x=x_labels,
                y=[ae25, cp25],
                name="2025",
                marker_color=YEAR_COLORS["2025"],
            ),
            row=1,
            col=1,
        )
        fig_macro.update_yaxes(title_text="Mds FCFA", row=1, col=1)

        # Variation AE par pilier si la table comparaison est disponible
        if isinstance(comparison, pd.DataFrame) and not comparison.empty:
            cmp = comparison.copy()
            cmp = cmp.reset_index()
            pilier_col = cmp.columns[0]

            num_cols = [c for c in cmp.columns if cmp[c].dtype.kind in "fi"]
            col_ae24 = "ae_2024" if "ae_2024" in cmp.columns else (num_cols[0] if num_cols else None)
            col_ae25 = "ae_2025" if "ae_2025" in cmp.columns else (num_cols[1] if len(num_cols) > 1 else None)
            if col_ae24 and col_ae25:
                cmp["pilier_court"] = cmp[pilier_col].apply(_map_pilier)
                cmp["delta"] = (cmp[col_ae25] - cmp[col_ae24]) / 1e9
                cmp = cmp.sort_values("delta", ascending=True)
                fig_macro.add_trace(
                    go.Bar(
                        x=cmp["delta"],
                        y=cmp["pilier_court"],
                        orientation="h",
                        marker_color=[
                            "#08f35e" if v >= 0 else "#ef4444" for v in cmp["delta"]
                        ],
                        name="Δ AE (2025-2024)",
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )
                fig_macro.update_xaxes(title_text="Δ AE (Mds FCFA)", row=1, col=2)
        fig_macro.update_layout(
            template="plotly_white",
            title="Portrait macrobudgétaire 2024 vs 2025",
            barmode="group",
            height=520,
        )

    # KPIs de synthèse (croissance AE/CP, objectifs, similarité)
    kpi_cards: list[dbc.Col] = []
    if not np.isnan(ae24) and not np.isnan(ae25):
        croissance_ae = (ae25 / ae24 - 1.0) * 100.0 if ae24 > 0 else np.nan
        kpi_cards.append(
            dbc.Col(
                _kpi_card(
                    "AE totales",
                    f"{ae24:.1f} → {ae25:.1f}",
                    subtitle=f"Croissance AE ≈ {croissance_ae:.1f}% (Mds FCFA)" if np.isfinite(croissance_ae) else "Croissance AE non calculable",
                    color=YEAR_COLORS["2025"],
                ),
                md=3,
            )
        )
    if not np.isnan(cp24) and not np.isnan(cp25):
        croissance_cp = (cp25 / cp24 - 1.0) * 100.0 if cp24 > 0 else np.nan
        kpi_cards.append(
            dbc.Col(
                _kpi_card(
                    "CP totaux",
                    f"{cp24:.1f} → {cp25:.1f}",
                    subtitle=f"Croissance CP ≈ {croissance_cp:.1f}% (Mds FCFA)" if np.isfinite(croissance_cp) else "Croissance CP non calculable",
                    color=YEAR_COLORS["2024"],
                ),
                md=3,
            )
        )

    # Nombre d'objectifs classifiés par année
    n_obj_24 = len(obj24) if isinstance(obj24, pd.DataFrame) and not obj24.empty else 0
    n_obj_25 = len(obj25) if isinstance(obj25, pd.DataFrame) and not obj25.empty else 0
    if n_obj_24 or n_obj_25:
        kpi_cards.append(
            dbc.Col(
                _kpi_card(
                    "Objectifs classifiés",
                    f"{n_obj_24} / {n_obj_25}",
                    subtitle="Nombre d'objectifs SND30 (2024 / 2025)",
                    color="#0f766e",
                ),
                md=3,
            )
        )

    # Similarité moyenne & % sous seuil
    mean_sim_24 = mean_sim_25 = pct_ss_24 = pct_ss_25 = np.nan
    if isinstance(sim_df, pd.DataFrame) and not sim_df.empty:
        score_col = None
        if "score_max_similarite" in sim_df.columns:
            score_col = "score_max_similarite"
        elif "similarity" in sim_df.columns:
            score_col = "similarity"
        if score_col:
            sims = sim_df.dropna(subset=[score_col]).copy()
            if "annee" not in sims.columns and "year" in sims.columns:
                sims["annee"] = sims["year"]
            if "annee" in sims.columns:
                s24 = sims[sims["annee"] == 2024][score_col]
                s25 = sims[sims["annee"] == 2025][score_col]
                if not s24.empty:
                    mean_sim_24 = float(s24.mean())
                    pct_ss_24 = float((s24 < seuil_sim).mean() * 100.0)
                if not s25.empty:
                    mean_sim_25 = float(s25.mean())
                    pct_ss_25 = float((s25 < seuil_sim).mean() * 100.0)
    if not np.isnan(mean_sim_24) or not np.isnan(mean_sim_25):
        val_txt = " / ".join(
            [
                _safe(mean_sim_24, ".3f") if not np.isnan(mean_sim_24) else "—",
                _safe(mean_sim_25, ".3f") if not np.isnan(mean_sim_25) else "—",
            ]
        )
        subtitle = []
        if not np.isnan(pct_ss_24):
            subtitle.append(f"2024 : {pct_ss_24:.1f}% sous θ={seuil_sim:.2f}")
        if not np.isnan(pct_ss_25):
            subtitle.append(f"2025 : {pct_ss_25:.1f}% sous θ={seuil_sim:.2f}")
        kpi_cards.append(
            dbc.Col(
                _kpi_card(
                    "Similarité moyenne",
                    val_txt,
                    subtitle=" · ".join(subtitle),
                    color="#7c3aed",
                ),
                md=3,
            )
        )

    # ── 2. Adéquation SND30 des objectifs ─────────────────────────────────
    fig_obj_piliers = go.Figure()
    fig_obj_scores = go.Figure()
    if isinstance(obj24, pd.DataFrame) and isinstance(obj25, pd.DataFrame):
        def _prepare_obj(df: pd.DataFrame, year_label: str) -> pd.DataFrame:
            if df.empty:
                return pd.DataFrame()
            pilier_col = None
            for c in ["pilier_court", "pilier_dominant"]:
                if c in df.columns:
                    pilier_col = c
                    break
            if pilier_col is None:
                return pd.DataFrame()
            out = df.copy()
            out["Pilier"] = out[pilier_col].astype(str).apply(_map_pilier)
            out["AE"] = _safe_numeric(out.get("ae", np.nan)) / 1e9
            out["annee"] = year_label
            return out[["Pilier", "AE", "annee", "score_pilier_dominant"]].copy()

        o24 = _prepare_obj(obj24, "2024")
        o25 = _prepare_obj(obj25, "2025")
        all_o = pd.concat([o24, o25], ignore_index=True) if not o24.empty or not o25.empty else pd.DataFrame()

        if not all_o.empty:
            agg = (
                all_o.groupby(["Pilier", "annee"], as_index=False)
                .agg(n_obj=("AE", "size"), AE_total=("AE", "sum"))
            )
            fig_obj_piliers = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Nombre d'objectifs par pilier", "AE classifiées par pilier"),
                specs=[[{"type": "bar"}, {"type": "bar"}]],
            )
            for year_label, color in YEAR_COLORS.items():
                sub = agg[agg["annee"] == year_label]
                fig_obj_piliers.add_trace(
                    go.Bar(
                        x=sub["Pilier"],
                        y=sub["n_obj"],
                        name=f"# objectifs {year_label}",
                        marker_color=color,
                    ),
                    row=1,
                    col=1,
                )
                fig_obj_piliers.add_trace(
                    go.Bar(
                        x=sub["Pilier"],
                        y=sub["AE_total"],
                        name=f"AE {year_label}",
                        marker_color=color,
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )
            fig_obj_piliers.update_yaxes(title_text="Nb objectifs", row=1, col=1)
            fig_obj_piliers.update_yaxes(title_text="AE (Mds FCFA)", row=1, col=2)
            fig_obj_piliers.update_layout(
                template="plotly_white",
                barmode="group",
                height=520,
            )

        # Distribution des scores de pilier dominant
        scores_parts = []
        for df, year_label in [(obj24, "2024"), (obj25, "2025")]:
            if "score_pilier_dominant" in df.columns:
                tmp = df[["score_pilier_dominant"]].copy()
                tmp["annee"] = year_label
                scores_parts.append(tmp)
        if scores_parts:
            scores = pd.concat(scores_parts, ignore_index=True).dropna(subset=["score_pilier_dominant"])
            fig_obj_scores = px.violin(
                scores,
                x="annee",
                y="score_pilier_dominant",
                color="annee",
                color_discrete_map=YEAR_COLORS,
                box=True,
                points=False,
                title="Distribution des scores de confiance (pilier dominant)",
            )
            fig_obj_scores.update_layout(template="plotly_white", height=420)

    # ── 3. Glissement sémantique & concentration ──────────────────────────
    fig_sim = go.Figure()
    fig_lorenz = go.Figure()
    fig_tests_summary = go.Figure()
    if isinstance(sim_df, pd.DataFrame) and not sim_df.empty:
        score_col = None
        if "score_max_similarite" in sim_df.columns:
            score_col = "score_max_similarite"
        elif "similarity" in sim_df.columns:
            score_col = "similarity"
        if score_col:
            sims = sim_df.dropna(subset=[score_col]).copy()
            if "annee" not in sims.columns and "year" in sims.columns:
                sims["annee"] = sims["year"]
            if "annee" in sims.columns:
                fig_sim = px.histogram(
                    sims,
                    x=score_col,
                    color="annee",
                    nbins=30,
                    barmode="overlay",
                    color_discrete_map=YEAR_COLORS,
                    labels={score_col: "Similarité cosinus"},
                    title="Distribution des similarités cosinus par année",
                )
                fig_sim.update_traces(opacity=0.65)
                fig_sim.update_layout(template="plotly_white", height=420)

    # Courbe de Lorenz AE (2024 vs 2025)
    if not budget24.empty and not budget25.empty and "ae" in budget24.columns and "ae" in budget25.columns:
        for year_label, df, color in [("2024", budget24, YEAR_COLORS["2024"]), ("2025", budget25, YEAR_COLORS["2025"])]:
            x_l, y_l = _lorenz_curve(df["ae"].values)
            g = _gini(df["ae"].values)
            fig_lorenz.add_trace(
                go.Scatter(x=x_l, y=y_l, mode="lines", name=f"{year_label} (Gini={g:.3f})", line=dict(color=color, width=2.5))
            )
        fig_lorenz.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Égalité parfaite", line=dict(color="gray", dash="dash"))
        )
        fig_lorenz.update_layout(
            template="plotly_white",
            title="Courbes de Lorenz — AE",
            xaxis_title="Fraction cumulée des lignes",
            yaxis_title="Fraction cumulée des AE",
            height=420,
        )

    # Synthèse des tests (répartition des p-values)
    tests_p: list[float] = []
    if isinstance(chi2_df, pd.DataFrame) and not chi2_df.empty and "p_value" in chi2_df.columns:
        tests_p.extend([float(p) for p in chi2_df["p_value"].dropna().tolist()])
    if isinstance(mw_df, pd.DataFrame) and not mw_df.empty and "p_value" in mw_df.columns:
        tests_p.extend([float(p) for p in mw_df["p_value"].dropna().tolist()])

    if tests_p:
        sig_001 = sum(1 for p in tests_p if p < 0.001)
        sig_01 = sum(1 for p in tests_p if 0.001 <= p < 0.01)
        sig_05 = sum(1 for p in tests_p if 0.01 <= p < 0.05)
        non_sig = sum(1 for p in tests_p if p >= 0.05)
        labels = [
            f"p<0.001 (***) n={sig_001}",
            f"p<0.01 (**) n={sig_01}",
            f"p<0.05 (*) n={sig_05}",
            f"n.s. n={non_sig}",
        ]
        sizes = [sig_001, sig_01, sig_05, non_sig]
        if sum(sizes) > 0:
            fig_tests_summary = go.Figure(
                data=[
                    go.Pie(
                        labels=[l for l, s in zip(labels, sizes) if s > 0],
                        values=[s for s in sizes if s > 0],
                        hole=0.4,
                    )
                ]
            )
            fig_tests_summary.update_layout(
                title=f"Bilan des tests (n={len(tests_p)})",
                template="plotly_white",
                height=380,
            )

    return html.Div([
        _section_header(
            "Adéquation budgétaire et sémantique",
            "Synthèse 2024–2025 : budget AE/CP, alignement SND30 des objectifs et stabilité du langage budgétaire.",
        ),
        (dbc.Row(kpi_cards, className="gy-3 mb-4") if kpi_cards else html.Div()),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_macro), md=12),
        ], className="mb-4"),
        html.Hr(className="my-4"),
        _section_header("Objectifs budgétaires et piliers SND30"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_obj_piliers), md=8),
            dbc.Col(dcc.Graph(figure=fig_obj_scores), md=4),
        ], className="mb-4"),
        html.Hr(className="my-4"),
        _section_header("Glissement sémantique et concentration budgétaire"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_sim), md=6),
            dbc.Col(dcc.Graph(figure=fig_lorenz), md=6),
        ]),
        html.Hr(className="my-4"),
        _section_header("Bilan des tests de significativité"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(figure=fig_tests_summary)
                if fig_tests_summary.data
                else dbc.Alert("Résultats de tests indisponibles pour la synthèse.", color="secondary"),
                md=6,
            ),
        ]),
    ])


# ═════════════════════════════════════════════════════════════════════════════=
# PAGE & CALLBACKS PUBLICS
# ═════════════════════════════════════════════════════════════════════════════=

def create_stats_page(DATA: dict) -> html.Div:
    """Construit la page complète "Analyses Statistiques Avancées"."""
    # On ne fait que définir la structure globale (onglets + conteneur).
    tabs = dbc.Tabs(
        [
            dbc.Tab(label="🎯 Vue d'ensemble",       tab_id="stats-overview"),
            dbc.Tab(label="📡 Glissement sémantique", tab_id="stats-semantic"),
            dbc.Tab(label="🧪 Tests statistiques",    tab_id="stats-tests"),
            dbc.Tab(label="🌊 Évolution thématique",  tab_id="stats-thematic"),
            dbc.Tab(label="🔵 Clustering",            tab_id="stats-clustering"),
            dbc.Tab(label="⚖️ Adéquation",            tab_id="stats-adequation"),
        ], id="stats-main-tabs", active_tab="stats-overview", class_name="page-tabs"
    )
    content = html.Div(id="stats-main-content", className="mt-4")
    return html.Div([
        html.H1("Analyses Statistiques Avancées", className="mb-1",
                 style={"fontWeight": 800, "color": "#1e3a5f"}),
        html.P(
            "Glissement sémantique · Tests de significativité · Évolution thématique · Clustering comparatif",
            className="text-muted mb-4",
            style={"fontSize": "0.95rem", "fontStyle": "italic"}
        ),
        tabs,
        content,
    ], className="page-container page-stats")


def register_stats_callbacks(app, DATA: dict) -> None:
    """Enregistre les callbacks nécessaires pour l'onglet stats avancé."""
    d_cache = _load_stats_data(DATA)

    @app.callback(
        Output("stats-main-content", "children"),
        Input("stats-main-tabs", "active_tab"),
    )
    def _render_stats(tab: str):
        if tab == "stats-overview":
            return _tab_overview(d_cache)
        if tab == "stats-semantic":
            return _tab_semantic(d_cache)
        if tab == "stats-tests":
            return _tab_tests(d_cache)
        if tab == "stats-thematic":
            return _tab_thematic(d_cache)
        if tab == "stats-clustering":
            return _tab_clustering(d_cache)
        if tab == "stats-adequation":
            return _tab_adequation(d_cache)
        # fallback
        return _tab_overview(d_cache)


def create_synthesis_page(DATA: dict) -> html.Div:
    """Page de synthèse regroupant les principaux graphiques du notebook.

    Cette vue assemble en une seule page les éléments clés déjà utilisés
    dans les onglets avancés :
      - Vue d'ensemble budgétaire et piliers
      - Glissement sémantique (similarités embeddings)
      - Évolution thématique (topics dominants)
      - Clustering (structure des groupes d'articles)

    Elle est pensée comme une page de présentation synthétique pour
    décideurs, sans contrôles interactifs supplémentaires.
    """
    d_cache = _load_stats_data(DATA)

    overview_section = _tab_overview(d_cache)
    semantic_section = _tab_semantic(d_cache)
    thematic_section = _tab_thematic(d_cache)
    clustering_section = _tab_clustering(d_cache)

    return html.Div(
        [
            html.H1(
                "Synthèse Analytique 2024–2025",
                className="mb-1",
                style={"fontWeight": 800, "color": "#1e3a5f"},
            ),
            html.P(
                "Résumé visuel des principaux résultats de l'audit sémantique "
                "et budgétaire (budget, similarité, thèmes, clusters).",
                className="text-muted mb-4",
                style={"fontSize": "0.95rem", "fontStyle": "italic"},
            ),
            overview_section,
            html.Hr(className="my-5"),
            semantic_section,
            html.Hr(className="my-5"),
            thematic_section,
            html.Hr(className="my-5"),
            clustering_section,
        ],
        className="page-container page-stats",
    )
