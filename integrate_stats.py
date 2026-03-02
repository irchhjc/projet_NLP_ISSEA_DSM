"""
INTÉGRATION DU NOUVEL ONGLET STATISTIQUES DANS app_dash.py
═══════════════════════════════════════════════════════════

Ce fichier documente les 4 modifications à apporter à
  src/audit_semantique/dashboard/app_dash.py

─────────────────────────────────────────────────────────────
MODIFICATION 1 — Import (ajouter après les imports existants)
─────────────────────────────────────────────────────────────

    # >>> AJOUTER (après les imports audit_semantique existants)
    try:
        from audit_semantique.dashboard.stats_page import (
            create_stats_page as _create_stats_page_advanced,
            register_stats_callbacks,
        )
        _ADVANCED_STATS = True
    except ImportError:
        _ADVANCED_STATS = False

─────────────────────────────────────────────────────────────
MODIFICATION 2 — Dans display_page() callback
─────────────────────────────────────────────────────────────

    Remplacer le bloc :
        elif pathname == '/stats':
            return create_stats_page()

    Par :
        elif pathname == '/stats':
            if _ADVANCED_STATS:
                return _create_stats_page_advanced(DATA)
            return create_stats_page()     # fallback ancienne version

─────────────────────────────────────────────────────────────
MODIFICATION 3 — Enregistrer les callbacks (fin de fichier)
─────────────────────────────────────────────────────────────

    Ajouter APRÈS toutes les définitions de callbacks (avant if __name__):

    if _ADVANCED_STATS:
        register_stats_callbacks(app, DATA)

─────────────────────────────────────────────────────────────
MODIFICATION 4 — run_dash.py
─────────────────────────────────────────────────────────────

    Aucune modification requise.
    Les callbacks sont enregistrés lors du chargement du module.

═══════════════════════════════════════════════════════════
ALTERNATIVE — Remplacement direct dans app_dash.py
═══════════════════════════════════════════════════════════

Si vous préférez remplacer directement la fonction existante,
copiez le contenu ci-dessous à la place des fonctions :
  - create_stats_page()
  - create_stats_descriptives()
  - create_stats_tests()
  - create_stats_distributions()
et du callback render_stats_content().
"""

# ════════════════════════════════════════════════════════════════════════════
# REPLACEMENT COMPLET — coller dans app_dash.py à la place des anciennes fonctions
# ════════════════════════════════════════════════════════════════════════════

REPLACEMENT_CODE = '''
# ─── Import du module stats avancé ──────────────────────────────────────────
# Ce bloc remplace create_stats_page / create_stats_descriptives / create_stats_tests
# / create_stats_distributions et le callback render_stats_content.

try:
    from audit_semantique.dashboard.stats_page import (
        create_stats_page as _stats_page_fn,
        register_stats_callbacks as _register_stats_cb,
    )
    _STATS_ADVANCED = True
except ImportError as _e:
    print(f"[WARN] stats_page non disponible : {_e}. Fallback sur version simple.")
    _STATS_ADVANCED = False


def create_stats_page():
    """Page Tests Statistiques (délègue au module stats_page avancé si dispo)."""
    if _STATS_ADVANCED:
        return _stats_page_fn(DATA)
    # Fallback minimaliste
    if "budget_2024" not in DATA or "budget_2025" not in DATA:
        return html.Div([dbc.Alert("Données non disponibles", color="warning")])
    return html.Div([
        html.H1("Tests Statistiques", className="mb-4"),
        dbc.Alert("Module stats_page non trouvé. Placez stats_page.py dans "
                  "src/audit_semantique/dashboard/", color="warning"),
    ])


# Enregistrer les callbacks avancés
if _STATS_ADVANCED:
    _register_stats_cb(app, DATA)

# NOTE : Supprimer / commenter les anciens callbacks et fonctions :
#   - render_stats_content()
#   - create_stats_descriptives()
#   - create_stats_tests()
#   - create_stats_distributions()
'''

if __name__ == "__main__":
    print(REPLACEMENT_CODE)
