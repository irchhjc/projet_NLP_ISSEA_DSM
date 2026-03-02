"""Script de lancement du Dashboard Dash.

On force l'utilisation du code local dans ``src/`` pour que
``audit_semantique.config`` pointe bien vers le dossier du projet
(``outputs/reports``) et pas vers le virtualenv (``.audit/Lib``).
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

# Ajouter le répertoire racine du projet (pour pouvoir importer stats_page.py)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ajouter le répertoire src/ pour prioriser le code local du package
if str(SRC_DIR) not in sys.path:
    # Insérer en première position pour qu'il soit prioritaire sur
    # la version installée dans le virtualenv.
    sys.path.insert(0, str(SRC_DIR))

from audit_semantique.dashboard import app_dash


def main() -> None:
    # Charger explicitement les données pour le dashboard dans l'espace
    # global du module app_dash, afin que le package lui-même ne fasse
    # aucune lecture de fichiers au moment de son import.
    app_dash.DATA = app_dash.load_all_data()

    # Enregistrer les callbacks avancés de la page de statistiques
    # uniquement si le module stats_page est disponible.
    if getattr(app_dash, "_ADVANCED_STATS", False):
        try:
            from audit_semantique.dashboard.stats_page import register_stats_callbacks

            register_stats_callbacks(app_dash.app, app_dash.DATA)
        except Exception as e:  # pragma: no cover
            print(f"[run_dash] Impossible d'enregistrer les callbacks stats avancés: {e}")

    app_dash.app.run(
        debug=True,
        host="127.0.0.1",
        port=8050,
    )


if __name__ == "__main__":
    main()
