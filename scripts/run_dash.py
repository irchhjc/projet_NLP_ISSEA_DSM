"""
Script de lancement du Dashboard Dash.
"""
import sys
from pathlib import Path

# Ajouter le chemin du dashboard au PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.app_dash import app

def main():
    app.run(
        debug=True,
        host='127.0.0.1',
        port=8050
    )

if __name__ == '__main__':
    main()
