"""Génère la projection UMAP des embeddings et la sauvegarde dans outputs/models.

À utiliser si le dashboard affiche :
"Projection UMAP non trouvée. Ajoutez outputs/models/umap_coords.npy".

Usage (depuis la racine du projet) :
    poetry run python scripts/generate_umap_coords.py
"""

import sys
from pathlib import Path

import numpy as np

# S'assurer qu'on utilise le code local dans src/ (et donc les bons chemins
# dans audit_semantique.config), pas une éventuelle version installée.
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audit_semantique.config import MODELS_DIR, UMAP_PARAMS  # noqa: E402


def main() -> None:
    emb_2024_path = MODELS_DIR / "embeddings_2024.npy"
    emb_2025_path = MODELS_DIR / "embeddings_2025.npy"

    print(f"MODELS_DIR = {MODELS_DIR}")
    print(f"embeddings_2024.npy existe : {emb_2024_path.exists()}")
    print(f"embeddings_2025.npy existe : {emb_2025_path.exists()}")

    if not emb_2024_path.exists() or not emb_2025_path.exists():
        print(
            "❌ Fichiers d'embeddings manquants. Assurez-vous d'avoir exécuté "
            "scripts/main_pipeline.py au moins une fois."
        )
        return

    try:
        import umap as umap_lib  # type: ignore[import]
    except ImportError:
        print(
            "❌ Le package 'umap-learn' n'est pas installé dans cet environnement.\n"
            "   Installez-le (poetry add umap-learn) puis relancez ce script."
        )
        return

    print("📥 Chargement des embeddings...")
    emb_2024 = np.load(emb_2024_path)
    emb_2025 = np.load(emb_2025_path)

    n24, n25 = len(emb_2024), len(emb_2025)
    print(f"  2024 : {emb_2024.shape}  |  2025 : {emb_2025.shape}")

    combined = np.vstack([emb_2024, emb_2025])

    print("🤖 Calcul de la projection UMAP (peut prendre quelques minutes)...")
    reducer = umap_lib.UMAP(**UMAP_PARAMS)
    coords = reducer.fit_transform(combined)

    if coords.shape[0] != n24 + n25 or coords.shape[1] < 2:
        print(
            f"❌ Projection UMAP incohérente : coords={coords.shape} "
            f"vs n_total={n24 + n25}"
        )
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Fichier agrégé utilisé en priorité par le dashboard
    all_path = MODELS_DIR / "umap_coords.npy"
    np.save(all_path, coords)
    print(f"✅ Coordonnées UMAP sauvegardées dans {all_path}")

    # Fichiers séparés par année (optionnels, pour compatibilité)
    coords_24 = coords[:n24]
    coords_25 = coords[n24:]
    path_24 = MODELS_DIR / "umap_2024.npy"
    path_25 = MODELS_DIR / "umap_2025.npy"
    np.save(path_24, coords_24)
    np.save(path_25, coords_25)
    print(f"✅ Coordonnées UMAP 2024 sauvegardées dans {path_24}")
    print(f"✅ Coordonnées UMAP 2025 sauvegardées dans {path_25}")

    print("\nTerminé. Rafraîchissez l'onglet Baromètre du dashboard.")


if __name__ == "__main__":
    main()
