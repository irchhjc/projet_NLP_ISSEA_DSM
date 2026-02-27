# 📂 Données — Lois de Finances Cameroun

Placez ici vos fichiers JSON extraits des PDF du MINFI.

## Format attendu

Chaque fichier doit être un tableau JSON d'articles :

```json
[
  {
    "id": "Art. 1",
    "titre": "TITRE I - BUDGET GENERAL",
    "chapitre": "Chapitre 1 - Recettes",
    "content": "Le budget général de l'État pour l'exercice 2024..."
  },
  ...
]
```

## Noms des fichiers attendus

| Fichier                    | Contenu                           |
|----------------------------|-----------------------------------|
| `loi_finances_2024.json`   | Loi de Finances 2023-2024 (MINFI) |
| `loi_finances_2025.json`   | Loi de Finances 2024-2025 (MINFI) |

## Extraction depuis les PDF

Utilisez `pdfplumber` ou `pypdf2` pour extraire le texte des PDF officiels.
Un script d'extraction est disponible dans `scripts/extract_pdf.py` (à venir).
