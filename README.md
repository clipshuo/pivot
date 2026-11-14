# Pivot

Assemble un dossier d'images prises en tournant sur soi-même en un panorama sphérique
**équirectangulaire 360×180° (ratio 2:1)**, prêt à être projeté sur une sphère
(viewers 360°, Street View, three.js, etc.).

Les captures de l'app Android Pivot Capture sont assemblées avec les orientations du gyroscope, ce qui
place aussi les images sans texture (mur blanc, ciel).

Pivot ne dépend d'aucun logiciel externe (ni Hugin ni PTGui) : uniquement Python, OpenCV, NumPy et SciPy.

## Installation

```bash
git clone https://github.com/clipshuo/pivot.git && cd pivot
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

L'installation crée la commande `pivot`. N'installez qu'une seule variante d'OpenCV
(`opencv-contrib-python`) : elle contient SIFT et `cv2.detail`.

## Utilisation

```bash
# Vidéo / séquence ordonnée (une rangée, images triées par nom)
pivot --input ./dataset --output pano.jpg --width 4096 \
      --detector sift --blend multiband --fill inpaint --debug

# Sphère complète sur plusieurs rangées, ou ordre inconnu : compare toutes les paires
pivot --input ./dataset --output pano.jpg --unordered

# Référence de comparaison (cv2.Stitcher)
pivot --input ./dataset --output ref.jpg --baseline
```

| Option | Défaut | Rôle |
|---|---|---|
| `--input` | (obligatoire) | dossier d'images jpg/png, triées par nom |
| `--output` | `pano.jpg` | image de sortie ; les métadonnées GPano ne sont écrites qu'en JPEG |
| `--width` | `4096` | largeur de sortie (hauteur = largeur / 2) ; `8192` pour la haute définition |
| `--detector` | `sift` | `sift` ou `orb` (plus rapide, moins précis) |
| `--ratio` | `0.75` | seuil du test de ratio de Lowe |
| `--unordered` | non | ordre inconnu ou plusieurs rangées : toutes les paires sont comparées |
| `--neighbor-range` | `2` | mode ordonné : chaque image est comparée à ses N suivantes (boucle fermée) |
| `--focal` | auto | focale imposée, en pixels pleine résolution |
| `--work-mp` | `1.0` | taille de la copie de travail, en mégapixels |
| `--no-sensors` | non | ignore `capture.json` (orientations du téléphone) même s'il est dans le dossier |
| `--no-distortion` | non | n'estime pas la distorsion radiale de l'objectif |
| `--no-wave` | non | désactive la correction d'horizon |
| `--no-expo` | non | désactive la compensation d'exposition |
| `--seam` | `voronoi` | `voronoi` ou `graphcut` (évite mieux les objets en mouvement, plus lent) |
| `--blend` | `multiband` | `multiband` ou `linear` |
| `--bands` | `5` | niveaux de la fusion multi-bandes (1 à 6) |
| `--fill` | `inpaint` | zones sans image : `none`, `inpaint`, `color`, `gradient` |
| `--fill-color B G R` | `0 0 0` | couleur pour `--fill color` |
| `--quality` | `95` | qualité JPEG |
| `--workers` | `0` | threads ; 0 = tous les cœurs disponibles |
| `--baseline` | non | assemble avec `cv2.Stitcher` au lieu du pipeline de Pivot |
| `--debug`, `--debug-dir` | non, `debug` | écrit les visualisations et les métriques |

`python -m pivot ...` est équivalent à `pivot ...`.

Code de sortie : `0` si un panorama a été écrit (éventuellement avec des avertissements), `2` en cas d'erreur bloquante.

### Captures de l'app Pivot Capture

Si le dossier d'entrée contient un `capture.json` (écrit par l'app Android Pivot Capture), Pivot utilise
les orientations mesurées par le téléphone : seules les paires qui se recouvrent sont comparées,
toutes les images sont placées (même un mur blanc ou un plafond), la focale du téléphone est gardée et
l'horizon vient de la verticale mesurée. `--unordered` et `--focal` deviennent inutiles :

```bash
pivot --input ./2026-11-12_050344 --output pano.jpg --seam graphcut
```

Le rapport gagne une section `capteurs` (paires rejetées, images placées par le gyroscope seul, écart
entre l'orientation finale et la mesure).

### Depuis du code Python (future API)

```python
from pivot import PivotConfig, run_pipeline

result = run_pipeline(PivotConfig(input="./dataset", output="pano.jpg", width=4096))
result.panorama          # image BGR (H, W, 3)
result.coverage          # masque des pixels réellement couverts
result.report.warnings   # avertissements, en clair
result.report.metrics    # focale, erreurs de reprojection, couverture...
```

`run_pipeline` est le seul point d'entrée métier : la ligne de commande et l'API ne font que construire
un `PivotConfig`.

### API HTTP

```bash
pip install -e ".[api]"
uvicorn pivot.api:app --host 127.0.0.1 --port 8010   # --root-path /pivot-api derrière un relais
```

| Route | Rôle |
|---|---|
| `GET /` | page d'envoi des photos, pour tester depuis un navigateur |
| `POST /panoramas` | photos (+ `capture.json`), options `width`, `seam`, `fill`, `unordered` ; renvoie un `id` (202) |
| `GET /panoramas` | liste des tâches |
| `GET /panoramas/{id}` | état (`en_attente`, `en_cours`, `termine`, `echec`), avertissements, durées, métriques |
| `GET /panoramas/{id}/pano.jpg` | le panorama |
| `GET /panoramas/{id}/log.txt` | journal de l'assemblage |

L'assemblage tourne dans un thread, une tâche à la fois. Les tâches sont dans `PIVOT_DATA`
(défaut `data/api/`) et survivent à un redémarrage ; une tâche interrompue passe en échec.
Pas d'authentification.

### Sorties de `--debug`

```
debug/
├── keypoints/<image>.jpg          points d'intérêt détectés
├── matches/<i>_<j>_avant.jpg      appariements des paires voisines avant RANSAC
├── matches/<i>_<j>_apres.jpg      ... et après RANSAC + modèle de rotation
├── graph.png                      graphe de connexion (épaisseur = nombre d'inliers)
├── masks/coverage.png             pixels couverts par au moins une image
├── masks/overlap_count.png        nombre d'images par pixel
├── masks/seams.png                coutures : une couleur par image source
├── log.txt                        journal complet
└── metrics.json                   inliers par paire, focale, erreur de reprojection
                                   avant/après, écart de boucle, couverture, temps par étape
```

## Tests

```bash
pytest
```

35 tests, environ 5 secondes. Ils couvrent la projection (aller-retour d'un point 3D, raccord à ±180°),
le jacobien analytique de l'ajustement de faisceaux (comparé à des différences finies), la fermeture
de boucle sur des données synthétiques, les coutures, la fusion, le remplissage, les métadonnées GPano,
et le pipeline complet sur un tour synthétique dont la focale et les rotations sont connues.

## Organisation du dépôt

```
pivot/
├── pyproject.toml          paquet, dépendances, commande `pivot`, configuration de pytest
├── requirements.txt        raccourci : pip install -r requirements.txt
├── README.md
├── docs/
│   ├── pipeline.md         principe des 10 étapes
│   ├── cas-d-echec.md      cas d'échec et signalement
│   ├── resultats.md        résultats de référence, limites connues
│   └── feuille-de-route.md GPU, service FastAPI, parallaxe
├── src/pivot/
│   ├── cli.py              ligne de commande
│   ├── api.py              service HTTP (FastAPI) : envoi des photos, file de tâches
│   ├── config.py           paramètres par défaut, détection de la machine
│   ├── models.py           structures de données, erreurs, conventions géométriques
│   ├── pipeline.py         orchestrateur run_pipeline (étapes 1 à 10)
│   ├── io_utils.py         étapes 1 et 10 : chargement, EXIF, JPEG + XMP GPano
│   ├── features.py         étapes 2 et 3 : SIFT / ORB, appariement, RANSAC
│   ├── geometry.py         étapes 4 à 6 : focale, rotations, ajustement de faisceaux, horizon
│   ├── projection.py       étape 7 : mapping équirectangulaire
│   ├── blending.py         étape 8 : gains, coutures, fusion
│   ├── fill.py             étape 9 : zénith / nadir
│   ├── sensors.py          orientations du téléphone (capture.json), a priori de l'ajustement
│   ├── baseline.py         mode de référence cv2.Stitcher
│   └── debug.py            visualisations, métriques, chronométrage
└── tests/                  35 tests (pytest)
```

Conventions (détaillées dans `src/pivot/models.py`) : repère caméra OpenCV (x à droite, y vers le bas,
z vers l'avant) ; `R` va du monde vers la caméra ; un pixel entier désigne le centre du pixel.

## Documentation

- [Principe de chaque étape](docs/pipeline.md)
- [Cas d'échec et comment ils sont signalés](docs/cas-d-echec.md)
- [Résultats de référence et limites connues](docs/resultats.md)
- [Pistes pour la suite](docs/feuille-de-route.md) : GPU, parallélisation, service FastAPI, parallaxe
