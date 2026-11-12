# Pivot

Assemble un dossier d'images prises en tournant sur soi-même en un panorama sphérique
**équirectangulaire 360×180° (ratio 2:1)**, prêt à être projeté sur une sphère
(viewers 360°, Street View, three.js, etc.).

Pivot ne dépend d'aucun logiciel externe (ni Hugin ni PTGui) : uniquement Python, OpenCV, NumPy et SciPy.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

N'installez qu'une seule variante d'OpenCV (`opencv-contrib-python`) : elle contient SIFT et `cv2.detail`.

## Utilisation

```bash
# Vidéo / séquence ordonnée (une rangée, images triées par nom)
python pivot.py --input ./dataset --output pano.jpg --width 4096 \
                --detector sift --blend multiband --fill inpaint --debug

# Sphère complète sur plusieurs rangées, ou ordre inconnu : compare toutes les paires
python pivot.py --input ./dataset --output pano.jpg --unordered

# Référence de comparaison (cv2.Stitcher)
python pivot.py --input ./dataset --output ref.jpg --baseline
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

Code de sortie : `0` si un panorama a été écrit (éventuellement avec des avertissements), `2` en cas d'erreur bloquante.

### Depuis du code Python (future API)

```python
from config import PivotConfig
from pipeline import run_pipeline

result = run_pipeline(PivotConfig(input="./dataset", output="pano.jpg", width=4096))
result.panorama          # image BGR (H, W, 3)
result.coverage          # masque des pixels réellement couverts
result.report.warnings   # avertissements, en clair
result.report.metrics    # focale, erreurs de reprojection, couverture...
```

`run_pipeline` est le seul point d'entrée métier : la ligne de commande ne fait que construire
un `PivotConfig`. Une route d'API pourra l'appeler telle quelle.

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
python -m pytest tests -q
```

31 tests, environ 5 secondes. Ils couvrent la projection (aller-retour d'un point 3D, raccord à ±180°),
le jacobien analytique de l'ajustement de faisceaux (comparé à des différences finies), la fermeture
de boucle sur des données synthétiques, les coutures, la fusion, le remplissage, les métadonnées GPano,
et le pipeline complet sur un tour synthétique dont la focale et les rotations sont connues.

## Organisation du code

| Fichier | Étapes | Contenu |
|---|---|---|
| `pivot.py` | | ligne de commande |
| `config.py` | | paramètres par défaut, détection de la machine (cœurs, RAM, CUDA, OpenCL) |
| `models.py` | | structures de données, erreurs, conventions géométriques |
| `pipeline.py` | 1 à 10 | orchestrateur `run_pipeline` |
| `io_utils.py` | 1, 10 | chargement, EXIF, netteté, JPEG + XMP GPano |
| `features.py` | 2, 3 | SIFT / ORB, appariement, RANSAC par paire |
| `geometry.py` | 4, 5, 6 | focale, rotations, graphe, ajustement de faisceaux, horizon |
| `projection.py` | 7 | mapping équirectangulaire (NumPy + `cv2.remap`) |
| `blending.py` | 8 | gains, coutures, fusion multi-bandes ou linéaire |
| `fill.py` | 9 | zénith / nadir |
| `baseline.py` | | mode de référence `cv2.Stitcher` |
| `debug.py` | | visualisations, métriques, chronométrage |

Conventions (détaillées dans `models.py`) : repère caméra OpenCV (x à droite, y vers le bas, z vers l'avant) ;
`R` va du monde vers la caméra ; un pixel entier désigne le centre du pixel.

## Principe de chaque étape

**Modèle géométrique.** La caméra tourne autour de son centre optique, sans translation. Deux images
sont alors liées par `H = K · R_j · R_iᵀ · K⁻¹`, avec `K` la matrice intrinsèque (une focale partagée,
point principal au centre) et `R` des rotations 3D. L'homographie libre à 8 degrés de liberté ne sert
qu'à trier les inliers et à initialiser la focale : le modèle final n'utilise que des rotations.

1. **Chargement.** Les images sont lues en parallèle. Une copie d'environ 1 mégapixel sert à la détection
   et à l'estimation ; la pleine résolution est gardée pour le rendu. La focale EXIF est lue si elle est
   exploitable (équivalent 35 mm ou densité du capteur). La netteté de chaque image (variance du laplacien)
   est mesurée pour signaler les images floues.
2. **Points d'intérêt.** SIFT par défaut (4000 points par image), ORB en option, une image par thread.
3. **Appariement.** FLANN (SIFT) ou force brute de Hamming (ORB), puis test de ratio de Lowe. En mode
   ordonné, chaque image est comparée à ses voisines `i+1 … i+N` modulo n, ce qui ferme la boucle
   (dernière ↔ première). En mode `--unordered`, toutes les paires sont comparées.
4. **Géométrie par paire.** RANSAC sur une homographie élimine les faux appariements. Une paire est
   retenue si elle a assez d'inliers et si le critère de Brown & Lowe `inliers / (8 + 0,3 · appariements)`
   dépasse 1. La focale initiale vient de l'EXIF, sinon de la médiane des focales déduites des homographies.
   Chaque homographie est ensuite remplacée par une vraie rotation (algorithme de Kabsch sur les rayons) ;
   les inliers incompatibles avec une rotation pure sont rejetés. Les paires valides forment le graphe de
   connexion ; on garde sa plus grande composante connexe, et les rotations initiales sont obtenues en
   chaînant les rotations relatives le long d'un arbre couvrant maximal.
5. **Ajustement de faisceaux.** Optimise ensemble toutes les rotations, la focale et un coefficient de
   distorsion radiale, en minimisant l'erreur de reprojection symétrique de tous les inliers
   (Levenberg-Marquardt, jacobien analytique creux, perte de Huber). C'est cette étape qui répartit la
   dérive accumulée sur le tour et ferme la boucle à 360°. La distorsion n'est estimée que si la boucle
   est fermée : sinon focale et distorsion se compensent et la focale dérive.
6. **Correction d'horizon.** On tient rarement la caméra en roulis : les axes x de toutes les caméras
   restent dans un plan, dont la normale est la verticale. Le repère monde est réorienté pour que cette
   verticale soit l'axe des pôles, et l'image de référence se retrouve au centre du panorama.
7. **Projection équirectangulaire.** Pour chaque pixel `(u, v)` de la sortie :
   `θ = ((u + 0,5) / W − 0,5) · 2π`, `φ = (0,5 − (v + 0,5) / H) · π`,
   rayon `d = (cos φ · sin θ, −sin φ, cos φ · cos θ)`, projeté dans chaque caméra par `K · R · d`, puis la
   couleur est lue avec `cv2.remap`. Chaque image n'est projetée que dans sa zone utile du panorama
   (la zone peut traverser le bord à ±180°). Les rayons situés derrière la caméra sont écartés. Si la
   source est beaucoup plus fine que la sortie, elle est d'abord réduite pour éviter le repliement.
8. **Exposition, coutures, fusion.** Un gain par image est calculé par moindres carrés sur les zones
   communes (Brown & Lowe). Les coutures attribuent chaque pixel à une seule image : la plus centrale
   (`voronoi`) ou par coupe de graphe guidée par la différence de couleur (`graphcut`). La fusion
   multi-bandes (pyramides laplaciennes de Burt & Adelson) mélange les basses fréquences sur une large zone
   et les hautes fréquences sur une zone étroite ; elle est circulaire en longitude pour qu'aucune ligne
   n'apparaisse à ±180°. Le fondu linéaire est l'option simple.
9. **Zones non couvertes.** Les pixels sans image source (zénith, nadir, tour incomplet) sont laissés
   noirs, remplis par inpainting, par une couleur, ou par un dégradé vers les pôles. L'image reste
   toujours un 2:1 valide.
10. **Sauvegarde.** JPEG haute qualité avec un segment XMP GPano (`ProjectionType=equirectangular`,
    `FullPanoWidthPixels`, etc.) : les viewers standards reconnaissent l'image comme un 360°.

**Mode `--baseline`.** `cv2.Stitcher` en mode PANORAMA (pipeline `cv2.detail`, projection sphérique).
Sa sortie est recadrée sur la zone couverte : ce n'est pas un équirectangulaire 2:1, elle sert seulement
à vérifier qu'un jeu d'images est assemblable et à comparer visuellement.

## Cas d'échec et comment ils sont signalés

Les avertissements sont affichés, stockés dans `result.report.warnings` et écrits dans `metrics.json`.
Les erreurs bloquantes lèvent une sous-classe de `PivotError` (code de sortie 2).

| Situation | Signal | Que faire |
|---|---|---|
| Moins de 2 images, dossier absent, tailles d'images différentes | erreur `NotEnoughImagesError` / `PivotError` | vérifier le dossier ; une seule caméra, sans zoom |
| Images floues | avertissement « Images probablement floues » (netteté < 35 % de la médiane) et « Moins de 100 points d'intérêt » | tourner plus lentement, plus de lumière |
| Recouvrement insuffisant entre voisines | avertissement « Paires voisines sans recouvrement fiable » ; images isolées « écartées » | plus d'images par tour (30 à 50 % de recouvrement) |
| Aucune paire fiable | erreur `InsufficientOverlapError` | scène sans texture, ou images sans rapport |
| Boucle non fermée | avertissement « Boucle non fermée » ; `boucle_fermee: false` dans les métriques | finir le tour en recouvrant largement la première image |
| Tour incomplet | avertissement « Tour incomplet : X° couverts » | idem ; le panorama est quand même produit, partiel |
| Parallaxe trop forte (la caméra se déplace) | avertissement « Erreur de reprojection élevée » au-delà de 2 px ; écart de boucle élevé | tourner autour de l'objectif, pas autour du corps ; éloigner les objets proches ; essayer `--seam graphcut` |
| Focale non estimable | avertissement « Focale non estimable », valeur par défaut | fournir `--focal` |
| Optimisation divergente | erreur `BundleAdjustmentError` | faux appariements ou scène incompatible avec une rotation pure |

Lecture rapide de `metrics.json` : une erreur de reprojection après ajustement inférieure à 1 px est
très bonne ; entre 1 et 2 px, correcte ; au-delà, la parallaxe ou la distorsion se voit (fantômes sur les
objets proches, fils électriques dédoublés).

## Résultats de référence

Machine : 12 cœurs, 83 Go de RAM, sans GPU. Sortie 4096×2048.

| Jeu d'images | Mode | Images retenues | Erreur de reprojection | Couverture | Temps |
|---|---|---|---|---|---|
| Faculty_of_arts_and_design_I (56 images 1280×720) | `--unordered` | 54 | 10,4 → 3,4 px | 360°, 94 % de la sphère | 31 s |
| Faculty_of_arts_and_design_II (56 images) | `--unordered` | 54 | 7,6 → 2,9 px | 360°, 94 % | 33 s |
| Faculty_of_Engineering_parking_lot (80 images) | `--unordered` | 69 | 8,9 → 3,1 px | 360°, 93 % | 57 s |
| GrandCanyon1 (35 images 1920×1080, une rangée) | ordonné | 35 | 0,69 → 0,67 px | 134° (tour incomplet) | 11 s |
| Faculty_of_arts_and_design_I | `--baseline` | (cv2.Stitcher) | | recadré, non 2:1 | 72 s |

Les trois jeux « Faculty » ont une erreur résiduelle d'environ 3 px : la caméra n'y tourne pas exactement
autour de son centre optique. Les panoramas sont complets et propres à l'échelle de la scène ; des
dédoublements restent visibles sur les objets fins et proches (câbles). Les images écartées sont des
images de ciel presque sans texture, dont le nombre varie légèrement d'une exécution à l'autre
(l'appariement FLANN est aléatoire).

## Limites connues

- Une seule focale et un seul coefficient de distorsion pour toutes les images : pas de zoom, pas de
  mélange de caméras. Le point principal est supposé au centre de l'image.
- La parallaxe n'est pas corrigée : le modèle est une rotation pure.
- Les images sans texture (ciel uniforme) ne peuvent pas être placées par la vision seule ; il faudra le
  gyroscope (voir plus bas).
- Le mode `--unordered` compare toutes les paires : le temps d'appariement croît avec le carré du nombre
  d'images.
- L'inpainting d'un grand trou (nadir entier) donne un aplat flou : c'est un bouche-trou, pas une
  reconstruction.

## Pistes pour la suite

**Performance.**
- L'appariement domine en mode `--unordered` (11 s sur 30). Présélectionner les paires par un descripteur
  global ou par un premier appariement en basse résolution, comme le fait `cv2.detail`.
- La fusion multi-bandes est la deuxième étape la plus lente (7 s). Les pyramides se prêtent bien au GPU.
- GPU : OpenCV compilé avec CUDA (`cv2.cuda` pour `remap` et les pyramides, SIFT via une bibliothèque
  dédiée), ou portage de la projection et de la fusion sur PyTorch/CuPy. `config.configure_runtime`
  détecte déjà CUDA et OpenCL.
- Parallélisation : le chargement, la détection, l'appariement, la projection et les pyramides utilisent
  déjà un pool de threads (OpenCV relâche le GIL). L'ajustement de faisceaux résout un système dense ;
  pour des centaines d'images, passer à un solveur creux (complément de Schur).
- Sortie 8192×4096 : la mémoire de la fusion croît avec la surface ; traiter par tuiles si nécessaire.

**Passage en service.**
- FastAPI : `POST /panoramas` reçoit les images (ou une vidéo), crée une tâche et renvoie un identifiant ;
  `GET /panoramas/{id}` renvoie l'état, les avertissements et le lien vers le résultat.
- File de tâches (Celery, RQ ou Arq avec Redis) : l'assemblage dure de 10 à 60 s, il ne doit pas tourner
  dans la requête HTTP. Un worker appelle `run_pipeline(PivotConfig(...))` sans modification.
- Stockage objet (S3 ou équivalent) pour les images sources et les panoramas ; `metrics.json` en base
  pour suivre la qualité.
- Extraction des images d'une vidéo avec choix des plus nettes (le score de netteté existe déjà).
- Les avertissements de `StitchReport` sont déjà rédigés pour l'utilisateur final : l'app peut les
  afficher tels quels (« tournez plus lentement », « finissez le tour »).

**Capteurs du téléphone.**
- Le gyroscope donne l'orientation de chaque image : il remplace `initial_rotations`, permet de ne
  comparer que les paires qui se recouvrent vraiment (fin du mode `--unordered` en n²), et place les
  images sans texture que la vision seule doit écarter.
- L'ajustement de faisceaux peut recevoir un terme d'a priori qui retient chaque rotation près de la
  mesure du gyroscope : plus robuste aux faux appariements et à la parallaxe.
- La focale et la distorsion sont connues par modèle de téléphone (API caméra) : `--focal` existe déjà.
- L'accéléromètre donne la verticale : la correction d'horizon devient exacte au lieu d'être déduite.
- Pendant la capture, l'app peut guider l'utilisateur (vitesse de rotation, zones manquantes) à partir des
  mêmes capteurs, ce qui supprime la plupart des cas d'échec en amont.
