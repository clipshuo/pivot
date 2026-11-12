# Cas d'échec et comment ils sont signalés

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
