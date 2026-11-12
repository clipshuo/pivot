# Résultats et limites

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
