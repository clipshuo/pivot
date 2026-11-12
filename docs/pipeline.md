# Principe de chaque étape

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
