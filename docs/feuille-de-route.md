# Pistes pour la suite

**Performance.**
- L'appariement domine en mode `--unordered` (11 s sur 30). Présélectionner les paires par un descripteur
  global ou par un premier appariement en basse résolution, comme le fait `cv2.detail`.
- La fusion multi-bandes est la deuxième étape la plus lente (7 s). Les pyramides se prêtent bien au GPU.
- GPU : OpenCV compilé avec CUDA (`cv2.cuda` pour `remap` et les pyramides, SIFT via une bibliothèque
  dédiée), ou portage de la projection et de la fusion sur PyTorch/CuPy. `pivot.config.configure_runtime`
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

**Capteurs du téléphone.** Fait : Pivot lit le `capture.json` de l'app Pivot Capture (orientations,
focale, verticale ; voir [pipeline.md](pipeline.md)). Reste à faire :
- La parallaxe est maintenant la première cause de défauts : l'app devrait détecter que le téléphone se
  déplace entre deux prises (accéléromètre) et le signaler pendant la capture.
- Côté assemblage, réduire les découpes sur les objets proches : coutures qui évitent les zones en
  désaccord entre images, ou recalage local autour des coutures.
- Le rejet des paires utilise un seuil fixe de 5° ; le relier à la distance estimée des objets.
- Utiliser les orientations en mode vidéo (une orientation par image extraite).
