# Compiler HYDRA en .exe Windows (PyInstaller)

Ce guide explique comment produire un exécutable Windows autonome de
HYDRA avec PyInstaller, et couvre les pièges connus pour la pile de
dépendances de cette application (PySide6, pymc/pytensor/arviz,
opencv-python-headless, scikit-image).

**Point important, qui corrige une hypothèse de départ courante : il
n'est PAS nécessaire d'embarquer g++ / un compilateur C++ avec l'exe.**
Voir la section "Pourquoi pas de g++ ?" plus bas.

## Pré-requis

- Un Windows (la compilation croisée depuis Linux/macOS vers un .exe
  Windows n'est pas possible avec PyInstaller -- il faut compiler
  directement sur (ou dans une VM/CI) Windows).
- Python, **la même version mineure** que celle utilisée pour développer
  l'application (ex. 3.11.x), installée depuis python.org (pas depuis le
  Microsoft Store, qui a son propre lot de restrictions de sandboxing
  parfois gênantes avec PyInstaller).
- Un environnement virtuel dédié à la compilation, propre :

  ```
  python -m venv build_env
  build_env\Scripts\activate
  pip install --upgrade pip
  pip install -r requirements.txt
  pip install pyinstaller pyinstaller-hooks-contrib
  ```

- **Si PySide6 est aussi installé "system-wide" (hors de cet
  environnement virtuel) sur la machine de build, désinstallez-le
  au niveau système avant de compiler.** PyInstaller peut sinon
  détecter/embarquer par erreur les DLL Qt de l'installation système
  plutôt que celles de l'environnement virtuel, ce qui produit un exe
  qui plante au lancement avec des erreurs Qt (voir dépannage plus bas).

## Compilation

Depuis la racine du projet (là où se trouve `main.py` et `hydra.spec`) :

```
pyinstaller hydra.spec --noconfirm
```

Le résultat se trouve dans `dist\HYDRA\` -- c'est tout ce dossier qu'il
faut distribuer (pas seulement `HYDRA.exe`), et `HYDRA.exe` est le point
d'entrée à lancer/raccourcir.

Un premier build complet peut prendre plusieurs minutes (pymc/pytensor/
numba sont volumineux) -- c'est normal.

## Pourquoi pas de g++ ?

L'application utilise PyMC 5 pour l'onglet "Bayesian Calibration". PyMC
compile en interne le modèle statistique vers du code exécutable rapide
avant l'échantillonnage -- historiquement via PyTensor "C backend", qui
appelle effectivement un compilateur C/C++ système (g++ sous
Windows/MinGW, ou cl.exe avec Visual Studio). C'est cette étape qui, à
l'origine, aurait posé problème pour un exe autonome (un utilisateur
final n'a pas de compilateur C++ installé).

Le code a été ajusté pour utiliser **`nutpie`** comme échantillonneur
par défaut (`core/calib_pymc_model.py`, `DEFAULT_NUTS_SAMPLER`) --
c'est d'ailleurs désormais le choix recommandé par PyMC lui-même quand
il est installé (`pip install "pymc[nutpie]"`). `nutpie` compile le
modèle via le backend **Numba** de PyTensor plutôt que le backend C.
Or la dépendance de Numba, `llvmlite`, embarque LLVM de façon statique
directement dans son paquet -- **aucun compilateur système n'est donc
nécessaire sur la machine de l'utilisateur final**, ni pour compiler
`llvmlite` lui-même (il est distribué pré-compilé) ni pour compiler le
modèle PyMC à l'exécution.

En complément, `main.py` fixe désormais explicitement
`PYTENSOR_FLAGS=cxx=` avant tout import -- ceinture et bretelles : même
si un chemin de code PyMC/PyTensor interne (hors du chemin
d'échantillonnage `nutpie`) tentait malgré tout d'utiliser le backend C,
cette variable force PyTensor à ne jamais essayer d'invoquer un
compilateur, plutôt que d'échouer avec une erreur confuse "g++
introuvable" chez l'utilisateur.

`requirements.txt` a été mis à jour avec `numba` et `nutpie` (en plus de
`pymc`/`pytensor`/`arviz` déjà présents) -- ils doivent être installés
dans l'environnement de build comme n'importe quelle autre dépendance
(l'étape `pip install -r requirements.txt` ci-dessus les couvre déjà).

## Dépannage

**Fenêtre blanche / vide au lancement, ou erreur mentionnant
`qwindows.dll` / "Could not find the Qt platform plugin"** : PySide6 n'a
pas correctement embarqué ses plugins Qt. Deux choses à vérifier dans
l'ordre :
1. Confirmez qu'aucune installation PySide6 "système" ne traîne sur la
   machine de build (voir Pré-requis ci-dessus) -- désinstallez-la si
   c'est le cas, puis recompilez.
2. Si le problème persiste, ajoutez `--collect-all PySide6` à la
   commande (ou l'équivalent `collect_all("PySide6")` dans `hydra.spec`,
   dans le bloc `hidden_imports`/`datas`) et recompilez.

**Erreur au lancement du type `PackageNotFoundError` / version
introuvable pour pymc, pytensor, arviz, xarray ou pandas** :
`hydra.spec` utilise déjà `copy_metadata(...)` pour ces paquets
spécifiquement à cause de ce piège (ils font une vérification
`importlib.metadata` au moment de l'import). Si l'erreur apparaît quand
même pour un paquet non listé, ajoutez `copy_metadata("<nom_du_paquet>")`
à la liste correspondante dans `hydra.spec` et recompilez.

**Erreur au lancement du type `[Errno 2] No such file or directory:
'...\\_MEIxxxxxx\\arviz_base\\example_data\\data_local.json'` (ou un
chemin similaire sous un autre sous-paquet `arviz_*`)** : ce piège a été
rencontré, et une première correction (basée sur les *distributions* pip
installées) s'est révélée insuffisante -- `arviz_base`/`arviz_stats`/
`arviz_plots` peuvent être des paquets Python importables séparément
(chacun son propre dossier sous `site-packages\`) sans pour autant être
des *distributions* pip séparées (`pip list` peut n'afficher qu'une
seule ligne "arviz"), ce qui faisait manquer `arviz_base` à un scan basé
sur `pip list`/les distributions installées. **`hydra.spec` a donc été
corrigé pour scanner directement les paquets Python réellement
importables** (`pkgutil.iter_modules()`, indépendant de la structure de
distribution) plutôt que la liste des paquets pip installés -- vérifiez
que vous avez bien la dernière version de ce fichier.

Si vous utilisez la ligne de commande PyInstaller directe plutôt que
`hydra.spec` : inspectez directement le dossier `site-packages` de votre
environnement virtuel de build (pas `pip list`, qui peut être trompeur
pour les raisons ci-dessus) :

```
dir build_env\Lib\site-packages\arviz*
```

et ajoutez `--collect-all <nom>` pour **chaque dossier** `arviz*`
trouvé (pas seulement celui nommé exactement `arviz`) -- typiquement
`arviz`, `arviz_base`, `arviz_stats`, `arviz_plots` selon votre version,
mais fiez-vous à ce que `dir` affiche réellement chez vous plutôt qu'à
cette liste. Pensez aussi à supprimer les dossiers `build\` et `dist\`
d'une compilation précédente avant de relancer, pour être certain de ne
pas retester un exe obsolète.

**Cliquer sur "Run" (Viscoelastic Model Calibration, "Run correction"
dans Dispersion Correction, ou tout autre endroit utilisant le réglage
"Performance" / nombre de cœurs du projet) rouvre une toute nouvelle
fenêtre de l'application au lieu de lancer le calcul** : piège classique
de `multiprocessing` dans un exécutable PyInstaller sous Windows, déjà
rencontré et corrigé. `core/parallel.py` répartit ces calculs sur
plusieurs processus (`ProcessPoolExecutor`, dès que le nombre de cœurs
réglé sur la page "Performance" du projet dépasse 1) -- sous Windows, un
exécutable figé (PyInstaller) qui lance de nouveaux processus de cette
façon doit d'abord appeler `multiprocessing.freeze_support()`, tout au
début de son point d'entrée, sans quoi chaque processus "worker" relance
l'exécutable entier depuis zéro au lieu de se comporter comme un simple
travailleur -- ce qui rouvre l'appli en boucle plutôt que d'exécuter le
calcul. `main.py` appelle maintenant `multiprocessing.freeze_support()`
en tout premier, avant toute autre chose, sous le `if __name__ ==
"__main__":`. Sans objet (no-op) si vous lancez l'appli normalement via
`python main.py` -- ce piège ne concerne que l'exe compilé.

**Erreur liée à opencv (`cv2`) ou scikit-image (`skimage`) au lancement
(fonctionnalité manquante, module non trouvé à l'exécution alors qu'il
s'importe normalement en dev)** : ces deux paquets utilisent des imports
paresseux/conditionnels selon les fonctionnalités utilisées. `hydra.spec`
appelle déjà `collect_submodules` pour les deux. Si un sous-module
précis manque encore, essayez `--collect-all opencv-python-headless`
(ou `scikit-image`) en plus.

**L'antivirus Windows Defender (ou un autre antivirus) bloque/supprime
l'exe généré, ou le signale comme suspect** : c'est un faux positif
courant avec les exécutables PyInstaller volumineux et non signés
(rien à voir avec un vrai problème de sécurité). Ajoutez une exception
locale pour tester, et pour une distribution à plus grande échelle,
envisagez la signature de code (hors du périmètre de ce guide).

**Pour déboguer un crash au lancement dont le message n'apparaît nulle
part** : dans `hydra.spec`, changez temporairement `console=False` en
`console=True` dans le bloc `EXE(...)`, recompilez, et lancez
`HYDRA.exe` depuis une invite de commandes (`cmd.exe`) plutôt qu'en
double-cliquant -- la trace complète de l'erreur Python s'affichera
dans la console au lieu de disparaître avec la fenêtre.

## Étape manuelle requise avant de considérer le build "terminé"

Rien de tout ce travail (le changement de code `nuts_sampler="nutpie"`,
`hydra.spec`, ce guide) n'a pu être testé avec de vrais PyMC/PyTensor/
numba/PyInstaller dans l'environnement où ce travail a été préparé
(pas d'accès réseau pour les installer, et un build PyInstaller doit de
toute façon tourner sur la machine cible -- Windows -- ce qui exclut
toute compilation croisée depuis cet environnement).

**Un test de compilation réel + un test fonctionnel (lancer l'exe généré
et faire tourner un vrai calcul de "Bayesian Calibration" jusqu'au bout,
sur une machine Windows propre si possible sans Python/compilateur
installé, pour confirmer qu'aucun compilateur n'est réellement requis)
est donc une étape obligatoire avant de distribuer cet exe -- pas
seulement recommandée.**
