"""Service HTTP de Pivot : reçoit des photos, assemble le panorama équirectangulaire en arrière-plan.

    uvicorn pivot.api:app --host 127.0.0.1 --port 8010

Routes :
    GET  /                        page d'envoi (formulaire) pour tester depuis un navigateur
    POST /panoramas               photos (+ capture.json de Pivot Capture) -> {"id", "etat", ...}
    GET  /panoramas               liste des tâches, la plus récente en premier
    GET  /panoramas/{id}          état, avertissements, métriques, lien vers l'image
    GET  /panoramas/{id}/pano.jpg panorama 2:1 avec métadonnées GPano

L'assemblage dure de 10 à 60 s : il tourne dans un thread, une tâche à la fois, et appelle
run_pipeline(PivotConfig(...)) sans modification. Chaque tâche a son dossier dans PIVOT_DATA
(défaut : <dépôt>/data/api) avec images/, pano.jpg, log.txt et job.json ; la liste survit à un redémarrage.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import traceback
import uuid
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .config import PivotConfig
from .io_utils import EXTENSIONS
from .models import PivotError
from .pipeline import run_pipeline
from .sensors import FILENAME as CAPTURE_JSON

log = logging.getLogger("pivot.api")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("PIVOT_DATA", os.path.join(_REPO, "data", "api"))
MAX_FILES = 300
MAX_BYTES = 40 * 1024 * 1024          # par fichier

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_queue: "queue.Queue[str]" = queue.Queue()

app = FastAPI(title="Pivot", description="Assemblage de panoramas 360° équirectangulaires")


# --- stockage des tâches -------------------------------------------------------------------------

def _job_dir(job_id: str) -> str:
    return os.path.join(DATA_DIR, job_id)


def _save(job: dict) -> None:
    path = os.path.join(_job_dir(job["id"]), "job.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=1, default=str)
    os.replace(path + ".tmp", path)


def _update(job_id: str, **fields) -> None:
    with _lock:
        _jobs[job_id].update(fields)
        _save(_jobs[job_id])


def _load_existing() -> None:
    """Recharge les tâches du disque ; celles qu'un arrêt a interrompues passent en échec."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, name, "job.json")
        try:
            with open(path, encoding="utf-8") as f:
                job = json.load(f)
        except (OSError, ValueError):
            continue
        if job.get("etat") in ("en_attente", "en_cours"):
            job.update(etat="echec", erreur="Service redémarré pendant l'assemblage : renvoyez les photos.")
            _save(job)
        _jobs[job["id"]] = job


# --- exécution ------------------------------------------------------------------------------------

def _run(job_id: str) -> None:
    job = _jobs[job_id]
    folder = _job_dir(job_id)
    opts = job["options"]
    handler = logging.FileHandler(os.path.join(folder, "log.txt"), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    pivot_log = logging.getLogger("pivot")
    pivot_log.addHandler(handler)
    _update(job_id, etat="en_cours", debut=time.time())
    try:
        cfg = PivotConfig(input=os.path.join(folder, "images"), output=os.path.join(folder, "pano.jpg"),
                          width=opts["width"], seam=opts["seam"], fill=opts["fill"],
                          ordered=not opts["unordered"])
        result = run_pipeline(cfg)
        report = result.report.to_dict()
        _update(job_id, etat="termine", fin=time.time(), avertissements=report.get("warnings", []),
                metriques=report.get("metrics", {}), durees=report.get("timings_s", {}))
    except PivotError as e:
        _update(job_id, etat="echec", fin=time.time(), erreur=str(e))
    except Exception as e:  # noqa: BLE001 : une tâche ne doit jamais tuer le worker
        log.error("Tâche %s : %s", job_id, traceback.format_exc())
        _update(job_id, etat="echec", fin=time.time(), erreur=f"Erreur interne : {e}")
    finally:
        pivot_log.removeHandler(handler)
        handler.close()


def _worker() -> None:
    while True:
        _run(_queue.get())


@app.on_event("startup")
def _startup() -> None:
    logging.getLogger("pivot").setLevel(logging.INFO)
    _load_existing()
    threading.Thread(target=_worker, name="pivot-worker", daemon=True).start()
    log.info("Pivot API : %d tâche(s) chargée(s) depuis %s", len(_jobs), DATA_DIR)


# --- routes ---------------------------------------------------------------------------------------

def _public(job: dict, request: Request) -> dict:
    root = request.scope.get("root_path", "").rstrip("/")
    out = {k: v for k, v in job.items() if k != "metriques"} | {"url": f"{root}/panoramas/{job['id']}"}
    if job["etat"] == "termine":
        out["image"] = f"{root}/panoramas/{job['id']}/pano.jpg"
    if job["etat"] in ("en_attente", "en_cours"):
        with _lock:
            waiting = [j for j in _jobs.values() if j["etat"] == "en_attente" and j["cree"] < job["cree"]]
        out["position_file"] = len(waiting)
    return out


@app.post("/panoramas", status_code=202)
async def create_panorama(
    request: Request,
    images: list[UploadFile] = File(..., description="photos jpg/png, et capture.json de Pivot Capture si disponible"),
    width: int = Form(4096, ge=512, le=8192),
    seam: Literal["voronoi", "graphcut"] = Form("voronoi"),
    fill: Literal["none", "inpaint", "color", "gradient"] = Form("inpaint"),
    unordered: bool = Form(False),
):
    if len(images) > MAX_FILES:
        raise HTTPException(413, f"{MAX_FILES} fichiers au maximum.")
    job_id = uuid.uuid4().hex[:12]
    img_dir = os.path.join(_job_dir(job_id), "images")
    os.makedirs(img_dir)
    n_images, has_capture = 0, False
    for up in images:
        name = os.path.basename(up.filename or "")
        if name == CAPTURE_JSON:
            has_capture = True
        elif name.lower().endswith(EXTENSIONS):
            n_images += 1
        else:
            continue                                   # fichier ignoré (ni image ni capture.json)
        data = await up.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"{name} dépasse {MAX_BYTES // 2**20} Mo.")
        with open(os.path.join(img_dir, name), "wb") as f:
            f.write(data)
    if n_images < 2:
        raise HTTPException(422, f"Il faut au moins 2 photos jpg/png ({n_images} reçue(s)).")
    job = {"id": job_id, "etat": "en_attente", "cree": time.time(), "images": n_images,
           "capture_json": has_capture,
           "options": {"width": width, "seam": seam, "fill": fill, "unordered": unordered},
           "avertissements": [], "erreur": None}
    with _lock:
        _jobs[job_id] = job
        _save(job)
    _queue.put(job_id)
    return _public(job, request)


@app.get("/panoramas")
def list_panoramas(request: Request):
    with _lock:
        jobs = sorted(_jobs.values(), key=lambda j: j["cree"], reverse=True)
    return [_public(j, request) for j in jobs]


@app.get("/panoramas/{job_id}")
def get_panorama(job_id: str, request: Request):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Tâche inconnue.")
    return _public(job, request) | {"metriques": job.get("metriques")}


@app.get("/panoramas/{job_id}/pano.jpg")
def get_panorama_image(job_id: str):
    job = _jobs.get(job_id)
    if not job or job["etat"] != "termine":
        raise HTTPException(404, "Panorama non disponible.")
    return FileResponse(os.path.join(_job_dir(job_id), "pano.jpg"), media_type="image/jpeg",
                        filename=f"pivot-{job_id}.jpg")


@app.get("/panoramas/{job_id}/log.txt")
def get_panorama_log(job_id: str):
    path = os.path.join(_job_dir(job_id), "log.txt")
    if job_id not in _jobs or not os.path.isfile(path):
        raise HTTPException(404, "Journal non disponible.")
    return FileResponse(path, media_type="text/plain; charset=utf-8")


@app.get("/", response_class=HTMLResponse)
def index():
    return _PAGE


_PAGE = """<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Pivot</title>
<style>
body{font-family:system-ui,sans-serif;max-width:640px;margin:0 auto;padding:16px;background:#fafafa;color:#222}
fieldset{border:1px solid #ccc;border-radius:8px;margin:12px 0}label{display:block;margin:6px 0}
button{padding:10px 18px;font-size:16px;border-radius:8px;border:0;background:#2563eb;color:#fff}
.job{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px;margin:8px 0}
.job img{width:100%;border-radius:4px}.w{color:#b45309}.e{color:#b91c1c}code{font-size:12px}
</style></head><body>
<h1>Pivot</h1>
<p>Envoyez les photos d'un tour (et le <code>capture.json</code> de Pivot Capture s'il existe).</p>
<form id="f">
<input type="file" name="images" multiple accept="image/jpeg,image/png,.json" required>
<fieldset><legend>Options</legend>
<label>Largeur <select name="width"><option>2048</option><option selected>4096</option><option>8192</option></select></label>
<label>Coutures <select name="seam"><option>voronoi</option><option>graphcut</option></select></label>
<label>Zénith / nadir <select name="fill"><option>inpaint</option><option>gradient</option><option>color</option><option>none</option></select></label>
<label><input type="checkbox" name="unordered" value="true"> Ordre inconnu / plusieurs rangées</label>
</fieldset><button>Assembler</button> <span id="msg"></span></form>
<h2>Tâches</h2><div id="jobs"></div>
<script>
const f=document.getElementById('f'),msg=document.getElementById('msg'),box=document.getElementById('jobs');
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
f.onsubmit=async e=>{e.preventDefault();msg.textContent='Envoi…';
  const r=await fetch('panoramas',{method:'POST',body:new FormData(f)});const j=await r.json();
  msg.textContent=r.ok?'Tâche '+j.id+' créée.':'Erreur : '+esc(JSON.stringify(j.detail));refresh();};
async function refresh(){const r=await fetch('panoramas');if(!r.ok)return;const jobs=await r.json();
  const html=jobs.map(j=>{const base='panoramas/'+j.id;let h='<div class="job"><b>'+j.id+'</b> · '+j.images+' photos'
   +(j.capture_json?' + capture.json':'')+' · <b>'+j.etat+'</b>';
   if(j.position_file)h+=' ('+j.position_file+' avant)';
   if(j.etat==='termine')h+='<br><a href="'+base+'/pano.jpg">pano.jpg</a> · <a href="/pano360/?src='
     +encodeURIComponent(j.image)+'">voir en 360°</a><br><img loading="lazy" src="'+base+'/pano.jpg">';
   if(j.erreur)h+='<div class="e">'+esc(j.erreur)+'</div>';
   (j.avertissements||[]).forEach(w=>h+='<div class="w">'+esc(w)+'</div>');
   if(j.etat!=='en_attente')h+='<a href="'+base+'/log.txt">journal</a>';return h+'</div>';}).join('');
  if(html!==last){last=html;box.innerHTML=html;}}
let last='';
refresh();setInterval(refresh,3000);
</script></body></html>"""
