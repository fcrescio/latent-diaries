"""Local, authenticated, human-in-the-loop gallery. Not intended for public deployment."""

import csv
import hmac
import io
import os
import hashlib
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from jinja2 import Environment, FileSystemLoader, select_autoescape
from latent_diaries.storage import open_db

APP_DIR = Path(__file__).parent
security = HTTPBasic(auto_error=False)
TEMPLATES = Environment(loader=FileSystemLoader(str(APP_DIR)), autoescape=select_autoescape(["html"]))



def create_app(db_path: str | None = None, assets_dir: str | None = None,
               review_user: str | None = None, review_password: str | None = None) -> FastAPI:
    app = FastAPI(title="Latent Diaries — ascolto", docs_url=None, redoc_url=None, openapi_url=None)
    db_path = Path(db_path or os.environ.get("REVIEW_DB", "run-data/reviews.sqlite3"))
    assets = Path(assets_dir or os.environ.get("REVIEW_ASSETS", "run-data/review-assets"))
    user = review_user if review_user is not None else os.environ.get("REVIEW_USER", "reviewer")
    password = review_password if review_password is not None else os.environ.get("REVIEW_PASSWORD", "")
    assets.mkdir(parents=True, exist_ok=True)
    with open_db(db_path):
        pass

    def authorize(credentials: Annotated[HTTPBasicCredentials | None, Depends(security)]) -> str:
        if not password:
            raise HTTPException(503, detail="Set REVIEW_PASSWORD in .env to enable review")
        if credentials is None or not (hmac.compare_digest(credentials.username, user) and
                                      hmac.compare_digest(credentials.password, password)):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required",
                                headers={"WWW-Authenticate": "Basic realm=LatentDiaries"})
        return credentials.username

    def csrf(clip_id: str) -> str:
        return hmac.new(password.encode(), ("rating:" + clip_id + ":" + user).encode(),
                        hashlib.sha256).hexdigest()

    @app.get("/health")
    def health():
        return {"status": "ok", "configured": bool(password)}

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request, reviewer: Annotated[str, Depends(authorize)]):
        with open_db(db_path) as con:
            clips = con.execute("""SELECT c.*, r.grounding, r.evocation, r.voice, r.surprise, r.note
                FROM clips c LEFT JOIN ratings r ON c.id=r.clip_id AND r.reviewer=?
                ORDER BY c.created_at, c.id""", (reviewer,)).fetchall()
        html = TEMPLATES.get_template("index.html").render(
            clips=[dict(c, csrf_token=csrf(c["id"])) for c in clips], username=reviewer)
        return HTMLResponse(html, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @app.get("/media/{clip_id}/{kind}")
    def media(clip_id: str, kind: str, _: Annotated[str, Depends(authorize)]):
        if kind not in {"image", "audio"}:
            raise HTTPException(404)
        with open_db(db_path) as con:
            row = con.execute("SELECT image_file,audio_file FROM clips WHERE id=?", (clip_id,)).fetchone()
        if row is None:
            raise HTTPException(404)
        filename = row[f"{kind}_file"]
        path = (assets / filename).resolve()
        if not path.is_relative_to(assets.resolve()) or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, headers={"Cache-Control": "private, max-age=60", "X-Content-Type-Options": "nosniff"})

    @app.post("/review/{clip_id}")
    def review(clip_id: str, reviewer: Annotated[str, Depends(authorize)],
               csrf_token: Annotated[str, Form()],
               grounding: Annotated[int, Form(ge=1, le=5)],
               voice: Annotated[int, Form(ge=1, le=5)],
               evocation: Annotated[int, Form(ge=1, le=5)],
               surprise: Annotated[int, Form(ge=1, le=5)],
               note: Annotated[str, Form(max_length=1000)] = ""):
        if not hmac.compare_digest(csrf_token, csrf(clip_id)):
            raise HTTPException(403, "Invalid review token")
        with open_db(db_path) as con:
            if not con.execute("SELECT 1 FROM clips WHERE id=?", (clip_id,)).fetchone():
                raise HTTPException(404)
            con.execute("""INSERT INTO ratings
                (clip_id,reviewer,grounding,voice,evocation,surprise,note)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(clip_id,reviewer) DO UPDATE SET grounding=excluded.grounding,
                voice=excluded.voice,evocation=excluded.evocation,
                surprise=excluded.surprise,note=excluded.note,updated_at=CURRENT_TIMESTAMP""",
                        (clip_id,reviewer,grounding,voice,evocation,surprise,note))
        return Response(status_code=303, headers={"Location": "/#" + clip_id})

    @app.get("/api/reviews")
    def api_reviews(_: Annotated[str, Depends(authorize)]):
        with open_db(db_path) as con:
            rows = [dict(r) for r in con.execute("""SELECT c.run_id,c.blind_label,r.*
                FROM ratings r JOIN clips c ON c.id=r.clip_id ORDER BY c.run_id,c.id""")]
        return JSONResponse({"ratings": rows}, headers={"Cache-Control": "no-store"})

    @app.get("/export.csv")
    def export(_: Annotated[str, Depends(authorize)]):
        with open_db(db_path) as con:
            rows = [dict(r) for r in con.execute("""SELECT c.run_id,c.blind_label,r.*
                FROM ratings r JOIN clips c ON c.id=r.clip_id ORDER BY c.run_id,c.id""")]
        output = io.StringIO()
        fields = ["run_id", "blind_label", "clip_id", "reviewer", "grounding", "voice",
                  "evocation", "surprise", "note", "updated_at"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        return Response(output.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": "attachment; filename=ratings.csv", "Cache-Control": "no-store"})

    from services.review_web.audio_probe import install
    install(app, db_path, assets, authorize, csrf, TEMPLATES)
    return app


app = create_app()
