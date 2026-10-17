"""Separate audio-only assessment; core votes stay blind until all are saved."""

import hmac
from typing import Annotated

from fastapi import Depends, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from latent_diaries.storage import open_db


def install(app, db_path, assets, authorize, csrf, templates):
    with open_db(db_path) as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS audio_probe_clips (
          id TEXT PRIMARY KEY, run_id TEXT NOT NULL, blind_label TEXT NOT NULL,
          audio_file TEXT NOT NULL, reference_file TEXT NOT NULL,
          voice_applies INTEGER NOT NULL CHECK(voice_applies IN (0,1)));
        CREATE TABLE IF NOT EXISTS audio_probe_ratings (
          clip_id TEXT NOT NULL REFERENCES audio_probe_clips(id), reviewer TEXT NOT NULL,
          language TEXT NOT NULL CHECK(language IN ('it','other','mixed','undecidable')),
          intelligibility INTEGER NOT NULL CHECK(intelligibility BETWEEN 0 AND 4),
          novelty TEXT NOT NULL CHECK(novelty IN ('new','repeat','undecidable')),
          coherence INTEGER CHECK(coherence BETWEEN 0 AND 4),
          voice INTEGER CHECK(voice BETWEEN 1 AND 5), note TEXT NOT NULL DEFAULT '',
          updated_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(clip_id,reviewer));
        """)

    def rows(con, reviewer):
        return con.execute(
            """SELECT c.*,r.language,r.intelligibility,r.novelty,r.coherence,r.voice,r.note
            FROM audio_probe_clips c LEFT JOIN audio_probe_ratings r
            ON c.id=r.clip_id AND r.reviewer=? ORDER BY c.blind_label""",
            (reviewer,),
        ).fetchall()

    def completed(items):
        return bool(items) and all(r["intelligibility"] is not None for r in items)

    @app.get("/audio-probe", response_class=HTMLResponse)
    def gallery(reviewer: Annotated[str, Depends(authorize)]):
        with open_db(db_path) as con:
            items = rows(con, reviewer)
        finished = completed(items)
        clips = []
        for row in items:
            c = dict(row)
            # Never send filenames or applicability while core voting remains blind.
            clips.append(
                {
                    k: c[k]
                    for k in [
                        "id",
                        "blind_label",
                        "language",
                        "intelligibility",
                        "novelty",
                        "coherence",
                        "voice",
                        "note",
                    ]
                }
                | {
                    "csrf_token": csrf("audio:" + c["id"]),
                    "voice_applies": bool(c["voice_applies"]) if finished else False,
                }
            )
        return HTMLResponse(
            templates.get_template("audio_probe.html").render(
                clips=clips,
                finished=finished,
                saved=sum(c["intelligibility"] is not None for c in clips),
            ),
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.get("/audio-probe/media/{clip_id}/{kind}")
    def media(clip_id: str, kind: str, _: Annotated[str, Depends(authorize)]):
        if kind not in {"generated", "reference"}:
            raise HTTPException(404)
        with open_db(db_path) as con:
            row = con.execute("SELECT * FROM audio_probe_clips WHERE id=?", (clip_id,)).fetchone()
        if row is None:
            raise HTTPException(404)
        path = (assets / row["audio_file" if kind == "generated" else "reference_file"]).resolve()
        if not path.is_relative_to(assets.resolve()) or not path.is_file():
            raise HTTPException(404)
        return FileResponse(
            path,
            media_type="audio/wav",
            filename=clip_id + ".wav",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.post("/audio-probe/review/{clip_id}")
    def review(
        clip_id: str,
        reviewer: Annotated[str, Depends(authorize)],
        csrf_token: Annotated[str, Form()],
        language: Annotated[str, Form()],
        intelligibility: Annotated[int, Form(ge=0, le=4)],
        novelty: Annotated[str, Form()],
        coherence: Annotated[str, Form()] = "",
        note: Annotated[str, Form(max_length=1000)] = "",
    ):
        if not hmac.compare_digest(csrf_token, csrf("audio:" + clip_id)):
            raise HTTPException(403)
        if language not in {"it", "other", "mixed", "undecidable"} or novelty not in {
            "new",
            "repeat",
            "undecidable",
        }:
            raise HTTPException(422)
        if coherence not in {"", "0", "1", "2", "3", "4"}:
            raise HTTPException(422)
        with open_db(db_path) as con:
            if not con.execute("SELECT 1 FROM audio_probe_clips WHERE id=?", (clip_id,)).fetchone():
                raise HTTPException(404)
            con.execute(
                """INSERT INTO audio_probe_ratings
               (clip_id,reviewer,language,intelligibility,novelty,coherence,note) VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(clip_id,reviewer) DO UPDATE SET language=excluded.language,
               intelligibility=excluded.intelligibility,novelty=excluded.novelty,
               coherence=excluded.coherence,note=excluded.note,updated_at=CURRENT_TIMESTAMP""",
                (
                    clip_id,
                    reviewer,
                    language,
                    intelligibility,
                    novelty,
                    int(coherence) if coherence else None,
                    note,
                ),
            )
        return Response(status_code=303, headers={"Location": "/audio-probe#" + clip_id})

    @app.post("/audio-probe/voice/{clip_id}")
    def voice(
        clip_id: str,
        reviewer: Annotated[str, Depends(authorize)],
        csrf_token: Annotated[str, Form()],
        voice: Annotated[int, Form(ge=1, le=5)],
    ):
        if not hmac.compare_digest(csrf_token, csrf("audio:" + clip_id)):
            raise HTTPException(403)
        with open_db(db_path) as con:
            items = rows(con, reviewer)
            row = next((r for r in items if r["id"] == clip_id), None)
            if not completed(items) or row is None or not row["voice_applies"]:
                raise HTTPException(
                    409,
                    "Complete blind core votes first; voice applies only to reference-conditioned clips",
                )
            con.execute(
                "UPDATE audio_probe_ratings SET voice=?,updated_at=CURRENT_TIMESTAMP WHERE clip_id=? AND reviewer=?",
                (voice, clip_id, reviewer),
            )
        return Response(status_code=303, headers={"Location": "/audio-probe#" + clip_id})

    @app.get("/audio-probe/ratings.json")
    def export(reviewer: Annotated[str, Depends(authorize)]):
        with open_db(db_path) as con:
            result = [
                dict(r)
                for r in con.execute(
                    """SELECT c.blind_label,r.* FROM audio_probe_ratings r
                JOIN audio_probe_clips c ON c.id=r.clip_id WHERE reviewer=? ORDER BY c.blind_label""",
                    (reviewer,),
                )
            ]
        return JSONResponse({"ratings": result}, headers={"Cache-Control": "no-store"})
