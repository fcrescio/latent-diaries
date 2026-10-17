import re

from fastapi.testclient import TestClient

from latent_diaries.storage import open_db
from services.review_web.app import create_app


def test_audio_probe_blinding_scores_and_voice_gate(tmp_path):
    db, assets = tmp_path / "db", tmp_path / "assets"
    client = TestClient(create_app(str(db), str(assets), review_user="a", review_password="test-only"))
    auth = ("a", "test-only")
    with open_db(db) as con:
        for n in range(2):
            con.execute(
                "INSERT INTO audio_probe_clips VALUES(?,?,?,?,?,?)",
                (f"omx-{n}", "OMX-01", f"OMX-{n + 1:02d}", f"secret-{n}.wav", "ref.wav", int(n == 0)),
            )
    assert client.get("/audio-probe").status_code == 401
    assert client.get("/audio-probe/media/omx-0/generated").status_code == 401
    html = client.get("/audio-probe", auth=auth).text
    assert "secret-" not in html and "voice_applies" not in html and "/voice/" not in html
    tokens = re.findall('name="csrf_token" value="([a-f0-9]+)"', html)
    vote = {"language": "it", "intelligibility": 4, "novelty": "new", "coherence": "", "note": "good"}
    assert (
        client.post(
            "/audio-probe/review/omx-0", auth=auth, data=vote | {"csrf_token": "bad"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/audio-probe/review/omx-0",
            auth=auth,
            data=vote | {"csrf_token": tokens[0], "intelligibility": 5},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/audio-probe/voice/omx-0", auth=auth, data={"csrf_token": tokens[0], "voice": 5}
        ).status_code
        == 409
    )
    for n in range(2):
        response = client.post(
            f"/audio-probe/review/omx-{n}",
            auth=auth,
            data=vote | {"csrf_token": tokens[n]},
            follow_redirects=False,
        )
        assert response.status_code == 303
    html = client.get("/audio-probe", auth=auth).text
    assert "/voice/omx-0" in html and "/voice/omx-1" not in html
    assert (
        client.post(
            "/audio-probe/voice/omx-0",
            auth=auth,
            data={"csrf_token": tokens[0], "voice": 5},
            follow_redirects=False,
        ).status_code
        == 303
    )
    result = client.get("/audio-probe/ratings.json", auth=auth).json()["ratings"]
    assert result[0]["voice"] == 5 and result[1]["voice"] is None and result[0]["coherence"] is None
    with open_db(db) as con:
        assert con.execute("SELECT count(*) FROM ratings").fetchone()[0] == 0
        con.execute("UPDATE audio_probe_clips SET audio_file='../private.wav' WHERE id='omx-0'")
    assert client.get("/audio-probe/media/omx-0/generated", auth=auth).status_code == 404
