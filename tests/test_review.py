from pathlib import Path
from fastapi.testclient import TestClient
from latent_diaries.demo import create_demo
from latent_diaries.storage import open_db
from services.review_web.app import create_app


def test_evaluation_flow(tmp_path: Path):
    db, assets = tmp_path / "reviews.db", tmp_path / "assets"
    assert create_demo(tmp_path / "demo", db, assets) == 3
    app = create_app(str(db), str(assets), review_user="alice", review_password="test-only")
    client = TestClient(app)
    assert client.get("/").status_code == 401
    assert client.get("/health").json()["configured"] is True
    auth = ("alice", "test-only")
    html = client.get("/", auth=auth).text
    assert "SYNTHETIC" not in html  # hide model identity in blind gallery
    assert "Sala d'ascolto" in html
    with open_db(db) as con:
        id_ = con.execute("SELECT id FROM clips ORDER BY id LIMIT 1").fetchone()["id"]
    bad = client.post(f"/review/{id_}", auth=auth,
                      data=dict(csrf_token="wrong", grounding="4", voice="4", evocation="4", surprise="4", note="Good"),
                      follow_redirects=False)
    assert bad.status_code == 403
    import re
    token = re.search(r'name="csrf_token" value="([0-9a-f]+)"', html).group(1)
    response = client.post(f"/review/{id_}", auth=auth,
                           data=dict(csrf_token=token, grounding="3", voice="4", evocation="5", surprise="4", note="Evocativo"),
                           follow_redirects=False)
    assert response.status_code == 303
    assert client.get("/api/reviews", auth=auth).json()["ratings"][0]["evocation"] == 5
    assert "Evocativo" in client.get("/export.csv", auth=auth).text
    assert client.get(f"/media/{id_}/image", auth=auth).status_code == 200
    assert client.get(f"/media/{id_}/audio", auth=auth).status_code == 200
    assert client.get(f"/media/{id_}/audio").status_code == 401


def test_review_requires_password(tmp_path):
    client = TestClient(create_app(str(tmp_path / "db"), str(tmp_path / "assets"), review_password=""))
    assert client.get("/").status_code == 503
