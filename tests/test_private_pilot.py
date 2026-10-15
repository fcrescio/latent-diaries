import random
import runpy


def test_group_split_keeps_capture_days_and_near_duplicates_together():
    split = runpy.run_path("scripts/select_private_pilot.py")["grouped_split"]
    generator = random.Random(123)
    rows = [{"id": f"fixture-{i}", "capture_group": f"day-{i}",
             "dhash": generator.getrandbits(64)} for i in range(40)]
    rows[1]["capture_group"] = rows[0]["capture_group"]
    rows[2]["dhash"] = rows[1]["dhash"] ^ 1  # link across capture days
    result = split(rows)
    by_id = {r['id']: r for r in result}
    assert by_id['fixture-0']['scene_group'] == by_id['fixture-2']['scene_group']
    train = [r for r in result if r['split'] == 'train']
    held = [r for r in result if r['split'] == 'heldout']
    assert len(train) >= 20 and len(held) >= 4
    assert not {r['capture_group'] for r in train} & {r['capture_group'] for r in held}
    assert not {r['scene_group'] for r in train} & {r['scene_group'] for r in held}
    assert result == split(rows)
