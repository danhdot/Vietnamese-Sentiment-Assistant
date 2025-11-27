from pathlib import Path

from app.db import HistoryRepository


def test_history_repository_persists_entries(tmp_path: Path):
    repo = HistoryRepository(tmp_path / "test.sqlite3", retain=5)
    repo.add_entry("Xin chào", "POSITIVE", 0.9)
    entries = repo.list_entries(limit=10)
    assert len(entries) == 1
    assert entries[0].text == "Xin chào"
    assert entries[0].sentiment == "POSITIVE"
