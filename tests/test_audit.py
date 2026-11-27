"""tests/test_audit.py - tests for tamper-evident hash-chained audit_log & WAL-safe backups."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from whc.audit import AuditRepository, BackupManager
from whc.database import Database
from whc.students import Student, StudentRepository


class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 11, 22, 8, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


def test_audit_log_is_append_only_and_hash_chain_detects_tampering(tmp_path: Path):
    clock = FakeClock()
    with Database(tmp_path / "audit.db") as db:
        audit = AuditRepository(db, clock=clock)
        e1 = audit.record(
            "STUDENT_ADDED",
            actor_user_id=1,
            actor_username="lead1",
            target_table="students",
            target_id="AAA0001",
            details={"password": "should-never-appear", "first_name": "Sam"},
        )
        e2 = audit.record(
            "RESULT_ADDED",
            actor_user_id=1,
            actor_username="lead1",
            target_table="test_results",
            target_id=42,
            details={"value": 34.5},
        )
        assert e1.details["password"] == "[REDACTED]"
        assert e2.prev_hash == e1.row_hash
        ok, msg = audit.verify_chain()
        assert ok is True and "2 entries verified" in msg

        # Triggers block UPDATE and DELETE
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            with db.transaction() as conn:
                conn.execute("UPDATE audit_log SET action = 'FAKE' WHERE audit_id = 1")
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            with db.transaction() as conn:
                conn.execute("DELETE FROM audit_log WHERE audit_id = 1")

        # Even if an attacker with raw file access DROPS the triggers and alters row 1,
        # verify_chain() cryptographically catches the tampering!
        with db.transaction() as conn:
            conn.execute("DROP TRIGGER audit_log_no_update")
            conn.execute("UPDATE audit_log SET actor_username = 'innocent' WHERE audit_id = 1")
        ok_after, reason = audit.verify_chain()
        assert ok_after is False
        assert "tampered row_hash at audit_id=1" in reason


def test_backup_manager_per_type_retention_daily_dedup_and_restore_drill(tmp_path: Path):
    clock = FakeClock()
    db_path = tmp_path / "live.db"
    backup_dir = tmp_path / "backups"

    with Database(db_path) as db:
        students = StudentRepository(db)
        audit = AuditRepository(db, clock=clock)
        students.add(Student.new("AAA0001", "Sam", "Lee", "14/03/2012", "Male"))

        bm = BackupManager(
            db,
            backup_dir=backup_dir,
            retention={"daily": 3, "pre_bulk": 2, "manual": 2},
            clock=clock,
        )

        # Daily dedup: two calls on same day create only 1 daily snapshot
        d1 = bm.ensure_daily_backup(audit=audit, actor_username="lead1")
        assert d1 is not None and d1.exists()
        assert bm.ensure_daily_backup(audit=audit, actor_username="lead1") is None

        # Create 4 pre_bulk backups on the same day -> pruned to 2, NEVER touches daily!
        for _ in range(4):
            clock.advance(seconds=5)
            bm.create_backup("pre_bulk", audit=audit, actor_username="lead1")

        # Create 3 more daily backups across subsequent days -> daily pruned to 3
        for _ in range(3):
            clock.advance(days=1)
            assert bm.ensure_daily_backup(audit=audit, actor_username="lead1") is not None

        listed = bm.list_backups()
        dailies = [b for b in listed if b.reason == "daily"]
        pre_bulks = [b for b in listed if b.reason == "pre_bulk"]
        assert len(dailies) == 3
        assert len(pre_bulks) == 2

        # Real restore drill: restore latest backup into a brand-new DB file and verify data!
        restored_path = tmp_path / "restored.db"
        bm.restore_to(dailies[0].path, restored_path)
        with Database(restored_path) as restored_db:
            restored_students = StudentRepository(restored_db)
            loaded = restored_students.get("AAA0001")
            assert loaded.full_name == "Sam Lee"
            restored_audit = AuditRepository(restored_db)
            ok, _ = restored_audit.verify_chain()
            assert ok is True
