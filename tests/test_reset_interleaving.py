"""Coordinator-level tests for reset/direct-write interleavings.

These use an in-process fake of the four simulated services so the timing of a
direct write relative to the per-service reset commands is fully deterministic
(the real uvicorn stack cannot guarantee which service reset_all touches last).
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import pytest

from app.config import DEPENDS_ON, ROLES, SERVICES, Settings
from app.coordinator import Coordinator
from app.store import Store


class _FakeHTTP:
    async def aclose(self) -> None:
        return None


class FakeService:
    def __init__(self, name: str) -> None:
        self.name = name
        self.live_ok = True
        self.ready_ok = True
        self.probe_delay_ms = 0
        self.config_epoch = 1
        self.state_version = 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "service": self.name,
            "role": ROLES[self.name],
            "dependencies": list(DEPENDS_ON[self.name]),
            "live_ok": self.live_ok,
            "ready_ok": self.ready_ok,
            "probe_delay_ms": self.probe_delay_ms,
            "config_epoch": self.config_epoch,
            "state_version": self.state_version,
            "reachable": True,
            "error": None,
        }

    def put_fault(self, patch: dict[str, Any]) -> dict[str, Any]:
        changed = False
        for key in ("live_ok", "ready_ok", "probe_delay_ms"):
            if key in patch and patch[key] != getattr(self, key):
                setattr(self, key, patch[key])
                changed = True
        if changed:
            self.config_epoch += 1
        return self.snapshot()

    def reset(self) -> dict[str, Any]:
        self.live_ok = True
        self.ready_ok = True
        self.probe_delay_ms = 0
        self.config_epoch += 1
        return self.snapshot()


class FakeServiceClient:
    """Mimics ServiceClient's admin surface; no network, deterministic order."""

    def __init__(self, services: dict[str, FakeService]) -> None:
        self.services = services
        self.client = _FakeHTTP()
        # Called with the service name after each per-service reset completes.
        self.reset_hook = None

    async def set_fault(self, service: str, patch: dict[str, Any]) -> dict[str, Any]:
        return self.services[service].put_fault(patch)

    async def reset_fault(self, service: str) -> dict[str, Any]:
        snap = self.services[service].reset()
        if self.reset_hook is not None:
            self.reset_hook(service)
        return snap

    async def sync_version(self, service: str, version: int) -> dict[str, Any]:
        self.services[service].state_version = version
        return {"synced": True, **self.services[service].snapshot()}

    async def get_fault_state(self, service: str) -> dict[str, Any]:
        return self.services[service].snapshot()

    async def snapshot_states(self) -> list[dict[str, Any]]:
        return [self.services[s].snapshot() for s in SERVICES]

    async def sync_all(self, version: int) -> list[dict[str, Any]]:
        return [await self.sync_version(s, version) for s in SERVICES]

    async def reset_all(self) -> list[dict[str, Any]]:
        out = []
        for name in SERVICES:
            body = await self.reset_fault(name)
            out.append({"service": name, "reset": True, "error": None, **body})
        return out

    async def probe_all(self) -> dict[str, dict[str, Any]]:
        probes: dict[str, dict[str, Any]] = {}
        for name in SERVICES:
            svc = self.services[name]
            blocking = [
                dep
                for dep in DEPENDS_ON[name]
                if not (self.services[dep].live_ok and self.services[dep].ready_ok)
            ]
            ready_ok = svc.live_ok and svc.ready_ok and not blocking
            live = {
                "kind": "ok", "ok": True, "http_status": 200, "latency_ms": 0.1,
                "timed_out": False, "unreachable": False, "reason": "alive",
                "detail": None, "state_version": svc.state_version,
                "config_epoch": svc.config_epoch, "local_ready_ok": None,
                "blocking_dependencies": [], "timed_out_dependencies": [],
            }
            ready = dict(live)
            ready.update({
                "kind": "ok" if ready_ok else "http_error",
                "ok": ready_ok,
                "http_status": 200 if ready_ok else 503,
                "reason": "ready" if ready_ok else "not ready",
                "local_ready_ok": svc.ready_ok,
                "blocking_dependencies": blocking,
            })
            probes[name] = {
                "service": name, "role": ROLES[name], "live": live, "ready": ready,
                "live_ok": True, "ready_ok": ready_ok, "failed": not ready_ok,
                "state_version": svc.state_version, "config_epoch": svc.config_epoch,
                "blocking_dependencies": blocking,
                "timed_out_dependencies": [], "local_ready_ok": svc.ready_ok,
                "failure_kind": None,
            }
        return probes


@pytest.fixture()
def coordinator():
    tmp = Path(tempfile.mkdtemp()) / "coordinator.db"
    settings = Settings(database_path=str(tmp), watcher_interval_seconds=0)
    store = Store(str(tmp))
    services = {name: FakeService(name) for name in SERVICES}
    fake = FakeServiceClient(services)
    coord = Coordinator(settings, store)

    async def factory() -> FakeServiceClient:
        return fake

    coord._make_client = factory  # type: ignore[method-assign]
    coord._fake = fake  # type: ignore[attr-defined]
    coord._fakes = services  # type: ignore[attr-defined]
    return coord


async def _run_one_diagnosis(coord: Coordinator) -> dict[str, Any]:
    created = await coord.create_batch()
    await coord._active[created["batch_id"]]
    record = coord.get_batch(created["batch_id"])
    assert record["status"] == "completed"
    return record["result"]


async def _test_direct_write_after_service_reset_is_preserved_with_own_version_impl(
    coordinator,
):
    # A normal API write first, so the coordinator has adopted every epoch.
    await coordinator.set_fault("auth", {"probe_delay_ms": 1})
    before = coordinator.state_version

    def hook(reset_service: str) -> None:
        # Cache is cleared during the reset, then immediately re-faulted
        # directly on the service, bypassing the diagnosis API.
        if reset_service == "cache":
            coordinator._fakes["cache"].put_fault({"ready_ok": False})  # type: ignore

    coordinator._fake.reset_hook = hook  # type: ignore[attr-defined]
    receipt = await coordinator.reset_faults()

    # The reset's version and the direct write's version are distinct.
    assert receipt["state_version"] == before + 2
    assert receipt["previous_state_version"] == before
    uncleared = receipt["observed_uncleared_faults"]
    assert [item["service"] for item in uncleared] == ["cache"]
    assert uncleared[0]["ready_ok"] is False
    assert any("interleaved" in w for w in receipt["warnings"])

    # The direct change is preserved on the service and adopted the new version.
    cache = coordinator._fakes["cache"]  # type: ignore[attr-defined]
    assert cache.ready_ok is False
    assert cache.state_version == before + 2

    # The watcher must not invent yet another bump: the reset response already
    # reconciled the epoch under the independent version.
    assert await coordinator.check_external_fault_changes() is False
    assert coordinator.state_version == before + 2

    # After alignment the cache fault is located confidently.
    result = await _run_one_diagnosis(coordinator)
    assert result["outcome"] == "root_cause_found"
    assert result["primary_root_cause"] == "cache"
    assert result["independent_faults_count"] == 1


async def _test_direct_write_before_reset_is_covered_without_extra_change_impl(
    coordinator,
):
    await coordinator.set_fault("auth", {"probe_delay_ms": 1})
    before = coordinator.state_version

    # Bypass write that the watcher has not yet adopted.
    coordinator._fakes["cache"].put_fault({"ready_ok": False})  # type: ignore
    receipt = await coordinator.reset_faults()

    # Only the reset's own version bump; the overwritten fault is not a new
    # change and is not reported as an uncleared survivor.
    assert receipt["state_version"] == before + 1
    assert receipt["observed_uncleared_faults"] == []
    assert coordinator._fakes["cache"].ready_ok is True  # type: ignore
    assert await coordinator.check_external_fault_changes() is False
    assert coordinator.state_version == before + 1

    result = await _run_one_diagnosis(coordinator)
    assert result["outcome"] == "healthy"


async def _test_unadopted_direct_write_blocks_a_verdict_until_alignment_impl(
    coordinator,
):
    await coordinator.set_fault("auth", {"probe_delay_ms": 1})
    aligned = coordinator.state_version
    coordinator._fakes["cache"].put_fault({"ready_ok": False})  # type: ignore

    # No watcher adoption yet: configuration and state version diverge.
    blocked = await _run_one_diagnosis(coordinator)
    assert blocked["outcome"] == "needs_retest"
    assert blocked["root_causes"] == []
    assert any("not yet aligned" in r for r in blocked["retest_reasons"])

    # The watcher assigns the independent version and adopts the epoch.
    assert await coordinator.check_external_fault_changes() is True
    assert coordinator.state_version == aligned + 1
    aligned_result = await _run_one_diagnosis(coordinator)
    assert aligned_result["outcome"] == "root_cause_found"
    assert aligned_result["primary_root_cause"] == "cache"


async def _test_external_epoch_change_during_set_fault_is_not_absorbed_impl(coordinator):
    await coordinator.set_fault("auth", {"probe_delay_ms": 1})
    fake = coordinator._fake  # type: ignore[attr-defined]
    original_sync_all = fake.sync_all

    async def sync_all_with_external_write(version: int):
        # set_fault has already taken its pre-snapshot; an external write to a
        # different service now lands inside the command's window.
        coordinator._fakes["order"].put_fault({"ready_ok": False})  # type: ignore
        return await original_sync_all(version)

    fake.sync_all = sync_all_with_external_write
    await coordinator.set_fault("gateway", {"probe_delay_ms": 5})
    fake.sync_all = original_sync_all

    # The order epoch must remain un-adopted, so the watcher still bumps for it.
    assert await coordinator.check_external_fault_changes() is True


def test_direct_write_after_service_reset_is_preserved_with_own_version(coordinator):
    asyncio.run(_test_direct_write_after_service_reset_is_preserved_with_own_version_impl(coordinator))

def test_direct_write_before_reset_is_covered_without_extra_change(coordinator):
    asyncio.run(_test_direct_write_before_reset_is_covered_without_extra_change_impl(coordinator))

def test_unadopted_direct_write_blocks_a_verdict_until_alignment(coordinator):
    asyncio.run(_test_unadopted_direct_write_blocks_a_verdict_until_alignment_impl(coordinator))

def test_external_epoch_change_during_set_fault_is_not_absorbed(coordinator):
    asyncio.run(_test_external_epoch_change_during_set_fault_is_not_absorbed_impl(coordinator))

