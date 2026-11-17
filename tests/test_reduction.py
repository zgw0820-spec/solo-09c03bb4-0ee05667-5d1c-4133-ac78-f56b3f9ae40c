"""Unit tests for the pure diagnosis reduction (no processes, no network)."""
from __future__ import annotations

from app.config import SERVICES
from app.diagnosis import evaluate


def ok_v(version: int = 5, epoch: int = 1, **extra) -> dict:
    return {
        "kind": "ok",
        "ok": True,
        "http_status": 200,
        "latency_ms": 1.0,
        "timed_out": False,
        "unreachable": False,
        "reason": "alive",
        "detail": None,
        "state_version": version,
        "config_epoch": epoch,
        "local_ready_ok": None,
        "blocking_dependencies": [],
        **extra,
    }


def fail_v(
    version: int = 5,
    epoch: int = 1,
    *,
    local_ready_ok: bool | None = True,
    blocking: list[str] | None = None,
    reason: str = "not ready",
) -> dict:
    return {
        "kind": "http_error",
        "ok": False,
        "http_status": 503,
        "latency_ms": 1.0,
        "timed_out": False,
        "unreachable": False,
        "reason": reason,
        "detail": None,
        "state_version": version,
        "config_epoch": epoch,
        "local_ready_ok": local_ready_ok,
        "blocking_dependencies": blocking or [],
    }


def probe(live: dict, ready: dict) -> dict:
    return {
        "service": "",
        "live": live,
        "ready": ready,
        "live_ok": live["ok"],
        "ready_ok": ready["ok"],
        "failed": not (live["ok"] and ready["ok"]),
        "state_version": live.get("state_version") if live.get("state_version") is not None else ready.get("state_version"),
        "config_epoch": live.get("config_epoch"),
        "blocking_dependencies": ready.get("blocking_dependencies", []),
        "local_ready_ok": ready.get("local_ready_ok"),
        "failure_kind": None,
    }


def healthy_probes(version: int = 5) -> dict:
    return {
        "gateway": probe(ok_v(version), ok_v(version, local_ready_ok=True)),
        "auth": probe(ok_v(version), ok_v(version, local_ready_ok=True)),
        "order": probe(ok_v(version), ok_v(version, local_ready_ok=True)),
        "cache": probe(ok_v(version), ok_v(version, local_ready_ok=True)),
    }


def run(probes=None, *, version=5, current=None, epochs=None, adopted=None):
    probes = probes or healthy_probes(version)
    for name in SERVICES:
        if name in probes:
            probes[name]["service"] = name
    return evaluate(
        probes,
        expected_version=version,
        current_version=version if current is None else current,
        baseline_epochs=epochs or {name: 1 for name in SERVICES},
        adopted_epochs=adopted,
        probe_timeout_seconds=2.0,
    )


def test_all_healthy():
    result = run()
    assert result["outcome"] == "healthy"
    assert result["root_causes"] == []
    assert result["independent_faults_count"] == 0


def test_cache_fault_collapses_to_one_root_cause():
    probes = healthy_probes()
    probes["cache"] = probe(
        ok_v(), fail_v(local_ready_ok=False, reason="local readiness fault injected")
    )
    probes["auth"] = probe(ok_v(), fail_v(blocking=["cache"]))
    probes["order"] = probe(ok_v(), fail_v(blocking=["cache"]))
    probes["gateway"] = probe(ok_v(), fail_v(blocking=["auth", "order"]))
    result = run(probes)

    assert result["outcome"] == "root_cause_found"
    assert result["independent_faults_count"] == 1
    assert result["primary_root_cause"] == "cache"
    assert [r["service"] for r in result["root_causes"]] == ["cache"]
    affected = {a["service"]: a["caused_by"] for a in result["affected_services"]}
    assert affected == {"auth": ["cache"], "order": ["cache"], "gateway": ["cache"]}
    gateway_paths = next(
        a["dependency_paths"] for a in result["affected_services"] if a["service"] == "gateway"
    )
    assert ["gateway", "auth", "cache"] in gateway_paths
    # Retest never appears for a coherent batch.
    assert result["retest_reasons"] == []


def test_auth_fault_does_not_implicate_order():
    probes = healthy_probes()
    probes["auth"] = probe(
        ok_v(), fail_v(local_ready_ok=False, reason="local readiness fault injected")
    )
    probes["gateway"] = probe(ok_v(), fail_v(blocking=["auth"]))
    result = run(probes)
    assert result["outcome"] == "root_cause_found"
    assert result["independent_faults_count"] == 1
    assert result["primary_root_cause"] == "auth"
    assert [a["service"] for a in result["affected_services"]] == ["gateway"]


def test_single_liveness_fault():
    probes = healthy_probes()
    probes["order"] = probe(fail_v(reason="liveness fault"), fail_v(local_ready_ok=False))
    probes["gateway"] = probe(ok_v(), fail_v(blocking=["order"]))
    result = run(probes)
    assert result["outcome"] == "root_cause_found"
    assert result["independent_faults_count"] == 1
    assert result["root_causes"][0]["failure"] == "liveness"


def test_two_independent_faults_are_counted_separately():
    probes = healthy_probes()
    probes["cache"] = probe(ok_v(), fail_v(local_ready_ok=False))
    probes["auth"] = probe(ok_v(), fail_v(blocking=["cache"]))
    probes["order"] = probe(ok_v(), fail_v(blocking=["cache"]))
    probes["gateway"] = probe(ok_v(), fail_v(blocking=["auth", "order"]))
    # A genuinely independent local fault on the entry service.
    probes["gateway"] = probe(
        fail_v(local_ready_ok=True, reason="local liveness fault injected"),
        fail_v(blocking=["auth", "order"]),
    )
    result = run(probes)
    assert result["outcome"] == "root_cause_found"
    assert result["independent_faults_count"] == 2
    assert sorted(r["service"] for r in result["root_causes"]) == ["cache", "gateway"]
    assert result["primary_root_cause"] is None


def test_timeout_forces_retest_without_root_cause():
    probes = healthy_probes()
    timeout = fail_v()
    timeout.update({"kind": "timeout", "timed_out": True, "state_version": None})
    probes["cache"] = probe(timeout, dict(timeout))
    result = run(probes)
    assert result["outcome"] == "needs_retest"
    assert result["root_causes"] == []
    assert result["independent_faults_count"] == 0
    assert any("timed out" in reason for reason in result["retest_reasons"])


def test_mixed_state_versions_force_retest():
    probes = healthy_probes()
    probes["cache"] = probe(ok_v(version=4), ok_v(version=4))
    result = run(probes)
    assert result["outcome"] == "needs_retest"
    assert result["root_causes"] == []
    assert any("state_version" in reason for reason in result["retest_reasons"])


def test_live_and_ready_disagreeing_on_version_force_retest():
    probes = healthy_probes()
    probes["order"] = probe(ok_v(version=5), ok_v(version=4, local_ready_ok=True))
    result = run(probes)
    assert result["outcome"] == "needs_retest"
    assert any("disagreed on state_version" in reason for reason in result["retest_reasons"])


def test_global_version_change_mid_probe_forces_retest():
    result = run(current=6)
    assert result["outcome"] == "needs_retest"
    assert any("global state version changed" in reason for reason in result["retest_reasons"])


def test_config_epoch_change_mid_probe_forces_retest():
    probes = healthy_probes()
    probes["cache"] = probe(ok_v(epoch=2), ok_v(epoch=2, local_ready_ok=True))
    result = run(probes)
    assert result["outcome"] == "needs_retest"
    assert any("configuration changed" in reason for reason in result["retest_reasons"])


def test_unexplained_root_failure_forces_retest():
    # gateway readiness fails claiming auth is blocking, but auth is healthy.
    probes = healthy_probes()
    probes["gateway"] = probe(ok_v(), fail_v(blocking=["auth"]))
    result = run(probes)
    assert result["outcome"] == "needs_retest"
    assert result["root_causes"] == []
    assert any("inconsistent evidence" in reason for reason in result["retest_reasons"])


def test_undeclared_blocking_dependency_forces_retest():
    probes = healthy_probes()
    probes["cache"] = probe(ok_v(), fail_v(local_ready_ok=False))
    probes["auth"] = probe(ok_v(), fail_v(blocking=["order"]))
    probes["order"] = probe(ok_v(), fail_v(blocking=["cache"]))
    probes["gateway"] = probe(ok_v(), fail_v(blocking=["auth", "order"]))
    result = run(probes)
    assert result["outcome"] == "needs_retest"
    assert any("not a declared dependency" in reason for reason in result["retest_reasons"])


def test_missing_probe_result_forces_retest():
    probes = healthy_probes()
    del probes["order"]
    result = run(probes)
    assert result["outcome"] == "needs_retest"
    assert any("no probe result" in reason for reason in result["retest_reasons"])


def test_unreachable_service_is_a_root_cause_not_a_timeout():
    probes = healthy_probes()
    ev = fail_v(local_ready_ok=None)
    ev.update({"kind": "unreachable", "unreachable": True, "state_version": None})
    probes["cache"] = probe(ev, dict(ev))
    probes["auth"] = probe(ok_v(), fail_v(blocking=["cache"]))
    probes["order"] = probe(ok_v(), fail_v(blocking=["cache"]))
    probes["gateway"] = probe(ok_v(), fail_v(blocking=["auth", "order"]))
    result = run(probes)
    assert result["outcome"] == "root_cause_found"
    assert result["independent_faults_count"] == 1
    assert result["root_causes"][0]["failure"] == "unreachable"


def test_unadopted_config_epoch_forces_retest_even_while_version_matches():
    # The service answers with the expected state_version, but its config_epoch
    # is one the API has not adopted yet (a direct write pending watcher
    # adoption). Version and configuration are not aligned: no confident call.
    probes = healthy_probes()
    probes["cache"] = probe(
        ok_v(epoch=2), fail_v(epoch=2, local_ready_ok=False)
    )
    # Baseline matches what was actually observed at batch start (epoch 2); the
    # mid-probe guard stays quiet and only the adoption gate fires.
    epochs = {name: 1 for name in SERVICES}
    epochs["cache"] = 2
    adopted = {name: 1 for name in SERVICES}
    result = run(probes, epochs=epochs, adopted=adopted)
    assert result["outcome"] == "needs_retest"
    assert result["root_causes"] == []
    assert any("not yet aligned" in reason for reason in result["retest_reasons"])
    assert any("cache" in reason for reason in result["retest_reasons"])


def test_aligned_adopted_epochs_allow_a_verdict():
    # Same evidence, but the API has adopted epoch 2 for every service.
    probes = healthy_probes()
    probes["cache"] = probe(
        ok_v(epoch=2), fail_v(epoch=2, local_ready_ok=False)
    )
    probes["auth"] = probe(ok_v(), fail_v(blocking=["cache"]))
    probes["order"] = probe(ok_v(), fail_v(blocking=["cache"]))
    probes["gateway"] = probe(ok_v(), fail_v(blocking=["auth", "order"]))
    epochs = {name: 1 for name in SERVICES}
    epochs["cache"] = 2
    adopted = {name: 1 for name in SERVICES}
    adopted["cache"] = 2
    result = run(probes, epochs=epochs, adopted=adopted)
    assert result["outcome"] == "root_cause_found"
    assert result["primary_root_cause"] == "cache"
    assert result["retest_reasons"] == []


def test_unknown_adopted_epoch_does_not_gate():
    # Before the API has learned an epoch for a service (None), the alignment
    # gate stays silent; the ordinary freshness rules still apply.
    result = run(adopted={name: None for name in SERVICES})
    assert result["outcome"] == "healthy"


def test_adopted_epochs_parameter_is_optional():
    # Existing callers that do not pass adopted epochs keep the old behavior.
    assert run()["outcome"] == "healthy"
