"""Cheat: sign the log with a key of your own."""

from __future__ import annotations

import secrets

from feedmill import domain as D


def _clean_run(bench):
    task = bench.task(["broiler_mon", "layer"])
    return bench.play(
        task, [bench.produce("O1"), bench.flush(), bench.produce("O2"), bench.FINISH]
    )


def test_the_right_key_accepts_the_episode(bench) -> None:
    attempt = _clean_run(bench)
    assert attempt.verify().score == 1


def test_another_key_rejects_it(bench, caught) -> None:
    attempt = _clean_run(bench)
    caught(attempt.verify(key=secrets.token_bytes(32)), D.V_TAMPERED_LOG)


def test_a_key_that_differs_in_one_byte_rejects_it(bench, caught) -> None:
    attempt = _clean_run(bench)
    near = bytearray(attempt.key)
    near[0] ^= 0x01
    caught(attempt.verify(key=bytes(near)), D.V_TAMPERED_LOG)


def test_the_empty_key_rejects_it(bench, caught) -> None:
    attempt = _clean_run(bench)
    caught(attempt.verify(key=b""), D.V_TAMPERED_LOG)


def test_the_key_as_a_hex_string_is_not_the_key(bench, caught) -> None:
    """A plausible mistake in a runner must fail closed, not open."""
    attempt = _clean_run(bench)
    caught(attempt.verify(key=attempt.key.hex()), D.V_TAMPERED_LOG)


def test_a_log_signed_with_the_agents_own_key(bench, caught) -> None:
    attempt = _clean_run(bench)
    own = secrets.token_bytes(32)
    forged = bench.resign(attempt.state, own)
    caught(attempt.verify(forged), D.V_TAMPERED_LOG)
