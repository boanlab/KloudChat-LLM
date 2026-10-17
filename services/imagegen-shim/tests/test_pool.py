"""The next picture goes to the least busy node; a node that refused sits out."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pool import Pool  # noqa: E402

A, B = "http://gpu-1:8091", "http://gpu-2:8091"


def test_the_least_busy_backend_goes_first():
    pool = Pool([A, B])
    pool.busy[A] = 2
    assert pool.order()[0] == B
    pool.busy[B] = 3
    assert pool.order()[0] == A


def test_idle_backends_take_turns():
    pool = Pool([A, B])
    assert [pool.order()[0] for _ in range(4)] == [A, B, A, B]


def test_a_refusing_backend_sits_out_its_cooldown():
    now = [100.0]
    pool = Pool([A, B], cooldown=30.0, clock=lambda: now[0])
    pool.busy[B] = 5
    pool.mark_down(A)
    assert pool.order() == [B, A]  # still tried last, if B fails too
    now[0] += 31
    assert pool.order()[0] == A
    pool.mark_down(A)
    pool.mark_up(A)
    assert pool.order()[0] == A


def test_one_backend_is_always_the_answer():
    pool = Pool([A])
    pool.mark_down(A)
    assert pool.order() == [A]
