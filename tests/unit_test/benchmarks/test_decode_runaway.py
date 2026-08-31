# SPDX-License-Identifier: Apache-2.0
"""Unit tests for decode-runaway detection helpers."""

from __future__ import annotations

import pytest

from benchmarks.benchmarker.data import RequestResult
from benchmarks.metrics.decode_runaway import (
    RUNAWAY_CHARS_PER_AUDIO_S,
    decode_chars_per_audio_s,
    summarize_decode_runaway,
)
from benchmarks.metrics.performance import compute_speed_metrics


def _transcription(
    *,
    text: str,
    audio_duration_s: float,
    latency_s: float,
) -> RequestResult:
    rtf = latency_s / audio_duration_s if audio_duration_s > 0 else 0.0
    return RequestResult(
        text=text,
        is_success=True,
        latency_s=latency_s,
        audio_duration_s=audio_duration_s,
        rtf=rtf,
    )


def _healthy(index: int) -> RequestResult:
    # A normal movies800-shaped clip: a short timestamped transcript for ~10 s of
    # audio, decoded in a fraction of a second.
    return _transcription(
        text=f"[0.12][S01] Fools.[1.22][S0{index % 2}] Oops.",
        audio_duration_s=9.7,
        latency_s=0.09,
    )


def _runaway() -> RequestResult:
    # The #975 shape: a 9.7 s clip whose greedy decode loops on laughter until it
    # hits the token budget, ~5.7K characters and ~10 s instead of ~40 tokens.
    return _transcription(
        text="[0.12][S01] Fools." + "[1.22][S01] Ha ha ha ha ha ha." * 190,
        audio_duration_s=9.7,
        latency_s=10.14,
    )


def test_decode_chars_per_audio_s_is_none_without_audio_duration() -> None:
    assert decode_chars_per_audio_s(_transcription(text="x", audio_duration_s=0.0, latency_s=1.0)) is None


def test_decode_chars_per_audio_s_separates_speech_from_a_loop() -> None:
    healthy_rate = decode_chars_per_audio_s(_healthy(0))
    runaway_rate = decode_chars_per_audio_s(_runaway())

    assert healthy_rate is not None and runaway_rate is not None
    assert healthy_rate < RUNAWAY_CHARS_PER_AUDIO_S < runaway_rate
    # The two are an order of magnitude apart, which is what makes a single
    # threshold safe across languages and markup styles.
    assert runaway_rate > 10 * healthy_rate


def test_summarize_counts_only_requests_above_the_threshold() -> None:
    successes = [_healthy(index) for index in range(798)] + [_runaway(), _runaway()]

    summary = summarize_decode_runaway(successes)

    assert summary["runaway_requests"] == 2
    assert summary["runaway_scored_requests"] == 800
    assert summary["decode_chars_per_audio_s_max"] > RUNAWAY_CHARS_PER_AUDIO_S


def test_summarize_reports_a_clean_corpus_as_zero() -> None:
    summary = summarize_decode_runaway([_healthy(index) for index in range(800)])

    assert summary["runaway_requests"] == 0
    assert summary["runaway_scored_requests"] == 800


def test_summarize_reports_the_tail_percentiles_miss() -> None:
    # One straggler in 800 sits beyond p95 and p99, which is why the existing
    # gates do not see it; the maxima do.
    successes = [_healthy(index) for index in range(799)] + [_runaway()]

    summary = summarize_decode_runaway(successes)

    assert summary["latency_max_s"] == pytest.approx(10.14, abs=1e-3)
    assert summary["rtf_max"] == pytest.approx(10.14 / 9.7, abs=1e-3)


def test_summarize_without_audio_durations_reports_none_not_zero() -> None:
    successes = [_transcription(text="hello", audio_duration_s=0.0, latency_s=0.2)]

    summary = summarize_decode_runaway(successes)

    assert summary["runaway_requests"] is None
    assert summary["runaway_scored_requests"] == 0
    assert summary["decode_chars_per_audio_s_max"] is None
    assert summary["latency_max_s"] == pytest.approx(0.2)


def test_summarize_rejects_a_non_positive_threshold() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        summarize_decode_runaway([_healthy(0)], chars_per_audio_s_threshold=0)


def test_speed_metrics_surface_the_runaway_counters() -> None:
    outputs = [_healthy(index) for index in range(9)] + [_runaway()]

    metrics = compute_speed_metrics(outputs, wall_clock_s=12.0)

    assert metrics["runaway_requests"] == 1
    assert metrics["runaway_scored_requests"] == 10
    assert metrics["latency_max_s"] == pytest.approx(10.14, abs=1e-3)
    assert metrics["rtf_max"] is not None
    # The straggler is invisible to the gates the ASR CI checks today.
    assert metrics["failed_requests"] == 0
    assert metrics["latency_p95_s"] < metrics["latency_max_s"]
