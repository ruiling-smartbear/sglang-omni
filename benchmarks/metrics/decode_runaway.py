# SPDX-License-Identifier: Apache-2.0
"""Decode-runaway detection for transcription benchmarks.

A transcription request that enters a repetition loop keeps decoding until it
hits its token budget instead of stopping at the end of speech (#975). The
existing speed gates do not see it: the request completes, so ``failed_requests``
stays 0; text normalization collapses the repeated span, so corpus CER barely
moves and the sample lands in the tolerated above-50 partition; and a single
straggler in a corpus of hundreds sits beyond p95. Throughput is the only
symptom, and it is the noisiest gate there is.

These helpers make the failure class directly visible from data the benchmark
already collects, at no runtime cost:

* ``decode_chars_per_audio_s`` — transcript characters produced per second of
  input audio, per request. Speech has a bounded transcription rate; a loop does
  not, which is what separates the two by an order of magnitude rather than by a
  few percent.
* ``runaway_requests`` — how many requests exceed the rate threshold.
* ``latency_max_s`` / ``rtf_max`` — the tail the existing percentiles omit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

# Speech transcription has a bounded character rate. Fast conversational speech
# runs near 20 characters of transcript per second of audio, and timestamped
# diarization markup roughly doubles that; the reported #975 loop produced about
# 590 (a 9.7 s clip decoded to ~5.7K characters of repeated markers). The default
# sits well above dense real output and well below an observed loop, so it flags
# runaway decoding without depending on language or markup style.
RUNAWAY_CHARS_PER_AUDIO_S: float = 120.0


class _TranscriptionResult(Protocol):
    text: str
    audio_duration_s: float
    latency_s: float
    rtf: float


def decode_chars_per_audio_s(result: _TranscriptionResult) -> float | None:
    """Transcript characters per second of input audio, or None if unmeasurable."""
    audio_duration_s = float(getattr(result, "audio_duration_s", 0.0) or 0.0)
    if audio_duration_s <= 0:
        return None
    text = getattr(result, "text", "") or ""
    return len(text) / audio_duration_s


def summarize_decode_runaway(
    successes: Sequence[_TranscriptionResult],
    *,
    chars_per_audio_s_threshold: float = RUNAWAY_CHARS_PER_AUDIO_S,
) -> dict[str, float | int | None]:
    """Runaway-decode counters plus the latency/RTF maxima.

    Only successful requests with a measurable audio duration are scored;
    workloads that report no audio duration (so the rate is undefined) get the
    counters as None rather than a misleading zero.
    """
    if chars_per_audio_s_threshold <= 0:
        raise ValueError("chars_per_audio_s_threshold must be positive")

    rates = [
        rate
        for rate in (decode_chars_per_audio_s(result) for result in successes)
        if rate is not None
    ]
    latencies = [
        float(getattr(result, "latency_s", 0.0) or 0.0) for result in successes
    ]
    rtfs = [
        float(result.rtf)
        for result in successes
        if 0 < float(getattr(result, "rtf", 0.0) or 0.0) < float("inf")
    ]

    summary: dict[str, float | int | None] = {
        "latency_max_s": round(max(latencies), 3) if latencies else None,
        "rtf_max": round(max(rtfs), 4) if rtfs else None,
    }
    if not rates:
        summary["decode_chars_per_audio_s_mean"] = None
        summary["decode_chars_per_audio_s_max"] = None
        summary["runaway_requests"] = None
        summary["runaway_scored_requests"] = 0
        return summary

    summary["decode_chars_per_audio_s_mean"] = round(sum(rates) / len(rates), 2)
    summary["decode_chars_per_audio_s_max"] = round(max(rates), 2)
    summary["runaway_requests"] = sum(
        1 for rate in rates if rate > chars_per_audio_s_threshold
    )
    summary["runaway_scored_requests"] = len(rates)
    return summary
