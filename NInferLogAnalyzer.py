#!/usr/bin/env python3
"""NInferLogAnalyzer: deterministic performance analyzer for NInfer server (ninfer-serve) logs.

Reads the human-readable server log (stderr) and/or the structured request log written with
--request-log-jsonl, and reports request throughput, time to first token, prefill/decode speed,
prefix-cache reuse, speculative decoding (MTP) statistics, thinking-budget use, context fill and
startup memory, with fixed heuristic recommendations.

Standard library only, so it runs on a stock Python 3.10+ installation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


SCHEMA_VERSION = "1.0"
TOOL_NAME = "NInferLogAnalyzer"

DEFAULT_THRESHOLDS: Dict[str, Any] = {
    # Prompt-size bands for the context table (tokens; upper bound exclusive, null = open).
    "context_bands": {
        "0-4K": [0, 4096],
        "4K-16K": [4096, 16384],
        "16K-32K": [16384, 32768],
        "32K-64K": [32768, 65536],
        "64K-96K": [65536, 98304],
        "96K+": [98304, None],
    },
    # A request counts as a cold prefill sample only if it computed at least this many tokens.
    "cold_prefill_min_tokens": 2048,
    # Full resets that recompute fewer tokens than this are cheap (e.g. client title requests).
    "costly_reset_min_tokens": 2048,
    "slack_warn_mib": 128.0,
    "costly_reset_rate_warn": 0.10,
    "costly_reset_ttft_warn_s": 120.0,
    "budget_hit_rate_warn": 0.30,
    "context_fill_warn": 0.85,
    "mtp_acceptance_low": 0.40,
    "warm_ttft_slow_s": 5.0,
    "long_context_decode_drop_warn": 0.30,
    # A request with no tool history and at most this many messages is a new conversation or a
    # client side request (title, summary, sub-agent); its full prefill is expected, not a cache loss.
    "fresh_request_max_messages": 3,
    # Decode below this fraction of the median for the same prompt-size band is an outlier
    # (GPU contention or memory pushed to system RAM); only requests generating enough tokens count.
    "decode_outlier_fraction": 0.5,
    "decode_outlier_min_tokens": 64,
}

RESET_CAUSES = {
    "new_conversation": "new conversation or client side request (expected full prefill)",
    "after_side_request": "evicted by a client side request (title, summary or sub-agent) just before",
    "interleaved": "another conversation ran in between and took the cache slot",
    "history_rewritten": "client changed earlier history (compaction, moved reminder, tool_choice change)",
}

REUSE_SHORT = {
    "full_reset": "reset",
    "append_frontier": "append",
    "restore_response_checkpoint": "resp-ckpt",
    "restore_turn_checkpoint": "turn-ckpt",
}


class UsageError(Exception):
    pass


# --------------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------------


@dataclass
class Request:
    request_id: int
    session: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    protocol: Optional[str] = None
    stream: Optional[bool] = None
    messages: Optional[int] = None
    max_tokens: Optional[int] = None
    tools: Optional[int] = None
    tool_choice: Optional[str] = None
    tool_history: Optional[bool] = None
    thinking: Optional[bool] = None
    reasoning_budget: Optional[int] = None
    preserve_thinking: Optional[bool] = None
    temperature: Optional[float] = None
    finish: Optional[str] = None
    tool_calls: Optional[int] = None
    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    computed_tokens: Optional[int] = None
    reuse: Optional[str] = None
    ttft_s: Optional[float] = None
    prefill_s: Optional[float] = None
    decode_s: Optional[float] = None
    wall_s: Optional[float] = None
    prefill_tps: Optional[float] = None
    decode_tps: Optional[float] = None
    spec_backend: Optional[str] = None
    spec_tokens_per_round: Optional[float] = None
    spec_acceptance: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    reset_cause: Optional[str] = None
    decode_outlier: bool = False

    @property
    def done(self) -> bool:
        return self.finish is not None


@dataclass
class Throughput:
    time: Optional[float]
    interval_s: float
    prefill_tps: float
    decode_tps: float
    running: int = 0
    prefilling: int = 0
    decode_ready: int = 0
    waiting: int = 0


@dataclass
class Session:
    source: str
    kind: str = "text"
    launch_stamp: Optional[str] = None
    requested_context: Optional[int] = None
    started: bool = False
    startup_error: Optional[str] = None
    model_id: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    load_seconds: Optional[float] = None
    weights_gib: Optional[float] = None
    kv_mode: Optional[str] = None
    context: Optional[int] = None
    kv_pages: Optional[int] = None
    kv_dtype: Optional[str] = None
    runtime_mib: Optional[float] = None
    free_after_weights_mib: Optional[float] = None
    free_after_startup_mib: Optional[float] = None
    headroom_mib: Optional[float] = None
    slack_mib: Optional[float] = None
    graph_observed_mib: Optional[float] = None
    graph_allowance_mib: Optional[float] = None
    cuda_graph: Optional[bool] = None
    speculative: Optional[str] = None
    prefill_chunk: Optional[int] = None
    gpu_name: Optional[str] = None
    first_time: Optional[float] = None
    last_time: Optional[float] = None
    requests: Dict[int, Request] = field(default_factory=dict)
    throughput: List[Throughput] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def touch(self, t: Optional[float]) -> None:
        if t is None:
            return
        if self.first_time is None or t < self.first_time:
            self.first_time = t
        if self.last_time is None or t > self.last_time:
            self.last_time = t

    def request(self, request_id: int) -> Request:
        if request_id not in self.requests:
            self.requests[request_id] = Request(request_id=request_id, session=self.source)
        return self.requests[request_id]


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


def to_mib(value: float, unit: str) -> float:
    unit = unit.strip()
    factor = {"B": 1 / (1024 * 1024), "KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0, "TiB": 1024.0 * 1024}.get(unit)
    if factor is None:
        raise ValueError(f"unknown unit {unit!r}")
    return value * factor


def parse_duration_s(text: str) -> Optional[float]:
    m = re.fullmatch(r"([\d.]+)(ms|s)", text)
    if not m:
        return None
    value = float(m.group(1))
    return value / 1000.0 if m.group(2) == "ms" else value


def parse_rate(text: str) -> Optional[float]:
    m = re.fullmatch(r"([\d.]+)tok/s", text)
    return float(m.group(1)) if m else None


def parse_int(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    m = re.match(r"-?\d+", text)
    return int(m.group(0)) if m else None


def parse_bool(text: Optional[str]) -> Optional[bool]:
    if text is None:
        return None
    lowered = text.lower()
    if lowered in ("on", "yes", "true", "1"):
        return True
    if lowered in ("off", "no", "false", "0"):
        return False
    return None


def percentile(values: List[float], fraction: float) -> Optional[float]:
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    position = (len(data) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return data[lower]
    return data[lower] + (data[upper] - data[lower]) * (position - lower)


def median(values: List[float]) -> Optional[float]:
    return percentile(values, 0.5)


def mean(values: List[float]) -> Optional[float]:
    data = [v for v in values if v is not None]
    return sum(data) / len(data) if data else None


def ratio(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_thresholds(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return dict(DEFAULT_THRESHOLDS)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            overrides = json.load(handle)
    except (OSError, ValueError) as exc:
        raise UsageError(f"Unable to read thresholds file {path!r}: {exc}") from exc
    if not isinstance(overrides, dict):
        raise UsageError("Thresholds file must contain a JSON object")
    unknown = sorted(set(overrides) - set(DEFAULT_THRESHOLDS))
    if unknown:
        raise UsageError(f"Unknown threshold keys: {', '.join(unknown)}")
    return deep_merge(DEFAULT_THRESHOLDS, overrides)


# --------------------------------------------------------------------------------------------
# Text log parsing (ninfer-serve stderr)
# --------------------------------------------------------------------------------------------


class Regexes:
    line = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\] \[(\w+)\] ninfer-serve: ?(.*)$")
    request = re.compile(r"^\[req (\d+)\] (.*)$")
    key_value = re.compile(r"([\w-]+)=([^\s\[\]()]+)")
    speculative = re.compile(r"speculative=(\w+) ([\d.]+)tok/round \(([\d.]+)%\)")
    throughput = re.compile(
        r"^throughput interval=([\d.]+)s prefill=([\d.]+)tok/s decode=([\d.]+)tok/s running=(\d+) "
        r"prefilling=(\d+) decode_ready=(\d+) waiting=(\d+)"
    )
    kv_capacity = re.compile(r"^KV capacity (\w+) resolved=(\d+) tokens pages=(\d+)/(\d+)")
    sized = re.compile(r"([\w-]+)=([\d.]+) ?(B|KiB|MiB|GiB)(?:/([\d.]+) ?(B|KiB|MiB|GiB))?")
    loaded = re.compile(r"^model loaded in ([\d.]+) s")
    weights = re.compile(r"^load\s+weights\s+100\.00%\s+([\d.]+) (B|KiB|MiB|GiB)\s*/")
    listening = re.compile(r"^listening on https?://([^:/\s]+):(\d+) \(model id: ([^,)]+)")
    file_name = re.compile(r"ninfer-serve-(\d{8}-\d{6})-(\d+)")


def parse_timestamp(text: str) -> Optional[float]:
    for pattern in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, pattern).timestamp()
        except ValueError:
            continue
    return None


def apply_request_start(req: Request, rest: str) -> None:
    words = rest.split()
    if words:
        req.protocol = words[0]
    req.stream = "stream" in words[1:3]
    values = dict(Regexes.key_value.findall(rest))
    req.messages = parse_int(values.get("msgs"))
    req.max_tokens = parse_int(values.get("max_tokens"))
    req.tools = parse_int(values.get("tools"))
    req.tool_choice = values.get("tool_choice")
    req.tool_history = parse_bool(values.get("tool_history"))
    req.thinking = parse_bool(values.get("thinking"))
    req.reasoning_budget = parse_int(values.get("reasoning_budget"))
    req.preserve_thinking = parse_bool(values.get("preserve_thinking"))
    if "temp" in values:
        try:
            req.temperature = float(values["temp"])
        except ValueError:
            pass


def apply_request_done(req: Request, rest: str) -> None:
    values = dict(Regexes.key_value.findall(rest))
    req.finish = values.get("finish")
    req.tool_calls = parse_int(values.get("tool_calls"))
    req.prompt_tokens = parse_int(values.get("prompt"))
    req.generated_tokens = parse_int(values.get("gen"))
    req.reasoning_tokens = parse_int(values.get("reasoning"))
    req.cached_tokens = parse_int(values.get("cache"))
    req.reuse = values.get("reuse")
    req.ttft_s = parse_duration_s(values.get("ttft", ""))
    req.wall_s = parse_duration_s(values.get("wall", ""))
    req.prefill_tps = parse_rate(values.get("prefill", ""))
    req.decode_tps = parse_rate(values.get("decode", ""))
    if req.prompt_tokens is not None and req.cached_tokens is not None:
        req.computed_tokens = max(0, req.prompt_tokens - req.cached_tokens)
    spec = Regexes.speculative.search(rest)
    if spec:
        req.spec_backend = spec.group(1)
        req.spec_tokens_per_round = float(spec.group(2))
        req.spec_acceptance = float(spec.group(3)) / 100.0
    elif "speculative=" in rest:
        req.spec_backend = values.get("speculative")


def apply_kv_capacity(session: Session, message: str) -> None:
    m = Regexes.kv_capacity.match(message)
    if not m:
        return
    session.kv_mode = m.group(1)
    session.context = int(m.group(2))
    session.kv_pages = int(m.group(3))
    for key, value, unit, second, second_unit in Regexes.sized.findall(message):
        mib = to_mib(float(value), unit)
        if key == "runtime":
            session.runtime_mib = mib
        elif key == "free-after-weights":
            session.free_after_weights_mib = mib
        elif key == "free-after-startup":
            session.free_after_startup_mib = mib
        elif key == "headroom":
            session.headroom_mib = mib
        elif key == "slack":
            session.slack_mib = mib
        elif key == "graphs":
            session.graph_observed_mib = mib
            if second:
                session.graph_allowance_mib = to_mib(float(second), second_unit)
                session.cuda_graph = session.graph_allowance_mib > 0


def parse_text_log(text: str, source: str) -> Session:
    session = Session(source=source, kind="text")
    name_match = Regexes.file_name.search(os.path.basename(source))
    if name_match:
        session.launch_stamp = name_match.group(1)
        session.requested_context = int(name_match.group(2))
    for raw in text.splitlines():
        m = Regexes.line.match(raw.strip("\r"))
        if not m:
            continue
        stamp, level, message = m.groups()
        t = parse_timestamp(stamp)
        session.touch(t)
        req_match = Regexes.request.match(message)
        if req_match:
            req = session.request(int(req_match.group(1)))
            rest = req_match.group(2)
            if rest.startswith("done "):
                apply_request_done(req, rest)
                req.end_time = t
            elif rest.startswith("error"):
                req.errors.append(rest[len("error"):].strip() or "error")
                if req.end_time is None:
                    req.end_time = t
            else:
                apply_request_start(req, rest)
                req.start_time = t
            continue
        tp = Regexes.throughput.match(message)
        if tp:
            session.throughput.append(Throughput(
                time=t, interval_s=float(tp.group(1)), prefill_tps=float(tp.group(2)),
                decode_tps=float(tp.group(3)), running=int(tp.group(4)), prefilling=int(tp.group(5)),
                decode_ready=int(tp.group(6)), waiting=int(tp.group(7))))
            continue
        if message.startswith("KV capacity"):
            apply_kv_capacity(session, message)
            continue
        loaded = Regexes.loaded.match(message)
        if loaded:
            session.load_seconds = float(loaded.group(1))
            continue
        weights = Regexes.weights.match(message)
        if weights:
            session.weights_gib = to_mib(float(weights.group(1)), weights.group(2)) / 1024.0
            continue
        listening = Regexes.listening.match(message)
        if listening:
            session.started = True
            session.host = listening.group(1)
            session.port = int(listening.group(2))
            session.model_id = listening.group(3).strip()
            continue
        if level.lower() == "error" and not session.started and session.startup_error is None:
            session.startup_error = message.strip()
    if not session.started and session.startup_error is None and not session.requests:
        session.warnings.append(f"{os.path.basename(source)}: server never reported 'listening' (still starting, or truncated log)")
    return session


# --------------------------------------------------------------------------------------------
# JSONL request-log parsing (ninfer-serve --request-log-jsonl)
# --------------------------------------------------------------------------------------------


def dig(record: Dict[str, Any], *path: str) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def bytes_to_mib(value: Any) -> Optional[float]:
    return value / (1024 * 1024) if isinstance(value, (int, float)) else None


def apply_json_request_fields(req: Request, info: Dict[str, Any]) -> None:
    req.protocol = info.get("protocol", req.protocol)
    req.stream = info.get("stream", req.stream)
    req.messages = info.get("message_count", req.messages)
    req.max_tokens = info.get("requested_output_tokens", req.max_tokens)
    req.tools = info.get("tool_count", req.tools)
    req.tool_choice = info.get("tool_choice", req.tool_choice)
    req.tool_history = info.get("has_tool_history", req.tool_history)
    req.thinking = info.get("enable_thinking", req.thinking)
    req.reasoning_budget = info.get("reasoning_budget", req.reasoning_budget)
    req.preserve_thinking = info.get("preserve_thinking", req.preserve_thinking)
    temperature = dig(info, "sampling", "temperature")
    if isinstance(temperature, (int, float)):
        req.temperature = float(temperature)


def parse_jsonl_log(text: str, source: str) -> List[Session]:
    sessions: List[Session] = []
    current: Optional[Session] = None
    bad_lines = 0

    def new_session(label: str) -> Session:
        session = Session(source=label, kind="jsonl")
        name_match = Regexes.file_name.search(os.path.basename(source))
        if name_match:
            session.launch_stamp = name_match.group(1)
            session.requested_context = int(name_match.group(2))
        sessions.append(session)
        return session

    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except ValueError:
            bad_lines += 1
            continue
        if not isinstance(record, dict):
            continue
        event = record.get("event")
        t_ms = record.get("timestamp_unix_ms")
        t = t_ms / 1000.0 if isinstance(t_ms, (int, float)) else None
        if event == "server_start" or current is None:
            suffix = f"#{len(sessions) + 1}" if sessions else ""
            current = new_session(source + suffix)
        if event == "server_start":
            current.started = True
            engine = record.get("engine") or {}
            memory = record.get("memory") or {}
            current.kv_mode = engine.get("kv_capacity_mode")
            current.context = engine.get("kv_capacity") or engine.get("max_context")
            current.kv_pages = engine.get("kv_capacity_page_groups")
            current.kv_dtype = engine.get("kv_cache")
            current.cuda_graph = engine.get("cuda_graph")
            current.prefill_chunk = engine.get("prefill_chunk")
            backend = engine.get("speculative_backend")
            if backend and backend != "none":
                current.speculative = f"{backend}-{engine.get('speculative_draft_window')}"
            current.free_after_weights_mib = bytes_to_mib(memory.get("available_after_weights_bytes"))
            current.free_after_startup_mib = bytes_to_mib(memory.get("available_after_startup_bytes"))
            current.runtime_mib = bytes_to_mib(memory.get("runtime_reservation_bytes"))
            current.headroom_mib = bytes_to_mib(memory.get("kv_capacity_headroom_bytes"))
            current.slack_mib = bytes_to_mib(memory.get("planned_slack_bytes"))
            current.graph_observed_mib = bytes_to_mib(memory.get("cuda_graph_observed_bytes"))
            current.graph_allowance_mib = bytes_to_mib(memory.get("cuda_graph_allowance_bytes"))
            current.load_seconds = dig(record, "artifact", "load_seconds")
            weights = dig(record, "memory", "weights", "used_bytes")
            current.weights_gib = weights / (1024 ** 3) if isinstance(weights, (int, float)) else None
            current.gpu_name = dig(record, "environment", "gpu_name")
            current.touch(t)
            continue
        current.touch(t)
        if event in ("request_start", "request_done"):
            info = record.get("request") or {}
            request_id = info.get("request_id")
            if not isinstance(request_id, int):
                continue
            req = current.request(request_id)
            apply_json_request_fields(req, info)
            if current.model_id is None and info.get("model"):
                current.model_id = info.get("model")
            if event == "request_start":
                req.start_time = t
                continue
            result = record.get("result") or {}
            timings = record.get("timings_seconds") or {}
            spec = record.get("speculative") or {}
            req.end_time = t
            req.finish = result.get("finish_reason")
            req.tool_calls = result.get("tool_call_count")
            # The JSONL log reports tool-call turns as stop_token with a tool-call count; the text log
            # says finish=tool_calls. Normalize to the text-log wording.
            if req.finish == "stop_token" and req.tool_calls:
                req.finish = "tool_calls"
            req.prompt_tokens = result.get("prompt_tokens")
            req.generated_tokens = result.get("completion_tokens")
            req.reasoning_tokens = result.get("reasoning_tokens")
            req.cached_tokens = result.get("prefix_cache_hit_tokens")
            req.computed_tokens = result.get("computed_prefill_tokens")
            req.reuse = result.get("prefix_reuse_path")
            req.ttft_s = timings.get("ttft")
            req.prefill_s = timings.get("prefill")
            req.decode_s = timings.get("decode")
            req.wall_s = timings.get("total")
            if req.computed_tokens and req.prefill_s:
                req.prefill_tps = req.computed_tokens / req.prefill_s
            if req.generated_tokens and req.decode_s:
                req.decode_tps = req.generated_tokens / req.decode_s
            backend = spec.get("backend")
            if backend and backend != "none":
                req.spec_backend = backend
                drafted = spec.get("drafted_tokens") or 0
                accepted = spec.get("accepted_tokens") or 0
                rounds = spec.get("rounds") or 0
                if drafted:
                    req.spec_acceptance = accepted / drafted
                if rounds:
                    req.spec_tokens_per_round = (accepted + rounds) / rounds
        elif event == "throughput":
            rates = record.get("throughput_tokens_per_second") or {}
            scheduler = record.get("scheduler") or {}
            current.throughput.append(Throughput(
                time=t, interval_s=float(record.get("interval_seconds") or 0.0),
                prefill_tps=float(rates.get("prefill") or 0.0), decode_tps=float(rates.get("decode") or 0.0),
                running=int(scheduler.get("running") or 0), prefilling=int(scheduler.get("prefilling") or 0),
                decode_ready=int(scheduler.get("decode_ready") or 0), waiting=int(scheduler.get("waiting") or 0)))
    if bad_lines:
        target = sessions[-1] if sessions else None
        message = f"{os.path.basename(source)}: skipped {bad_lines} unreadable JSONL line(s)"
        if target:
            target.warnings.append(message)
    return sessions


def looks_like_jsonl(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped.startswith("{") and "ninfer_serve_request_log" in stripped
    return False


def parse_text(text: str, source: str) -> List[Session]:
    if looks_like_jsonl(text):
        return parse_jsonl_log(text, source)
    return [parse_text_log(text, source)]


# --------------------------------------------------------------------------------------------
# Input selection
# --------------------------------------------------------------------------------------------


def read_file(path: str, encoding: str) -> str:
    try:
        with open(path, "r", encoding=encoding, errors="replace") as handle:
            return handle.read()
    except OSError as exc:
        raise UsageError(f"Unable to read input file {path!r}: {exc}") from exc


def read_clipboard(encoding: str) -> str:
    try:
        import tkinter  # type: ignore

        root = tkinter.Tk()
        root.withdraw()
        try:
            text = root.clipboard_get()
        finally:
            root.destroy()
        if text:
            return text
    except Exception:
        pass
    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        except OSError as exc:
            raise UsageError(f"Clipboard unavailable: unable to launch PowerShell fallback: {exc}") from exc
        if completed.returncode == 0:
            text = completed.stdout.decode(encoding, errors="replace")
            if text:
                return text
    raise UsageError("Clipboard unavailable or empty")


def log_group_key(path: str) -> str:
    name = os.path.basename(path)
    for suffix in (".requests.jsonl", ".jsonl", ".log"):
        if name.endswith(suffix):
            return os.path.join(os.path.dirname(path), name[: -len(suffix)])
    return path


def discover_directory(directory: str, encoding: str) -> Tuple[List[Session], List[Session]]:
    """Return (started sessions, failed startup attempts) for every server log in a directory.

    When both a text log and a JSONL request log exist for the same launch, the JSONL log is used
    for request data (full precision) and the text log fills in startup fields it lacks.
    """
    try:
        names = sorted(os.listdir(directory))
    except OSError as exc:
        raise UsageError(f"Unable to list directory {directory!r}: {exc}") from exc
    groups: Dict[str, Dict[str, str]] = {}
    for name in names:
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        if name.endswith(".log"):
            groups.setdefault(log_group_key(path), {})["text"] = path
        elif name.endswith(".jsonl"):
            groups.setdefault(log_group_key(path), {})["jsonl"] = path
    started: List[Session] = []
    failed: List[Session] = []
    for key in sorted(groups):
        entry = groups[key]
        text_session = parse_text_log(read_file(entry["text"], encoding), entry["text"]) if "text" in entry else None
        sessions: List[Session] = []
        if "jsonl" in entry:
            sessions = parse_jsonl_log(read_file(entry["jsonl"], encoding), entry["jsonl"])
            if text_session is not None:
                for session in sessions:
                    merge_startup_fields(session, text_session)
            if not sessions and text_session is not None:
                sessions = [text_session]
        elif text_session is not None:
            sessions = [text_session]
        for session in sessions:
            (started if session.started or session.requests else failed).append(session)
    started.sort(key=lambda s: (s.first_time or 0.0, s.source))
    failed.sort(key=lambda s: (s.first_time or 0.0, s.source))
    return started, failed


def merge_startup_fields(target: Session, text_session: Session) -> None:
    for name in ("model_id", "host", "port", "load_seconds", "weights_gib", "kv_mode", "context", "kv_pages",
                 "runtime_mib", "free_after_weights_mib", "free_after_startup_mib", "headroom_mib", "slack_mib",
                 "graph_observed_mib", "graph_allowance_mib", "cuda_graph", "launch_stamp", "requested_context"):
        if getattr(target, name) is None and getattr(text_session, name) is not None:
            setattr(target, name, getattr(text_session, name))


def load_sessions(args: argparse.Namespace, inputs: List[str]) -> Tuple[List[Session], List[Session], List[str]]:
    """Resolve inputs into (sessions to analyze, failed startup attempts, source labels)."""
    selected: List[Session] = []
    attempts: List[Session] = []
    labels: List[str] = []
    if args.clipboard:
        text = read_clipboard(args.encoding)
        selected.extend(parse_text(text, "clipboard"))
        labels.append("clipboard")
    for item in inputs:
        if os.path.isdir(item):
            started, failed = discover_directory(item, args.encoding)
            if not started:
                raise UsageError(f"No server sessions with a started server found in {item!r}")
            chosen = started if args.all else [started[-1]]
            selected.extend(chosen)
            stamps = {s.launch_stamp for s in chosen if s.launch_stamp}
            attempts.extend(f for f in failed if f.launch_stamp in stamps)
            labels.append(item)
        else:
            text = read_file(item, args.encoding)
            if not text.strip():
                raise UsageError(f"Input file {item!r} is empty")
            parsed = parse_text(text, item)
            selected.extend(p for p in parsed if p.started or p.requests)
            attempts.extend(p for p in parsed if not (p.started or p.requests))
            labels.append(item)
    if not inputs and not args.clipboard:
        if args.stdin or not sys.stdin.isatty():
            text = sys.stdin.read()
            if not text.strip():
                raise UsageError("Input is empty")
            parsed = parse_text(text, "stdin")
            selected.extend(p for p in parsed if p.started or p.requests)
            attempts.extend(p for p in parsed if not (p.started or p.requests))
            labels.append("stdin")
        else:
            raise UsageError("No input provided")
    if not selected and attempts:
        # A lone failed startup log is still worth reporting.
        selected, attempts = attempts, []
    return selected, attempts, labels


# --------------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------------


def band_for(tokens: Optional[int], bands: Dict[str, Any]) -> Optional[str]:
    if tokens is None:
        return None
    for name, bounds in bands.items():
        low, high = bounds[0], bounds[1]
        if tokens >= low and (high is None or tokens < high):
            return name
    return None


def is_fresh(r: Request, thresholds: Dict[str, Any]) -> bool:
    """A request with no prior conversation: new chat, or a client side request."""
    return (not r.tool_history) and (r.messages is not None and r.messages <= thresholds["fresh_request_max_messages"])


def classify_resets(done_by_session: Dict[str, List[Request]], thresholds: Dict[str, Any]) -> None:
    """Label every full-reset request (except each session's first) with a likely cause.

    The server keeps one resident prefix at max-concurrency 1, so any other request in between
    replaces it. Requests are compared with the request that finished just before them.
    """
    for requests in done_by_session.values():
        ordered = sorted(requests, key=lambda r: (r.end_time or 0.0, r.request_id))
        for index, r in enumerate(ordered):
            if index == 0 or r.reuse != "full_reset":
                continue
            previous = ordered[index - 1]
            if is_fresh(r, thresholds):
                r.reset_cause = "new_conversation"
            elif is_fresh(previous, thresholds):
                r.reset_cause = "after_side_request"
            elif (previous.tools != r.tools
                  or (previous.messages is not None and r.messages is not None and previous.messages > r.messages)):
                r.reset_cause = "interleaved"
            else:
                r.reset_cause = "history_rewritten"


def mark_decode_outliers(done: List[Request], thresholds: Dict[str, Any]) -> None:
    bands = thresholds["context_bands"]
    medians: Dict[str, Optional[float]] = {}
    for name in bands:
        medians[name] = median([r.decode_tps for r in done
                                if r.decode_tps and band_for(r.prompt_tokens, bands) == name])
    for r in done:
        band_median = medians.get(band_for(r.prompt_tokens, bands) or "")
        if (r.decode_tps and band_median and (r.generated_tokens or 0) >= thresholds["decode_outlier_min_tokens"]
                and r.decode_tps < thresholds["decode_outlier_fraction"] * band_median):
            r.decode_outlier = True


def token_weighted_rate(requests: List[Request]) -> Optional[float]:
    tokens = 0.0
    seconds = 0.0
    for r in requests:
        if r.generated_tokens and r.decode_tps:
            tokens += r.generated_tokens
            seconds += r.generated_tokens / r.decode_tps
    return tokens / seconds if seconds else None


def stat_block(values: List[float]) -> Dict[str, Optional[float]]:
    data = [v for v in values if v is not None]
    return {
        "samples": len(data),
        "median": median(data),
        "mean": mean(data),
        "p10": percentile(data, 0.10),
        "p90": percentile(data, 0.90),
        "p95": percentile(data, 0.95),
        "min": min(data) if data else None,
        "max": max(data) if data else None,
    }


def session_info(session: Session) -> Dict[str, Any]:
    duration = (session.last_time - session.first_time) if session.first_time and session.last_time else None
    return {
        "source": session.source,
        "kind": session.kind,
        "started": session.started,
        "startup_error": session.startup_error,
        "model_id": session.model_id,
        "endpoint": f"http://{session.host}:{session.port}/v1" if session.host and session.port else None,
        "context": session.context or session.requested_context,
        "requested_context": session.requested_context,
        "kv_mode": session.kv_mode,
        "kv_pages": session.kv_pages,
        "kv_dtype": session.kv_dtype,
        "load_seconds": session.load_seconds,
        "weights_gib": session.weights_gib,
        "runtime_mib": session.runtime_mib,
        "free_after_weights_mib": session.free_after_weights_mib,
        "free_after_startup_mib": session.free_after_startup_mib,
        "slack_mib": session.slack_mib,
        "graph_observed_mib": session.graph_observed_mib,
        "graph_allowance_mib": session.graph_allowance_mib,
        "cuda_graph": session.cuda_graph,
        "speculative": session.speculative,
        "gpu_name": session.gpu_name,
        "first_time": session.first_time,
        "last_time": session.last_time,
        "duration_s": duration,
        "request_count": len(session.requests),
    }


def analyze(sessions: List[Session], attempts: List[Session], thresholds: Dict[str, Any],
            last: Optional[int] = None) -> Dict[str, Any]:
    warnings: List[str] = []
    for session in sessions + attempts:
        warnings.extend(session.warnings)

    all_requests: List[Request] = []
    first_done_ids = set()
    for session in sessions:
        ordered = sorted(session.requests.values(), key=lambda r: (r.start_time or r.end_time or 0.0, r.request_id))
        done_in_order = [r for r in ordered if r.done]
        if done_in_order:
            first_done_ids.add((session.source, done_in_order[0].request_id))
        all_requests.extend(ordered)
    if last:
        all_requests = all_requests[-last:]

    done = [r for r in all_requests if r.done]
    in_progress = [r for r in all_requests if not r.done and not r.errors]

    # Request ids restart with every server start; prefix them with the session number when several
    # sessions are combined so ids in the report stay unambiguous.
    session_number = {s.source: index + 1 for index, s in enumerate(sessions)}

    def label(r: Request) -> Any:
        return f"{session_number.get(r.session, 0)}:{r.request_id}" if len(sessions) > 1 else r.request_id
    errored = [r for r in all_requests if r.errors]

    finish_counts: Dict[str, int] = {}
    for r in done:
        finish_counts[r.finish or "unknown"] = finish_counts.get(r.finish or "unknown", 0) + 1
    reuse_counts: Dict[str, int] = {}
    for r in done:
        reuse_counts[r.reuse or "unknown"] = reuse_counts.get(r.reuse or "unknown", 0) + 1

    prompt_total = sum(r.prompt_tokens or 0 for r in done)
    cached_total = sum(r.cached_tokens or 0 for r in done)
    computed_total = sum((r.computed_tokens if r.computed_tokens is not None else (r.prompt_tokens or 0)) for r in done)
    generated_total = sum(r.generated_tokens or 0 for r in done)
    reasoning_total = sum(r.reasoning_tokens or 0 for r in done)

    by_session: Dict[str, List[Request]] = {}
    for r in done:
        by_session.setdefault(r.session, []).append(r)
    classify_resets(by_session, thresholds)
    mark_decode_outliers(done, thresholds)
    for session in sessions:
        if session.speculative is None:
            backends = {r.spec_backend for r in session.requests.values() if r.spec_backend}
            if backends:
                session.speculative = "/".join(sorted(backends))

    costly_min = thresholds["costly_reset_min_tokens"]
    resets = [r for r in done if r.reuse == "full_reset" and (r.session, r.request_id) not in first_done_ids]
    cause_counts: Dict[str, int] = {}
    cause_ttft: Dict[str, float] = {}
    for r in resets:
        cause = r.reset_cause or "history_rewritten"
        cause_counts[cause] = cause_counts.get(cause, 0) + 1
        cause_ttft[cause] = cause_ttft.get(cause, 0.0) + (r.ttft_s or 0.0)
    # A new conversation's first prefill is unavoidable, so only lost history counts as costly.
    costly_resets = [r for r in resets if r.reset_cause != "new_conversation"
                     and (r.computed_tokens or r.prompt_tokens or 0) >= costly_min]
    outliers = [r for r in done if r.decode_outlier]

    warm = [r for r in done if r.reuse and r.reuse != "full_reset"]
    cold = [r for r in done if r.reuse == "full_reset"]
    cold_prefill = [r for r in done if (r.computed_tokens or 0) >= thresholds["cold_prefill_min_tokens"] and r.prefill_tps]

    thinking = [r for r in done if r.thinking]
    budget_hits = [r for r in thinking if r.reasoning_budget and r.reasoning_tokens is not None
                   and r.reasoning_tokens >= r.reasoning_budget]

    spec = [r for r in done if r.spec_acceptance is not None and r.generated_tokens]
    spec_weight = sum(r.generated_tokens or 0 for r in spec)
    weighted_acceptance = (sum((r.spec_acceptance or 0.0) * (r.generated_tokens or 0) for r in spec) / spec_weight
                           if spec_weight else None)

    # Each request is measured against the context of the server session that handled it.
    session_capacity = {s.source: (s.context or s.requested_context) for s in sessions}
    fills = [((r.prompt_tokens or 0) / session_capacity[r.session], r) for r in done
             if session_capacity.get(r.session) and r.prompt_tokens]
    fullest = max(fills, key=lambda item: item[0]) if fills else None
    fill = fullest[0] if fullest else None
    capacity = session_capacity.get(fullest[1].session) if fullest else next(
        (c for c in session_capacity.values() if c), None)
    max_prompt = fullest[1].prompt_tokens if fullest else max((r.prompt_tokens or 0 for r in done), default=0)
    near_full = [r for f, r in fills if f >= thresholds["context_fill_warn"]]

    throughput = [tp for s in sessions for tp in s.throughput]
    busy_s = sum(tp.interval_s for tp in throughput if tp.prefill_tps > 0 or tp.decode_tps > 0)
    max_waiting = max((tp.waiting for tp in throughput), default=0)
    queued_intervals = sum(1 for tp in throughput if tp.waiting > 0)

    durations = [(s.last_time - s.first_time) for s in sessions if s.first_time and s.last_time]
    session_s = sum(durations) if durations else None
    request_wall_s = sum(r.wall_s or 0.0 for r in done)

    decode_values = [r.decode_tps for r in done if r.decode_tps]
    summary = {
        "requests_seen": len(all_requests),
        "requests_done": len(done),
        "requests_in_progress": len(in_progress),
        "requests_with_errors": len(errored),
        "client_disconnects": sum(1 for r in errored if any("disconnect" in e for e in r.errors)),
        "admission_timeouts": sum(1 for r in errored if any("expired while waiting" in e for e in r.errors)),
        "error_counts": error_counts(errored),
        "finish_counts": finish_counts,
        "tool_calls": sum(r.tool_calls or 0 for r in done),
        "tokens": {
            "prompt": prompt_total,
            "cached": cached_total,
            "computed": computed_total,
            "generated": generated_total,
            "reasoning": reasoning_total,
        },
        "cache_reuse_rate": ratio(cached_total, prompt_total),
        "reuse_paths": reuse_counts,
        "full_resets_after_first": len(resets),
        "costly_resets": len(costly_resets),
        "costly_reset_ids": [label(r) for r in costly_resets],
        "costly_reset_ttft_s": sum(r.ttft_s or 0.0 for r in costly_resets),
        "costly_reset_rate": ratio(len(costly_resets), max(len(done) - len(first_done_ids), 0)),
        "reset_causes": cause_counts,
        "reset_cause_ttft_s": cause_ttft,
        "reset_cause_ids": {cause: [label(r) for r in resets if (r.reset_cause or "history_rewritten") == cause]
                            for cause in cause_counts},
        "decode_outliers": len(outliers),
        "decode_outlier_ids": [label(r) for r in outliers],
        "decode_outlier_window": ([fmt_time(min(r.start_time or r.end_time or 0 for r in outliers)),
                                   fmt_time(max(r.end_time or 0 for r in outliers))] if outliers else None),
        "decode_tps": stat_block(decode_values),
        "decode_tps_token_weighted": token_weighted_rate(done),
        "ttft_s": stat_block([r.ttft_s for r in done if r.ttft_s is not None]),
        "ttft_warm_s": stat_block([r.ttft_s for r in warm if r.ttft_s is not None]),
        "ttft_cold_s": stat_block([r.ttft_s for r in cold if r.ttft_s is not None]),
        "prefill_cold_tps": stat_block([r.prefill_tps for r in cold_prefill]),
        "wall_s": stat_block([r.wall_s for r in done if r.wall_s is not None]),
        "generated_per_request": stat_block([float(r.generated_tokens) for r in done if r.generated_tokens is not None]),
        "prompt_tokens": stat_block([float(r.prompt_tokens) for r in done if r.prompt_tokens is not None]),
        "mtp_tokens_per_round": stat_block([r.spec_tokens_per_round for r in spec]),
        "mtp_acceptance_weighted": weighted_acceptance,
        "mtp_acceptance_median": median([r.spec_acceptance for r in spec]),
        "thinking_requests": len(thinking),
        "reasoning_share": ratio(reasoning_total, generated_total),
        "budget_hits": len(budget_hits),
        "budget_hit_rate": ratio(len(budget_hits), len(thinking)),
        "context_capacity": capacity,
        "max_prompt_tokens": max_prompt or None,
        "max_context_fill": fill,
        "near_full_requests": len(near_full),
        "session_seconds": session_s,
        "request_wall_seconds": request_wall_s,
        # The server only logs throughput for intervals with activity, so busy time is compared with
        # the whole session rather than with logged time.
        "busy_seconds": busy_s if throughput else None,
        "busy_fraction": (min(1.0, busy_s / session_s) if throughput and session_s else None),
        "max_waiting": max_waiting if throughput else None,
        "queued_intervals": queued_intervals if throughput else None,
        "generated_per_hour": (generated_total / (session_s / 3600.0)) if session_s else None,
        "requests_per_hour": (len(done) / (session_s / 3600.0)) if session_s else None,
    }

    bands: Dict[str, Dict[str, Any]] = {}
    for name in thresholds["context_bands"]:
        members = [r for r in done if band_for(r.prompt_tokens, thresholds["context_bands"]) == name]
        members_warm = [r for r in members if r.reuse and r.reuse != "full_reset"]
        members_cold = [r for r in members if r.reuse == "full_reset"]
        members_prefill = [r for r in members if (r.computed_tokens or 0) >= thresholds["cold_prefill_min_tokens"]
                           and r.prefill_tps]
        prompt_sum = sum(r.prompt_tokens or 0 for r in members)
        bands[name] = {
            "range": thresholds["context_bands"][name],
            "requests": len(members),
            "median_decode_tps": median([r.decode_tps for r in members if r.decode_tps]),
            "token_weighted_decode_tps": token_weighted_rate(members),
            "median_ttft_warm_s": median([r.ttft_s for r in members_warm if r.ttft_s is not None]),
            "median_ttft_cold_s": median([r.ttft_s for r in members_cold if r.ttft_s is not None]),
            "median_cold_prefill_tps": median([r.prefill_tps for r in members_prefill]),
            "median_mtp_acceptance": median([r.spec_acceptance for r in members if r.spec_acceptance is not None]),
            "cache_reuse_rate": ratio(sum(r.cached_tokens or 0 for r in members), prompt_sum),
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "sessions": [session_info(s) for s in sessions],
        "startup_attempts": [
            {"source": a.source, "requested_context": a.requested_context, "error": a.startup_error,
             "started": a.started} for a in attempts],
        "summary": summary,
        "context_bands": bands,
        "requests": [dict(request_dict(r), label=label(r)) for r in all_requests],
        "warnings": warnings,
        "recommendations": [],
        "thresholds": thresholds,
    }
    report["recommendations"] = recommendations(report, thresholds)
    return report


def error_counts(requests: List[Request]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in requests:
        for message in r.errors:
            counts[message] = counts.get(message, 0) + 1
    return counts


def request_dict(r: Request) -> Dict[str, Any]:
    data = asdict(r)
    data["done"] = r.done
    return data


def recommendations(report: Dict[str, Any], thresholds: Dict[str, Any]) -> List[Dict[str, str]]:
    s = report["summary"]
    recs: List[Dict[str, str]] = []

    def add(severity: str, code: str, message: str, evidence: str) -> None:
        recs.append({"severity": severity, "code": code, "message": message, "evidence": evidence})

    for session in report["sessions"]:
        slack = session.get("slack_mib")
        if slack is not None and slack < thresholds["slack_warn_mib"]:
            add("warn", "low_startup_slack",
                "Startup memory slack is thin; other GPU apps can push server memory into system RAM and slow it down.",
                f"{os.path.basename(session['source'])}: planned slack {slack:.0f} MiB < {thresholds['slack_warn_mib']:.0f} MiB")

    rate = s.get("costly_reset_rate")
    lost_s = s.get("costly_reset_ttft_s") or 0.0
    if s["costly_resets"] > 0 and ((rate is not None and rate > thresholds["costly_reset_rate_warn"])
                                   or lost_s > thresholds["costly_reset_ttft_warn_s"]):
        causes = s["reset_causes"]
        cost = s["reset_cause_ttft_s"]
        ids = s["reset_cause_ids"]

        def cause_evidence(cause: str) -> str:
            listed = ", ".join(str(i) for i in ids.get(cause, [])[:10])
            return f"{causes[cause]} request(s), {cost.get(cause, 0.0):.0f} s of TTFT; ids {listed}"

        if causes.get("after_side_request"):
            add("warn", "cache_evicted_by_side_requests",
                "One-off client requests (title generation, summaries, sub-agents) replaced the conversation in the "
                "server's single prefix-cache slot, so the next turn re-read the whole conversation.",
                cause_evidence("after_side_request"))
        if causes.get("interleaved"):
            add("warn", "cache_thrash_interleaved",
                "Several conversations used the server at the same time (parallel agents, sub-agents or two clients) and "
                "kept evicting each other from the single prefix-cache slot. Run them one at a time where possible.",
                cause_evidence("interleaved"))
        if causes.get("history_rewritten"):
            preserve = any(r.get("preserve_thinking") for r in report["requests"] if r.get("done"))
            if preserve:
                message = ("The client rewrote earlier history, so the cached prefix no longer matched (history compaction, "
                           "a plan-mode reminder moved to the latest message, or a tool_choice change).")
            else:
                message = ("Cached prefixes were lost on follow-up turns. Start the server with --preserve-thinking "
                           "(opencode and Pi echo reasoning_content, which keeps the history append-only).")
            add("warn", "cache_history_rewritten", message, cause_evidence("history_rewritten"))

    if s.get("decode_outliers"):
        window = s.get("decode_outlier_window") or ["?", "?"]
        add("warn", "decode_outliers",
            "Some requests decoded far slower than others of the same size. Typical causes: another app using the GPU "
            "(games, video, browsers, VMs) or Windows moving server memory to system RAM when VRAM ran short.",
            f"{s['decode_outliers']} request(s) below {thresholds['decode_outlier_fraction']:.0%} of the median for their "
            f"prompt size, {window[0]} to {window[1]}; ids "
            + ", ".join(str(i) for i in s["decode_outlier_ids"][:12]))

    budget_rate = s.get("budget_hit_rate")
    if budget_rate is not None and budget_rate > thresholds["budget_hit_rate_warn"]:
        add("info", "thinking_budget_hits",
            "Thinking often runs into the reasoning budget. Raise --default-thinking-budget if answers look cut short; "
            "keep it if bounding latency matters more.",
            f"{s['budget_hits']} of {s['thinking_requests']} thinking requests ({budget_rate:.0%}) used the full budget")

    fill = s.get("max_context_fill")
    if fill is not None and fill > thresholds["context_fill_warn"]:
        add("info", "context_nearly_full",
            "Prompts came close to the server's context limit. Expect client-side compaction; start the server with a "
            "larger context when VRAM allows.",
            f"max prompt {s['max_prompt_tokens']:,} of {s['context_capacity']:,} tokens ({fill:.0%}); "
            f"{s['near_full_requests']} request(s) above {thresholds['context_fill_warn']:.0%}")

    output_limit = sum(v for k, v in s["finish_counts"].items() if k in ("output_limit", "length", "max_tokens"))
    if output_limit:
        add("info", "output_limit_hits", "Some responses stopped at the client's max_tokens; raise the client output limit.",
            f"{output_limit} request(s) finished with output_limit")

    context_full = s["finish_counts"].get("context_capacity", 0)
    if context_full:
        add("warn", "context_capacity_hits",
            "Responses were cut off because the conversation filled the server's whole context. Compact or start a new "
            "session sooner, or start the server with a larger context when VRAM allows.",
            f"{context_full} request(s) finished with context_capacity (server context {fmt(s['context_capacity'])})")

    if s.get("admission_timeouts"):
        add("warn", "admission_timeouts",
            "Requests timed out waiting for their turn (the server runs one request at a time and drops requests that "
            "wait longer than --pending-timeout-ms). This happens when several agents or clients share the server.",
            f"{s['admission_timeouts']} request(s) expired while waiting for admission")

    if s.get("client_disconnects"):
        add("info", "client_disconnects", "Clients disconnected mid-request (user cancel or client timeout).",
            f"{s['client_disconnects']} disconnect(s), {s['finish_counts'].get('cancelled', 0)} cancelled")

    acceptance = s.get("mtp_acceptance_weighted")
    if acceptance is not None and acceptance < thresholds["mtp_acceptance_low"]:
        add("info", "low_mtp_acceptance",
            "MTP acceptance is low for this workload, so speculative decoding gains less. This is normal for long, "
            "varied prose or very long contexts; code and tool calls usually accept more.",
            f"token-weighted acceptance {acceptance:.0%}")

    warm_p95 = s["ttft_warm_s"].get("p95")
    if warm_p95 is not None and warm_p95 > thresholds["warm_ttft_slow_s"]:
        add("info", "slow_warm_ttft",
            "Some cache-hit turns still waited a long time for the first token, usually because a large new suffix "
            "(a big tool result or file) had to be processed.",
            f"warm TTFT p95 {warm_p95:.1f} s")

    short = [b for name, b in report["context_bands"].items()
             if b["requests"] and b["range"][1] is not None and b["range"][1] <= 16384 and b["token_weighted_decode_tps"]]
    long_ = [b for name, b in report["context_bands"].items()
             if b["requests"] and b["range"][0] >= 32768 and b["token_weighted_decode_tps"]]
    if short and long_:
        short_rate = max(b["token_weighted_decode_tps"] for b in short)
        long_rate = min(b["token_weighted_decode_tps"] for b in long_)
        drop = 1 - long_rate / short_rate if short_rate else 0
        if drop > thresholds["long_context_decode_drop_warn"]:
            add("info", "long_context_slowdown",
                "Decode slows substantially at long context (attention cost grows with the KV cache).",
                f"{short_rate:.1f} tok/s under 16K vs {long_rate:.1f} tok/s at 32K+ ({drop:.0%} lower)")

    if s.get("max_waiting"):
        add("info", "queueing", "Requests queued behind each other (the server runs one request at a time).",
            f"max waiting {s['max_waiting']} in {s['queued_intervals']} throughput interval(s)")

    for attempt in report["startup_attempts"]:
        if attempt.get("error"):
            add("info", "startup_attempts",
                "Larger contexts were tried first and did not fit; this is the launcher stepping down as designed.",
                f"{len(report['startup_attempts'])} failed attempt(s) before the chosen context")
            break
    return recs


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def fmt(value: Any, suffix: str = "", digits: int = 2) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:,.{digits}f}{suffix}"
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    return f"{value}{suffix}"


def fmt_pct(value: Optional[float]) -> str:
    return "unknown" if value is None else f"{value * 100:.1f}%"


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {secs:02d}s"


def fmt_time(epoch: Optional[float]) -> str:
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S") if epoch else "unknown"


def table(headers: List[str], rows: List[List[str]], align: Optional[List[str]] = None) -> List[str]:
    align = align or ["<"] + [">"] * (len(headers) - 1)
    widths = [max(len(str(h)), *(len(str(row[i])) for row in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    lines = ["  ".join(f"{str(h):{a}{w}}" for h, a, w in zip(headers, align, widths)).rstrip(),
             "  ".join("-" * w for w in widths)]
    for row in rows:
        lines.append("  ".join(f"{str(c):{a}{w}}" for c, a, w in zip(row, align, widths)).rstrip())
    return lines


def server_line(report: Dict[str, Any]) -> str:
    sessions = report["sessions"]
    if not sessions:
        return "Server: unknown"
    s = sessions[-1]
    parts = [f"Server: {s.get('model_id') or 'unknown'}"]
    parts.append(f"Context: {fmt(s.get('context'))}" + (f" ({s['kv_mode']})" if s.get("kv_mode") else ""))
    if s.get("slack_mib") is not None:
        parts.append(f"Slack: {fmt(s['slack_mib'], ' MiB', 0)}")
    if s.get("cuda_graph") is not None:
        parts.append(f"CUDA Graph: {fmt(s['cuda_graph'])}")
    if s.get("speculative"):
        parts.append(f"Spec: {s['speculative']}")
    if len(sessions) > 1:
        parts.append(f"Sessions: {len(sessions)}")
    return " | ".join(parts)


def requests_line(summary: Dict[str, Any]) -> str:
    fc = summary["finish_counts"]
    detail = []
    if fc.get("tool_calls"):
        detail.append(f"{fc['tool_calls']} tool-call turns")
    if fc.get("stop_token"):
        detail.append(f"{fc['stop_token']} answers")
    for key, value in sorted(fc.items()):
        if key not in ("tool_calls", "stop_token"):
            detail.append(f"{value} {key}")
    text = f"Requests: {summary['requests_done']:,} done"
    if detail:
        text += f" ({', '.join(detail)})"
    if summary["requests_in_progress"]:
        text += f" | {summary['requests_in_progress']} in progress"
    if summary["session_seconds"]:
        text += f" | Session: {fmt_duration(summary['session_seconds'])}"
    return text


def startup_failure_lines(report: Dict[str, Any]) -> List[str]:
    lines = []
    for session in report["sessions"]:
        if not session.get("started") and session.get("startup_error"):
            lines.append(f"Server did not start at {fmt(session.get('context'))} tokens: {session['startup_error']}")
    return lines


def render_human(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines = [server_line(report)] + startup_failure_lines(report) + [requests_line(s)]
    if s["error_counts"]:
        lines.append("Errors: " + ", ".join(f"{count} x {message}" for message, count in s["error_counts"].items()))
    decode = s["decode_tps"]
    lines.append(f"Median decode: {fmt(decode['median'], ' tok/s')} ({decode['samples']} samples)")
    lines.append(f"Average decode: {fmt(s['decode_tps_token_weighted'], ' tok/s')} (token-weighted)")
    lines.append(f"High decode: {fmt(decode['max'], ' tok/s')}")
    lines.append(f"Low decode: {fmt(decode['min'], ' tok/s')}")
    lines.append(
        f"Median TTFT: {fmt(s['ttft_s']['median'], ' s')} "
        f"(cache hit {fmt(s['ttft_warm_s']['median'], ' s')}, full prefill {fmt(s['ttft_cold_s']['median'], ' s')})")
    lines.append(f"Prefill: {fmt(s['prefill_cold_tps']['median'], ' tok/s')} median on large new input "
                 f"({s['prefill_cold_tps']['samples']} samples)")
    lines.append(f"Prompt cache: {fmt_pct(s['cache_reuse_rate'])} of prompt tokens reused | "
                 f"cache lost on {s['costly_resets']} turns ({fmt_duration(s['costly_reset_ttft_s'])} waiting)")
    if s["decode_outliers"]:
        lines.append(f"Slow outliers: {s['decode_outliers']} request{'s' if s['decode_outliers'] != 1 else ''} decoded < "
                     f"{report['thresholds']['decode_outlier_fraction']:.0%} of normal speed")
    if s["mtp_tokens_per_round"]["samples"]:
        lines.append(f"MTP: {fmt(s['mtp_tokens_per_round']['median'], ' tok/round')} median, "
                     f"{fmt_pct(s['mtp_acceptance_weighted'])} acceptance (token-weighted)")
    if report["recommendations"]:
        count = len(report["recommendations"])
        lines.append(f"Recommendations: {count} (show with -t or -v)")
    return "\n".join(lines)


def render_time_human(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines = [f"{TOOL_NAME} Performance Summary", "=" * (len(TOOL_NAME) + 20), "", server_line(report)]
    lines += startup_failure_lines(report) + [requests_line(s)]
    if s["error_counts"]:
        lines.append("Errors: " + ", ".join(f"{count} x {message}" for message, count in s["error_counts"].items()))
    lines.append("")

    lines += ["Decode", "------"]
    d = s["decode_tps"]
    lines.append(f"Median: {fmt(d['median'], ' tok/s')} | token-weighted: {fmt(s['decode_tps_token_weighted'], ' tok/s')}"
                 f" | p10-p90: {fmt(d['p10'])}-{fmt(d['p90'])} | range: {fmt(d['min'])}-{fmt(d['max'])} tok/s")
    lines.append(f"Generated: {fmt(s['tokens']['generated'])} tokens ({fmt(s['tokens']['reasoning'])} thinking, "
                 f"{fmt_pct(s['reasoning_share'])}) | per request median {fmt(s['generated_per_request']['median'], '', 0)}")
    if s["mtp_tokens_per_round"]["samples"]:
        lines.append(f"MTP: {fmt(s['mtp_tokens_per_round']['median'], ' tok/round')} median | acceptance "
                     f"{fmt_pct(s['mtp_acceptance_weighted'])} token-weighted, {fmt_pct(s['mtp_acceptance_median'])} median")
    lines.append("")

    lines += ["Time to first token", "-------------------"]
    for label, key in (("All", "ttft_s"), ("Cache hit", "ttft_warm_s"), ("Full prefill", "ttft_cold_s")):
        block = s[key]
        lines.append(f"{label + ':':<14}median {fmt(block['median'], ' s')} | p90 {fmt(block['p90'], ' s')} | "
                     f"p95 {fmt(block['p95'], ' s')} | max {fmt(block['max'], ' s')} ({block['samples']} samples)")
    p = s["prefill_cold_tps"]
    lines.append(f"Prefill speed: median {fmt(p['median'], ' tok/s')} | range {fmt(p['min'])}-{fmt(p['max'])} tok/s "
                 f"({p['samples']} requests computing >= {report['thresholds']['cold_prefill_min_tokens']:,} new tokens)")
    if s["decode_outliers"]:
        window = s["decode_outlier_window"]
        lines.append(f"Slow outliers: {s['decode_outliers']} request{'s' if s['decode_outliers'] != 1 else ''} decoded < "
                     f"{report['thresholds']['decode_outlier_fraction']:.0%} of the median for their prompt size "
                     f"({window[0]} to {window[1]})")
    lines.append("")

    lines += ["Prompt cache", "------------"]
    t = s["tokens"]
    lines.append(f"Prompt tokens: {fmt(t['prompt'])} | reused from cache: {fmt(t['cached'])} "
                 f"({fmt_pct(s['cache_reuse_rate'])}) | computed: {fmt(t['computed'])}")
    paths = ", ".join(f"{k} {v}" for k, v in sorted(s["reuse_paths"].items(), key=lambda kv: -kv[1]))
    lines.append(f"Reuse paths: {paths or 'none'}")
    lines.append(f"Full resets after the first request: {s['full_resets_after_first']} "
                 f"(cache lost on {s['costly_resets']}, {fmt_duration(s['costly_reset_ttft_s'])} of TTFT)")
    for cause, count in sorted(s["reset_causes"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {count:>4} x {RESET_CAUSES.get(cause, cause)} "
                     f"({fmt_duration(s['reset_cause_ttft_s'].get(cause, 0.0))})")
    lines.append("")

    lines += ["Context and load", "----------------"]
    lines.append(f"Largest prompt: {fmt(s['max_prompt_tokens'])} of {fmt(s['context_capacity'])} tokens "
                 f"({fmt_pct(s['max_context_fill'])}) | median prompt {fmt(s['prompt_tokens']['median'], '', 0)}")
    lines.append(f"Thinking: {s['thinking_requests']} requests | hit budget: {s['budget_hits']} "
                 f"({fmt_pct(s['budget_hit_rate'])})")
    if s["busy_fraction"] is not None:
        lines.append(f"GPU busy: {fmt_pct(s['busy_fraction'])} of session time | max queued requests: {s['max_waiting']}")
    if s["session_seconds"]:
        lines.append(f"Rate: {fmt(s['requests_per_hour'], ' requests/h', 1)} | {fmt(s['generated_per_hour'], ' generated tok/h', 0)}")
    lines.append("")

    lines += render_startup(report)
    lines += render_recommendations(report)
    return "\n".join(lines).rstrip()


def render_startup(report: Dict[str, Any]) -> List[str]:
    lines = ["Startup", "-------"]
    for s in report["sessions"]:
        lines.append(f"{os.path.basename(s['source'])}: started {fmt_time(s['first_time'])}"
                     + (f", {s['endpoint']}" if s.get("endpoint") else "")
                     + (f", model id {s['model_id']}" if s.get("model_id") else ""))
        lines.append(f"  context {fmt(s['context'])} ({s.get('kv_mode') or 'unknown'}), load {fmt(s['load_seconds'], ' s')}, "
                     f"weights {fmt(s['weights_gib'], ' GiB')}, runtime {fmt(s['runtime_mib'], ' MiB', 0)}")
        lines.append(f"  free after weights {fmt(s['free_after_weights_mib'], ' MiB', 0)}, after startup "
                     f"{fmt(s['free_after_startup_mib'], ' MiB', 0)}, slack {fmt(s['slack_mib'], ' MiB', 0)}, "
                     f"graph {fmt(s['graph_observed_mib'], '', 1)}/{fmt(s['graph_allowance_mib'], ' MiB', 1)}")
        if s.get("startup_error"):
            lines.append(f"  startup error: {s['startup_error']}")
    for a in report["startup_attempts"]:
        reason = a.get("error") or "did not start"
        lines.append(f"Attempt at {fmt(a.get('requested_context'))} tokens: {reason}")
    lines.append("")
    return lines


def render_recommendations(report: Dict[str, Any]) -> List[str]:
    lines = ["Recommendations (fixed heuristics)", "----------------------------------"]
    if not report["recommendations"]:
        lines.append("None.")
    for rec in report["recommendations"]:
        lines.append(f"[{rec['severity']}] {rec['message']}")
        lines.append(f"       evidence: {rec['evidence']}")
    if report["warnings"]:
        lines += ["", "Warnings", "--------"] + report["warnings"]
    return lines


def band_range_text(bounds: List[Optional[int]]) -> str:
    low, high = bounds
    def k(v: int) -> str:
        return f"{v // 1024}K" if v % 1024 == 0 and v else str(v)
    return f">= {k(low)}" if high is None else f"{k(low)}-{k(high)}"


def render_context_table(report: Dict[str, Any]) -> str:
    rows = []
    for name, b in report["context_bands"].items():
        rows.append([name, str(b["requests"]), fmt(b["token_weighted_decode_tps"]), fmt(b["median_ttft_warm_s"], " s"),
                     fmt(b["median_ttft_cold_s"], " s"), fmt(b["median_cold_prefill_tps"], "", 0),
                     fmt_pct(b["median_mtp_acceptance"]), fmt_pct(b["cache_reuse_rate"])])
    lines = ["Performance by Prompt Size", "--------------------------"]
    lines += table(["Prompt", "Reqs", "Decode tok/s", "TTFT hit", "TTFT cold", "Prefill tok/s", "MTP acc", "Cache"], rows)
    lines.append("Decode is token-weighted; TTFT hit = prefix-cache reuse, cold = full prefill; prefill counts "
                 "only prompts computing >= 2K tokens.")
    return "\n".join(lines)


def render_speed_graph(report: Dict[str, Any]) -> str:
    samples = sorted((r["prompt_tokens"], r["decode_tps"]) for r in report["requests"]
                     if r.get("done") and r.get("prompt_tokens") and r.get("decode_tps"))
    lines = ["Decode Speed vs Prompt Size", "---------------------------"]
    if not samples:
        lines.append("No decode samples with prompt size found.")
        return "\n".join(lines)
    width, height = 60, 12
    xs = [x for x, _ in samples]
    ys = [y for _, y in samples]
    x_min, x_max = min(xs), max(xs)
    y_max = max(ys) or 1.0
    grid = [[" "] * width for _ in range(height)]
    for x, y in samples:
        col = width // 2 if x_max == x_min else int(round((x - x_min) / (x_max - x_min) * (width - 1)))
        row = height - 1 - int(round(y / y_max * (height - 1)))
        row = max(0, min(height - 1, row))
        grid[row][col] = "#" if grid[row][col] != " " else "*"
    for index, cells in enumerate(grid):
        y_value = y_max - y_max * index / (height - 1)
        lines.append(f"{y_value:>8.1f} |{''.join(cells)}")
    lines.append(f"{'':>8} +{'-' * width}")
    lines.append(f"{'Prompt':>8}: {x_min:,} -> {(x_min + x_max) // 2:,} -> {x_max:,} tokens")
    lines.append(f"Samples: {len(samples)} | Decode range: {min(ys):.1f} to {max(ys):.1f} tok/s")
    if any(c == "#" for cells in grid for c in cells):
        lines.append("Legend: * = one request, # = several in the same cell")
    return "\n".join(lines)


def request_rows(report: Dict[str, Any]) -> Tuple[List[str], List[List[str]]]:
    headers = ["Req", "Time", "Finish", "Msgs", "Prompt", "Cached", "Reuse", "TTFT", "Gen", "Think", "Decode", "MTP",
               "Wall", "Note"]
    notes = {"new_conversation": "new/side", "after_side_request": "evicted by side req",
             "interleaved": "interleaved", "history_rewritten": "history rewritten"}
    rows = []
    for r in report["requests"]:
        when = datetime.fromtimestamp(r["start_time"]).strftime("%H:%M:%S") if r.get("start_time") else ""
        if not r.get("done"):
            status = "error" if r.get("errors") else "running"
            note = "; ".join(r.get("errors") or [])
            rows.append([str(r.get("label", r["request_id"])), when, status, fmt(r.get("messages")), "", "", "", "", "",
                         "", "", "", "", note])
            continue
        note_parts = []
        if r.get("reset_cause"):
            note_parts.append(notes.get(r["reset_cause"], r["reset_cause"]))
        if r.get("decode_outlier"):
            note_parts.append("SLOW")
        if r.get("errors"):
            note_parts.extend(r["errors"])
        finish = r.get("finish") or ""
        if finish == "tool_calls" and r.get("tool_calls"):
            finish = f"tools x{r['tool_calls']}"
        elif finish == "stop_token":
            finish = "answer"
        rows.append([
            str(r.get("label", r["request_id"])), when, finish, fmt(r.get("messages")), fmt(r.get("prompt_tokens")),
            fmt(r.get("cached_tokens")), REUSE_SHORT.get(r.get("reuse") or "", r.get("reuse") or ""),
            fmt(r.get("ttft_s"), " s"), fmt(r.get("generated_tokens")), fmt(r.get("reasoning_tokens")),
            fmt(r.get("decode_tps"), "", 1),
            fmt_pct(r.get("spec_acceptance")) if r.get("spec_acceptance") is not None else "",
            fmt(r.get("wall_s"), " s", 1), "; ".join(note_parts)])
    return headers, rows


def render_requests(report: Dict[str, Any]) -> str:
    headers, rows = request_rows(report)
    lines = ["Requests", "--------"]
    if not rows:
        lines.append("No requests.")
        return "\n".join(lines)
    lines += table(headers, rows, ["<", "<", "<"] + [">"] * (len(headers) - 4) + ["<"])
    return "\n".join(lines)


def render_verbose_human(report: Dict[str, Any]) -> str:
    return "\n\n".join([render_time_human(report), render_context_table(report), render_speed_graph(report),
                        render_requests(report)])


def md_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def render_markdown(report: Dict[str, Any], include_requests: bool = False) -> str:
    s = report["summary"]
    lines = [f"# {TOOL_NAME} report", "", f"**{server_line(report)}**", "", requests_line(s), "", "## Summary", ""]
    rows = [
        ["Median decode", fmt(s["decode_tps"]["median"], " tok/s")],
        ["Token-weighted decode", fmt(s["decode_tps_token_weighted"], " tok/s")],
        ["Decode range", f"{fmt(s['decode_tps']['min'])} - {fmt(s['decode_tps']['max'])} tok/s"],
        ["Median TTFT (cache hit / full prefill)",
         f"{fmt(s['ttft_warm_s']['median'], ' s')} / {fmt(s['ttft_cold_s']['median'], ' s')}"],
        ["TTFT p95", fmt(s["ttft_s"]["p95"], " s")],
        ["Prefill on large new input (median)", fmt(s["prefill_cold_tps"]["median"], " tok/s")],
        ["Prompt tokens reused from cache", fmt_pct(s["cache_reuse_rate"])],
        ["Turns that lost the cache", f"{s['costly_resets']} ({fmt_duration(s['costly_reset_ttft_s'])} waiting)"],
        ["Slow decode outliers", str(s["decode_outliers"])],
        ["MTP tokens/round (median)", fmt(s["mtp_tokens_per_round"]["median"])],
        ["MTP acceptance (token-weighted)", fmt_pct(s["mtp_acceptance_weighted"])],
        ["Thinking share of generated tokens", fmt_pct(s["reasoning_share"])],
        ["Thinking budget hits", f"{s['budget_hits']} ({fmt_pct(s['budget_hit_rate'])})"],
        ["Largest prompt / context", f"{fmt(s['max_prompt_tokens'])} / {fmt(s['context_capacity'])}"],
    ]
    lines += md_table(["Metric", "Value"], rows)
    lines += ["", "## Performance by prompt size", ""]
    band_rows = [[name, str(b["requests"]), fmt(b["token_weighted_decode_tps"]), fmt(b["median_ttft_warm_s"], " s"),
                  fmt(b["median_ttft_cold_s"], " s"), fmt(b["median_cold_prefill_tps"], "", 0),
                  fmt_pct(b["median_mtp_acceptance"]), fmt_pct(b["cache_reuse_rate"])]
                 for name, b in report["context_bands"].items()]
    lines += md_table(["Prompt", "Reqs", "Decode tok/s", "TTFT hit", "TTFT cold", "Prefill tok/s", "MTP acc", "Cache"],
                      band_rows)
    lines += ["", "## Startup", ""]
    lines += [line if line else "" for line in render_startup(report)[2:]]
    lines += ["## Recommendations", ""]
    if not report["recommendations"]:
        lines.append("None.")
    for rec in report["recommendations"]:
        lines.append(f"- **{rec['severity']}** {rec['message']} _({rec['evidence']})_")
    if report["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in report["warnings"]]
    if include_requests:
        headers, req_rows = request_rows(report)
        lines += ["", "## Requests", ""] + md_table(headers, req_rows)
    return "\n".join(lines)


def render_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=False)


# --------------------------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------------------------


def comparison_rows(reports: List[Tuple[str, Dict[str, Any]]]) -> Tuple[List[str], List[List[str]]]:
    headers = ["Metric"] + [label for label, _ in reports]
    metrics = [
        ("Context", lambda r: fmt(r["summary"]["context_capacity"])),
        ("Requests", lambda r: fmt(r["summary"]["requests_done"])),
        ("Median decode tok/s", lambda r: fmt(r["summary"]["decode_tps"]["median"])),
        ("Token-weighted decode tok/s", lambda r: fmt(r["summary"]["decode_tps_token_weighted"])),
        ("Median TTFT hit", lambda r: fmt(r["summary"]["ttft_warm_s"]["median"], " s")),
        ("Median TTFT cold", lambda r: fmt(r["summary"]["ttft_cold_s"]["median"], " s")),
        ("Prefill tok/s (large input)", lambda r: fmt(r["summary"]["prefill_cold_tps"]["median"], "", 0)),
        ("Cache reuse", lambda r: fmt_pct(r["summary"]["cache_reuse_rate"])),
        ("Turns that lost the cache", lambda r: fmt(r["summary"]["costly_resets"])),
        ("Slow decode outliers", lambda r: fmt(r["summary"]["decode_outliers"])),
        ("MTP acceptance", lambda r: fmt_pct(r["summary"]["mtp_acceptance_weighted"])),
        ("Thinking budget hits", lambda r: fmt_pct(r["summary"]["budget_hit_rate"])),
        ("Largest prompt", lambda r: fmt(r["summary"]["max_prompt_tokens"])),
        ("Startup slack MiB", lambda r: fmt(r["sessions"][-1]["slack_mib"] if r["sessions"] else None, "", 0)),
    ]
    rows = [[name] + [getter(report) for _, report in reports] for name, getter in metrics]
    return headers, rows


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="NInferLogAnalyzer.py",
        description="Performance analyzer for NInfer server (ninfer-serve) logs. Accepts server log files, "
                    "--request-log-jsonl files, or a directory of logs (the latest session is used unless --all).")
    parser.add_argument("inputs", nargs="*", help="log file(s) or a log directory")
    parser.add_argument("--file", action="append", default=[], help="log file or directory (repeatable)")
    parser.add_argument("--clipboard", action="store_true")
    parser.add_argument("--stdin", action="store_true")
    parser.add_argument("--all", action="store_true", help="directory input: analyze every started session")
    parser.add_argument("--last", type=int, help="only the last N requests")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs="+", metavar="INPUT", help="compare sessions (files or directories)")
    parser.add_argument("--thresholds", help="JSON file overriding heuristic thresholds")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("-t", "--time", action="store_true", help="detailed performance summary")
    parser.add_argument("--context-table", "--speed-table", dest="context_table", action="store_true",
                        help="performance by prompt size")
    parser.add_argument("--speed-graph", action="store_true", help="decode speed vs prompt size")
    parser.add_argument("-r", "--requests", action="store_true", help="per-request table")
    parser.add_argument("-v", "--verbose", action="store_true", help="everything")
    parser.add_argument("-w", "--watch", nargs="?", type=float, const=5.0, metavar="SECONDS",
                        help="refresh continuously (default every 5 s)")
    return parser


def render_report(report: Dict[str, Any], args: argparse.Namespace) -> str:
    if args.json:
        return render_json(report)
    if args.markdown:
        return render_markdown(report, include_requests=args.requests or args.verbose)
    if args.verbose:
        return render_verbose_human(report)
    sections = [render_time_human(report) if args.time else render_human(report)]
    if args.context_table:
        sections.append(render_context_table(report))
    if args.speed_graph:
        sections.append(render_speed_graph(report))
    if args.requests:
        sections.append(render_requests(report))
    return "\n\n".join(sections)


def write_output(text: str, out_path: Optional[str], encoding: str) -> None:
    if out_path:
        with open(out_path, "w", encoding=encoding, errors="replace") as handle:
            handle.write(text)
            handle.write("\n")
    else:
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode(sys.stdout.encoding or "ascii", errors="replace").decode(sys.stdout.encoding or "ascii"))


def build_report(args: argparse.Namespace, inputs: List[str], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    sessions, attempts, labels = load_sessions(args, inputs)
    report = analyze(sessions, attempts, thresholds, last=args.last)
    report["inputs"] = labels
    return report


def run(args: argparse.Namespace) -> int:
    if args.json and args.markdown:
        raise UsageError("Choose either --json or --markdown, not both")
    if args.last is not None and args.last <= 0:
        raise UsageError("--last must be a positive number")
    thresholds = load_thresholds(args.thresholds)
    inputs = list(args.inputs) + list(args.file)

    if args.compare:
        reports = []
        for item in args.compare:
            sub = argparse.Namespace(**vars(args))
            sub.clipboard = False
            sub.stdin = False
            report = build_report(sub, [item], thresholds)
            label = os.path.basename(report["sessions"][-1]["source"]) if report["sessions"] else os.path.basename(item)
            reports.append((label, report))
        if args.json:
            output = render_json({"schema_version": SCHEMA_VERSION, "tool": TOOL_NAME,
                                  "reports": {label: rep for label, rep in reports}})
        else:
            headers, rows = comparison_rows(reports)
            if args.markdown:
                output = "\n".join([f"# {TOOL_NAME} comparison", ""] + md_table(headers, rows))
            else:
                output = "\n".join(["Session Comparison", "------------------"] + table(headers, rows))
        write_output(output, args.out, args.encoding)
        return 0

    if args.watch is not None:
        if args.clipboard or not inputs:
            raise UsageError("--watch needs a log file or directory")
        interval = max(1.0, args.watch)
        try:
            while True:
                report = build_report(args, inputs, thresholds)
                text = render_report(report, args)
                if sys.stdout.isatty():
                    os.system("cls" if os.name == "nt" else "clear")
                print(text)
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] refreshing every {interval:g} s - Ctrl+C to stop",
                      flush=True)
                time.sleep(interval)
        except KeyboardInterrupt:
            return 0

    report = build_report(args, inputs, thresholds)
    write_output(render_report(report, args), args.out, args.encoding)
    return 1 if report["warnings"] or not report["summary"]["requests_done"] else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    try:
        return run(args)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if str(exc) == "No input provided":
            parser.print_help(sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive
        print(f"internal error: {exc}", file=sys.stderr)
        if getattr(args, "verbose", False):
            raise
        return 3


if __name__ == "__main__":
    sys.exit(main())
