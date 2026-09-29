# NInferLogAnalyzer

NInferLogAnalyzer is a deterministic command-line performance analyzer for [NInfer](https://github.com/Neroued/ninfer)
server (`ninfer-serve`) logs. It reports decode and prefill speed, time to first token, prefix-cache reuse (and why the
cache was lost), speculative decoding (MTP) acceptance, thinking-budget use, context fill, errors and startup memory,
followed by fixed heuristic recommendations. It uses no LLM. It is a companion to
[LLLogAnalyzer](https://github.com/JonLundberg/LLLogAnalyzer) (LM Studio / llama.cpp logs) and follows its conventions.

## Requirements

- Python 3.10 or newer
- No third-party packages
- Windows-first behavior, with standard file/stdin support on other platforms

## Inputs

- **Server log**: the `ninfer-serve` console output (stderr), for example the files a launcher writes to
  `logs\server\ninfer-serve-<date>-<time>-<context>.log`.
- **Request log** (recommended): the structured file written with `ninfer-serve --request-log-jsonl FILE`.
  Full-precision timings, detected automatically.
- **A directory** of either: the most recent session that actually started is analyzed; failed startup attempts from
  the same launch are reported. Use `--all` for every session. When a text log and a JSONL log exist for the same
  launch, the JSONL log is used.

## Quickstart

```powershell
python .\NInferLogAnalyzer.py C:\path\to\logs\server           # latest session in a log directory
python .\NInferLogAnalyzer.py .\ninfer-serve.log                # one log file
python .\NInferLogAnalyzer.py .\requests.jsonl                  # structured request log
Get-Content .\ninfer-serve.log -Raw | python .\NInferLogAnalyzer.py
python .\NInferLogAnalyzer.py --clipboard
```

Or install it:

```powershell
python -m pip install -e .
ninferloganalyzer C:\path\to\logs\server
```

## Views

```powershell
python .\NInferLogAnalyzer.py <input> -t                  # detailed performance summary + recommendations
python .\NInferLogAnalyzer.py <input> --context-table     # performance by prompt size
python .\NInferLogAnalyzer.py <input> --speed-graph       # decode speed vs prompt size
python .\NInferLogAnalyzer.py <input> -r --last 20        # per-request table (last 20 requests)
python .\NInferLogAnalyzer.py <input> -v                  # everything
python .\NInferLogAnalyzer.py <dir> --watch               # live view, refreshes every 5 s (Ctrl+C to stop)
python .\NInferLogAnalyzer.py <dir> --all                 # all sessions in a directory
python .\NInferLogAnalyzer.py --compare <a.log> <b.log>   # sessions side by side
python .\NInferLogAnalyzer.py <input> --markdown --out report.md
python .\NInferLogAnalyzer.py <input> --json --out report.json
python .\NInferLogAnalyzer.py <input> --thresholds thresholds.json
```

## Example output

```text
Server: llama | Context: 65,536 (explicit) | Slack: 100 MiB | CUDA Graph: on | Spec: mtp
Requests: 8 done (4 tool-call turns, 2 answers, 1 cancelled, 1 output_limit) | Session: 7m 00s
Errors: 1 x client disconnected, 1 x inference request expired while waiting for admission
Median decode: 59.00 tok/s (8 samples)
Average decode: 51.66 tok/s (token-weighted)
High decode: 70.00 tok/s
Low decode: 5.00 tok/s
Median TTFT: 6.75 s (cache hit 0.95 s, full prefill 19.50 s)
Prefill: 1,642.85 tok/s median on large new input (4 samples)
Prompt cache: 52.8% of prompt tokens reused | cache lost on 2 turns (46.0 s waiting)
Slow outliers: 1 request decoded < 50% of normal speed
MTP: 2.75 tok/round median, 59.9% acceptance (token-weighted)
Recommendations: 8 (show with -t or -v)
```

## Concepts

- **Decode** is the generation speed of a request. *Average* is token-weighted (total tokens / total decode time), so
  long answers count more than short ones; the median treats every request equally.
- **TTFT** (time to first token) is split into *cache hit* (the server reused a cached prefix of the conversation) and
  *full prefill* (the whole prompt was processed).
- **Prefill** speed is measured only on requests that processed at least 2,048 new tokens; small suffixes are dominated
  by fixed overhead.
- **Prefix-cache reuse**: `append_frontier` extends the cached conversation, `restore_*_checkpoint` rolls back to a
  checkpoint and recomputes a suffix, `full_reset` recomputes everything.
- **Why the cache was lost.** At `--max-concurrency 1` the server keeps one cached conversation, so each full reset is
  classified by comparing the request with the one before it:
  - *new conversation or side request*: no tool history and at most 3 messages. Expected, not counted as a loss.
  - *evicted by a side request*: the previous request was a one-off (title generation, summary, sub-agent).
  - *interleaved*: another conversation (different tool set or shorter history) ran in between.
  - *history rewritten*: the same conversation came back with changed history (compaction, a moved plan-mode
    reminder, a `tool_choice` change, or reasoning not echoed back without `--preserve-thinking`).
- **Slow outliers** decoded at less than half the median speed of requests with a similar prompt size. On a desktop
  GPU this usually means another application was using the GPU, or Windows moved server memory to system RAM.
- **MTP acceptance** is the share of drafted tokens the model accepted; tokens/round includes the one token each
  verification round always produces.
- **Thinking budget hits** are thinking-enabled requests whose reasoning used the whole reasoning budget.
- **Startup slack** is the server's planned spare VRAM after all reservations. It is measured against the per-process
  budget Windows grants, not against physical VRAM.

## Heuristics

Recommendations are fixed rules with built-in thresholds; they are convenience labels, not measurements. Override any
threshold with `--thresholds file.json`, using the keys of `DEFAULT_THRESHOLDS` in `NInferLogAnalyzer.py`
(for example `{"slack_warn_mib": 64, "costly_reset_ttft_warn_s": 300}`). Unknown keys are rejected.

## Exit codes

`0` success; `1` report produced with warnings, or no completed requests (for example a failed startup); `2` usage or
input error; `3` internal error.

## Privacy notes

Server logs contain request counts, token counts, timings, seeds and local endpoints, but not prompts or responses.
JSONL request logs also contain the server command line, including local file paths. Review logs before sharing them.
Tests use small hand-written fixtures under `tests/fixtures/`.

## Known limitations

- Request ids restart with every server start; with several sessions they are shown as `session:id`.
- Reset causes are inferred from message counts, tool counts and ordering; a client that sends identical-looking
  requests for different conversations can be misclassified.
- The text log rounds speeds to 0.1 tok/s and times to milliseconds; the JSONL log has full precision.
- GPU busy time is derived from throughput lines, which the server only writes for active intervals.

## Testing

```powershell
python -m unittest discover -v
```

## License

MIT. See [LICENSE](LICENSE).
