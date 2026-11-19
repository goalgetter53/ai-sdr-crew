# Sample trace (mock mode)

Produced by `sdr-crew demo` followed by the trace viewer. Latencies are offline wall-clock; tokens and cost are mock estimates.

```
$ sdr-crew trace view traces/demo.jsonl --trace-id brightforge.example-9c6d4255
trace brightforge.example-9c6d4255
  run:run  27.6ms $0.04275 status_out=ready
    guardrail:crm_gate  0.0ms
    agent:researcher  8.8ms
      llm:researcher  0.3ms in=237 out=89 cache_r=0 cache_w=1623 $0.00813
      tool:crm_lookup  0.0ms tool=crm_lookup allowed=True
      tool:lookup_signals  0.0ms tool=lookup_signals allowed=True
      tool:fetch_url  7.2ms tool=fetch_url allowed=True
        guardrail:quarantine  0.0ms findings=0
      llm:researcher  0.7ms in=635 out=320 cache_r=1623 cache_w=0 $0.00719
    guardrail:grounding  0.1ms
    agent:strategist  0.4ms
      llm:strategist  0.3ms in=531 out=81 cache_r=0 cache_w=1210 $0.00734
    agent:copywriter  0.7ms
      llm:copywriter  0.5ms in=608 out=330 cache_r=0 cache_w=1270 $0.01154
    guardrail:deterministic_checks  7.6ms
    agent:critic  1.2ms
      llm:critic  1.1ms in=937 out=53 cache_r=0 cache_w=1317 $0.00855

$ sdr-crew trace summary traces/demo.jsonl
kind:name                          count    p50 ms    p95 ms    tokens    cost $
--------------------------------------------------------------------------------
agent:copywriter                      18       0.4       0.6         0    0.0000
agent:critic                          18       1.2       5.1         0    0.0000
agent:researcher                      19       2.1       4.1         0    0.0000
agent:strategist                      18       0.3       0.7         0    0.0000
guardrail:crm_gate                    22       0.0       0.0         0    0.0000
guardrail:deterministic_checks        18       1.7       7.6         0    0.0000
guardrail:grounding                   19       0.1       0.1         0    0.0000
guardrail:quarantine                  18       0.0       0.0         0    0.0000
llm:copywriter                        18       0.3       0.5     38197    0.1290
llm:critic                            18       1.0       5.0     39977    0.0719
llm:researcher                        38       0.3       0.4     83187    0.1638
llm:strategist                        18       0.2       0.4     31219    0.0558
run:run                               22      11.2      18.0         0    0.4205
tool:crm_lookup                       19       0.0       0.0         0    0.0000
tool:fetch_url                        19       1.0       3.2         0    0.0000
tool:lookup_signals                   19       0.0       0.0         0    0.0000
--------------------------------------------------------------------------------
runs=22 llm_tokens=192580 llm_cost=$0.4205 blocked_tool_calls=0 error_spans=0
```
