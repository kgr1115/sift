# Provider Comparison

_Generated 2026-04-21 15:38:31. 40 labeled fixtures._

| Provider | Model | Accuracy | Errors | Total cost | $/1k threads | Avg latency | In / Out tokens |
|----------|-------|---------:|-------:|-----------:|-------------:|------------:|---------------:|
| groq | `llama-3.3-70b-versatile` | 97.5% | 0 | $0.0299 | $0.747 | 5254 ms | 48,136 / 1,891 |
| anthropic | `claude-sonnet-4-6` | 95.0% | 0 | $0.3032 | $7.581 | 3109 ms | 73,317 / 5,552 |
| openai | `gpt-4o-mini` | 92.5% | 0 | $0.0075 | $0.187 | 1479 ms | 42,284 / 1,886 |
| google | `gemini-2.5-flash` | 90.0% | 0 | $0.0182 | $0.455 | 1020 ms | 39,559 / 2,525 |

## Per-category recall

| Provider | fyi | needs_reply | newsletter | trash | urgent |
|----------|---:|---:|---:|---:|---:|
| groq | 100% | 100% | 88% | 100% | 100% |
| anthropic | 100% | 80% | 100% | 100% | 100% |
| openai | 91% | 90% | 100% | 83% | 100% |
| google | 100% | 80% | 88% | 100% | 80% |

## Notes

- **Accuracy** is raw classification accuracy over all labeled fixtures.
- **Errors** are calls where the provider raised or returned an invalid payload (counted as a miss against the `fyi` fallback).
- **Cost** uses the per-MTok prices declared on each provider class (`pricing` attribute). Update those if pricing moves.
- **Avg latency** is wall-clock per-call time and includes network round-trip.
- Pricing is approximate. These numbers are a *decision aid*, not an audit.
