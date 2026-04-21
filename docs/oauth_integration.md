# Gmail connector + SQLite cache: build log

This is a post-hoc writeup of how Sift moved from "fixtures-only demo" to "talks to a real Gmail inbox and drafts replies there." It's the kind of document I'd want to read if I were evaluating whether another engineer understood what they built; if you're skimming, the two interesting sections are [Bugs caught during integration](#bugs-caught-during-integration) and [Why we cache, and how](#why-we-cache-and-how).

---

## Scope

The brief for this build was deliberately end-to-end:

1. **Auth** — one-command OAuth setup (`sift auth`) that caches a token, refreshes silently, and falls back to a browser flow when the token's gone.
2. **Read** — `sift brief --source gmail` fetches N recent threads through the Gmail API and runs them through the existing classifier / drafter / brief pipeline with no other changes.
3. **Write** — `sift push-drafts` classifies recent threads, drafts replies for urgent / needs-reply ones, and creates the drafts *inside* the original Gmail thread (correct `In-Reply-To` / `References` headers) so the user sees them in Drafts, threaded under the original.
4. **Cache** — a SQLite layer keyed by `(thread_id, history_id)` so morning-rerun ergonomics work: if nothing's changed since last run, we don't re-fetch or re-classify.

## Scope choices worth calling out

**Why `gmail.compose` and not `gmail.send`.** The product rule is "Sift never sends." Using `gmail.compose` makes that a property of the OAuth grant, not just code discipline — even if a future bug tries to call `users.messages.send`, Google will reject it. This is a textbook case of using the smallest scope that does the job.

**Desktop OAuth client, not a server-side one.** Sift is a CLI you run on your own machine. `InstalledAppFlow` with a Desktop-type OAuth client matches that deployment — the flow opens `localhost` as the redirect, which Google preconfigures for Desktop clients. No hosted callback URL, no Cloud Run, no additional moving parts.

**SQLite, not Redis or Postgres.** Single-user tool, single machine, no concurrency beyond the `ThreadPoolExecutor` the classifier already uses. SQLite is one file, zero admin, and with `check_same_thread=False` it handles our write patterns fine.

---

## The fetch path

`gmail_client.fetch_recent_threads(limit, query, label_ids)` is a thin wrapper over the three-call pattern you need to actually get thread bodies out of Gmail:

1. `users.threads.list` — returns thread IDs only (Gmail is stingy about payload by default).
2. `users.threads.get(format="full")` — one call per thread, returns the full MIME tree.
3. A recursive MIME walker (`_extract_body`) that picks `text/plain` when present, falls back to `text/html` with tags stripped.

The interesting nuance is the MIME walker. Real-world email has arbitrary nesting: `multipart/mixed` wrapping `multipart/alternative` wrapping the actual `text/plain` / `text/html` pair. You can't just grab the first part — you have to walk the tree and prefer plain over HTML when both are present. Writing it recursively is shorter *and* clearer than a depth counter.

One thing I resisted the urge to add: parsing attachments, inline images, or quoted-reply histories. Those would be great if we were building a Gmail client; for classification and drafting, the top-level body is what matters, and the LLM will happily ignore the quoted footer if we don't feed it.

## The push path

`push_draft(draft)` is a single `users.drafts.create` call with a crafted MIME body. The part that matters is threading — you want the draft to show up inside the original conversation in Gmail's UI, not as a new thread.

Two things make threading work:

1. **`threadId` on the draft resource.** This is how Gmail knows which conversation the draft belongs to.
2. **`In-Reply-To` and `References` headers in the MIME.** These are the RFC 2822 way of chaining messages; Gmail needs them for its own threading heuristics on top of `threadId`.

Both must be present. With only `threadId`, Gmail sometimes attaches the draft but orphans it in the thread UI. With only the headers, it's random.

Unit tests for the MIME builder live in `tests/test_gmail_client.py` — they construct a fake `Draft` + `Thread` pair and assert the `In-Reply-To` header matches the original `Message-ID`. Doesn't require network, runs in milliseconds, caught a typo early.

---

## Bugs caught during integration

### Bug 1: Wrong OAuth client type

**Symptom.** `sift auth` raised `KeyError: 'installed'` before opening the browser.

**Cause.** I'd created a "Web application" OAuth client in the Google Cloud Console. `credentials.json` for a Web client has a top-level `"web"` key; `InstalledAppFlow.from_client_secrets_file` expects `"installed"`.

**Fix.** Delete the Web client, create a new OAuth client with **Application type: Desktop app**, download the new `credentials.json`. `localhost` redirect is preconfigured for Desktop clients.

**Why it matters.** If I'd been running this as a hosted web app, Web would've been correct — but Sift is a CLI, and the Desktop flow is what matches its topology. The error message Google's library gave wasn't specific ("installed" is not a recognized name to most developers); `docs/gmail_setup.md` now calls this out explicitly.

### Bug 2: Mixed-timezone datetime sort crashes push-drafts

**Symptom.** `sift push-drafts --limit 20` got all the way through classification and drafting, then crashed with:

```
TypeError: can't compare offset-naive and offset-aware datetimes
```

when sorting classified threads for the brief.

**Cause.** `email.utils.parsedate_to_datetime` returns a **naive** `datetime` for headers with a `-0000` time zone. RFC 2822 defines `-0000` as "UTC, but origin unknown" — Python's stdlib treats that as "no timezone info." A batch of threads from a real inbox will have a mix: most have `-0700` or `+0100`, some automated senders use `-0000`. `sorted()` over a mixed list raises.

**Fix.** In `gmail_client._parse_date`, coerce naive results to aware UTC:

```python
dt = parsedate_to_datetime(raw)
if dt.tzinfo is None:
    dt = dt.replace(tzinfo=timezone.utc)
```

**Why it matters.** Two reasons. First, this is the kind of bug that you only see against real data, not fixtures — every synthetic thread in `evals/fixtures/labeled_threads.json` had a timezone offset. Second, it's a nice illustration of why the evals harness alone isn't sufficient: you also want end-to-end smoke tests against real inputs. Added `test_parse_date_minus_zero_zero_is_aware` and `test_parse_date_sort_mixed_batch` as regressions so this can't silently come back.

---

## Why we cache, and how

### Motivation

During the first push-drafts run against 20 real threads, I watched roughly half the Anthropic requests return 429 (rate limit) before the retry-with-backoff recovered them. End-to-end wall-clock was ~90 seconds. Most of that is unavoidable on a cold run. But the typical morning workflow isn't a cold run — it's "do this once, then again five minutes later after a few new emails arrived." On the second run, 18 of the 20 threads haven't changed, and re-classifying them is pure waste.

The cache targets exactly this case.

### Keying: `(thread_id, history_id)`

Gmail exposes a `historyId` on every thread resource. It bumps on *any* change to the thread — new message, label change, marked read, archived. That's the semantics we want: if the thread is byte-identical to what we classified last time, the cache hits; if anything's changed, it misses and we re-run.

For code paths where `history_id` isn't available (running against fixtures, for example), we key by `thread_id` alone and cache indefinitely. `sift cache-clear` wipes the whole thing for a clean slate.

### Schema

Three tables — `threads`, `classifications`, `drafts` — kept separate so we can evict one independently of the others. Payloads are Pydantic-serialized JSON (`model_dump_json` / `model_validate_json`), which means inspecting the DB by hand with `sqlite3 sift.db` gives you readable records. Upserts use `INSERT ... ON CONFLICT ... DO UPDATE` so re-caching with the same key is last-write-wins.

### Connection pooling

One `sqlite3.Connection` per resolved DB path, cached in a module-level dict behind a `threading.Lock`. SQLite opens are cheap but the schema bootstrap (`CREATE TABLE IF NOT EXISTS ...`) isn't free, so we want to amortize it. `check_same_thread=False` plus SQLite's own file locking lets the classifier's `ThreadPoolExecutor` write in parallel safely.

### Cache poisoning avoidance

The classifier has a fallback: if an LLM call fails irrecoverably, it returns a `fyi` / confidence=0 placeholder classification so the brief isn't empty. We **don't** cache those — only classifications with `confidence > 0.0` are persisted. Transient errors shouldn't poison the next run.

### CLI ergonomics

```bash
sift cache-stats                      # row counts per table
sift cache-clear                      # wipe all cache tables
sift cache-clear classifications      # wipe one table
sift brief --source gmail --no-cache  # one-off bypass for a fresh run
```

`--no-cache` exists because when you change the classifier prompt or switch models, you want to force re-classification — but you don't necessarily want to wipe the cache (the previous classifications might be useful to diff against).

---

## Voice learning

The drafter has always had a clean seam for voice — `VoiceProfile` with a `render_for_prompt()` method that injects a style description into the drafter's system prompt. Through the cache and OAuth work above, that seam was filled by a hand-written `DEFAULT_VOICE` (my best guess at how I write). With Gmail read access in hand, we can now fill it from real data.

### Approach

One structured Claude call over a batch of recent sent messages. The schema is intentionally small:

- **summary** — a compressed 150–300 word style description (register rules, opener/closer patterns, length expectations, vocabulary tells).
- **style_examples** — three verbatim snippets copied directly from the input, picked to span the user's register range (personal ↔ professional).

The verbatim examples matter more than the summary. LLMs imitate concrete patterns far better than abstract descriptions — "writes lowercase with no salutation" is a lot less effective than showing the model three actual emails that do that. The drafter's system prompt includes both.

### Fetch path: why `users.messages.list`, not threads

For classification we fetch whole threads because context matters — a reply makes sense only in light of what came before. For voice learning we want the *opposite*: just the user's outbound text, without the inbound messages mixed in. So `fetch_sent_messages` uses `users.messages.list(labelIds=["SENT"])` at the message level rather than the thread-level API. Each returned body is plain text (subjects and "To:" included) with no quoted reply history.

### Caching + TTL

Voice profiles live in a fourth table, `voice_profiles`, keyed by `user_email` rather than thread id. The cache accessor takes an optional `max_age_seconds` — on a hit older than the TTL it returns None so the caller can treat it as a miss and re-learn. Default TTL is seven days: writing style drifts on the order of months, but a weekly refresh picks up genuinely new patterns (a new employer, a new regular contact, a style shift) without manual intervention.

### Resolver ergonomics

`current_voice_profile(user_email=...)` is the seam the drafter actually calls. Behavior:

1. If `user_email` is set and `use_cache=True`, look up the cache with TTL.
2. On a hit, return it.
3. On a miss or no email, fall back to `DEFAULT_VOICE`.

Deliberately no learn-on-miss. Turning every first `sift brief` run into an extra-expensive fetch+learn call (which also requires network) is the kind of hidden cost users hate. Voice learning is explicit via `sift learn-voice`.

### CLI

```bash
sift learn-voice              # fetch + learn + cache, skipping if fresh
sift learn-voice --force      # re-learn even if a fresh profile is cached
sift learn-voice --limit 100  # analyze more sent mail (costs more tokens)
```

After that, `sift brief --source gmail` and `sift push-drafts` resolve the cached profile via `current_voice_profile(user_email=whoami())` — no extra flag needed.

### What I'd do differently with more time

- **Register-conditional profiles.** Right now we learn one profile that covers both registers. A cleaner approach would be to cluster sent mail by recipient (personal addresses vs. work domains) and learn two profiles, with the drafter picking the right one based on the incoming thread. Deferred because it doubles the LLM cost for a feature the current `summary` already approximates via register-shift description.
- **Eval for voice drift.** The drafter eval's LLM-as-judge rubric has a "register-match" dimension, but it scores against hand-written reference replies, not against the learned voice profile. A cleaner eval would ask "given this voice profile, does this draft match it?" — closer to the production loop.

---

## What I'd do next

A few things I deliberately punted on:

- **Incremental fetch via `users.history.list`.** Right now `fetch_recent_threads` always pulls the N newest threads. A more efficient approach is to record the `historyId` at the end of each run and use `users.history.list(startHistoryId=...)` to pull only what changed. Not worth the extra code for a single-user CLI; would matter if this were a hosted service.
- **Cache warm-up for voice learning.** Once the real voice learner lands (pulling sentences from `in:sent` to profile Kyle's register), that's another expensive computation we'd want to cache. Probably a fourth table keyed by `user_email`, with a TTL rather than an invalidation key.
- **Eviction policy.** Currently the cache grows forever. For this use case that's fine (10k threads at ~2KB each is 20MB), but if we ever cache anything large — message bodies with attachments, say — we'd want an LRU or a size cap.

---

## Files involved

- `src/sift/gmail_client.py` — OAuth, fetch threads, MIME parse, reply-MIME build, push to Drafts, fetch sent messages.
- `src/sift/cache.py` — SQLite layer (threads / classifications / drafts / voice_profiles).
- `src/sift/voice.py` — voice learner + cache-aware resolver + `DEFAULT_VOICE` fallback.
- `src/sift/models.py` — `VoiceProfile` Pydantic model (added `user_email`, `learned_at`).
- `src/sift/prompts/voice.md` — voice-learner system prompt.
- `src/sift/classifier.py`, `src/sift/drafter.py` — cache integration (check first, run misses, cache successes). Drafter also threads `user_email` to resolve the right voice profile.
- `src/sift/cli.py` — `sift auth`, `sift push-drafts`, `sift learn-voice`, `sift cache-stats`, `sift cache-clear`, `--source gmail`, `--no-cache`.
- `tests/test_gmail_client.py` — pure-unit tests for MIME parser, header helpers, reply builder, `_parse_date` regressions.
- `tests/test_cache.py` — round-trip tests for each entity, history-id invalidation, voice-profile TTL, clear/stats, last-write-wins.
- `tests/test_voice.py` — mocked-LLM tests for the voice learner and `current_voice_profile` resolution.
- `docs/gmail_setup.md` — user-facing setup walkthrough.
