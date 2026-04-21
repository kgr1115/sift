# Design Decisions

This doc is the "why" behind the code. Every section here is something I could go 10 minutes deep on in an interview without pulling up notes.

---

## 1. Pipeline, not agent

**Decision:** A fixed sequence of discrete LLM calls (fetch → classify → summarize → draft → assemble), not a tool-using agent that decides what to do next.

**Why:** The inbox problem has a fixed structure. Every thread gets classified; urgent and needs_reply threads get drafted; everything gets assembled into a brief. An agent loop would add cost (the extra "which tool next?" turns), latency, and variance, while giving me nothing in return. Agents earn their keep when the action space is open-ended — here it isn't.

**Trade-off:** I lose the "ask your inbox questions" use case ("find me that email from last week about the contract"). That's a real use case, and I'd revisit it as a Phase-2 feature — but scoped as a separate agent with its own evals. Bolting it onto the daily-brief pipeline would muddy both.

**What I'd change at 10x scale:** The classifier would live in a batch-inference system (Anthropic's Message Batches API) because latency per thread stops mattering when you're processing them in bulk overnight. For Kyle's personal inbox, on-demand is fine.

## 2. Structured output via tool-use, not JSON mode

**Decision:** Every structured LLM call uses Claude's tool-use pattern with a schema, not free-text JSON parsing.

**Why:** Claude is trained to output valid tool-call arguments matching the declared schema. The SDK handles schema enforcement and gives you clean errors when it fails. Doing `response.content.json()` on a free-text completion means writing your own robustness layer against markdown code fences, schema drift, and trailing commas — all of which are solved problems at the API level.

**Trade-off:** You can't easily stream tool-use output. If I wanted "watch the classifier run in real time," I'd need a different approach. For a morning brief, batched results in <15 seconds is the right shape.

## 3. Two models, not one

**Decision:** The classifier defaults to whatever the user sets as their general model (Sonnet 4.6 by default), but the codebase is structured so switching just the classifier to Haiku is a one-line change. Drafting always uses the "nicer" model.

**Why:** 5-way classification with confidence is a task Haiku nails. It's ~5x cheaper per call and noticeably faster, which matters when you're classifying 40 threads concurrently. Drafting a reply in someone's voice is harder — register matching, addressing every ask, no AI tells — and benefits from the bigger model.

**Trade-off:** Two models means two prompt-quality distributions to watch. The evals suite scores both, so a drift in one doesn't get masked by the other.

## 4. Evals are the product

**Decision:** Hand-labeled 40-thread fixture set. Classifier eval measures per-category precision/recall and fails the test suite if any floor is breached. Drafter eval uses LLM-as-judge against a 5-dimension rubric.

**Why:** This is the biggest separator between "LLM wrapper" and "AI engineering." Every prompt change goes through the eval suite before it's merged. The alternative — eyeballing outputs on a handful of examples — is how subtle regressions ship. Recall floors are set per-category with *recall* weighted heavier than precision, because the cost of missing an urgent email (buried in FYI) is much higher than the cost of over-flagging (a one-line summary in the urgent section that turns out to be noise).

**Trade-off:** LLM-as-judge is imperfect — the judge has its own biases. I mitigate this by (a) using a rubric rather than a single holistic score, and (b) tracking *deltas* between runs. I care less about "is draft X scored 4.2/5 in absolute terms" and more about "did this prompt change move the average up or down."

**What a skeptic would push on:** "40 examples is a small dataset." True. This is a demo project with a dataset I wrote by hand in an evening, and the right framing is "the eval scaffolding is the point." In a real product I'd build this to 300+ labeled threads, spread across multiple users, and track metrics per user-segment.

## 5. Synthetic fixture inbox, not real Gmail for dev

**Decision:** All classifier + drafter work happens against a hand-labeled JSON fixture. Gmail OAuth gets wired up after the prompts are solid.

**Why:** OAuth is slow to iterate on, and it's not where the interesting product work lives. Getting to a working pipeline without real email means I spent 100% of my prompt-engineering time on prompts, not on refreshing tokens. Once the logic works, wiring real Gmail is mostly connector code.

## 6. Drafts only — never auto-send

**Decision:** The system composes drafts but never sends. Full stop.

**Why:** Auto-send is an asymmetric risk. 99% of the time it saves 20 seconds. The 1% where it sends the wrong draft to the CEO is a disaster that overwhelms every other gain. And drafts capture most of the value anyway — the hard part is starting from a blank page, not polishing.

**Product angle:** This is also a trust argument. Users let AI help them once they trust it not to take irreversible actions without consent. "I draft, you send" is the best version of that contract.

## 7. Concurrent LLM calls with ThreadPoolExecutor

**Decision:** Classifier and drafter fan out to up to 8 and 6 concurrent requests respectively using `concurrent.futures.ThreadPoolExecutor`.

**Why:** A 40-thread inbox serially at ~1.5s per classification is a minute of waiting. Concurrent it's 6-8 seconds. The Anthropic API tolerates this well, and the per-thread retry logic is in the SDK. Python's async would be another option, but `ThreadPoolExecutor` keeps the call sites synchronous and readable for people without asyncio muscle.

**What breaks at scale:** When you hit your rate limit you want backoff + retry with jitter. `anthropic` SDK has basic retry; production would wrap this in tenacity or similar.

## 8. SQLite cache (not yet wired — but designed in)

**Decision:** Pydantic-model-based cache layer over SQLite, keyed by thread ID + content hash.

**Why:** Running the classifier every morning against the last 50 threads would re-pay for every thread that hasn't changed. A content-hash cache means classifier costs scale with *new* emails, not total emails. SQLite (single file, zero ops) is the right storage for something running on Kyle's laptop.

## 9. Prompts live on disk, not in Python strings

**Decision:** Every prompt is a markdown file under `src/sift/prompts/`. Python loads them at import time.

**Why:** Prompts diff cleanly in PRs, can be reviewed by people who can't read Python, and are version-controlled alongside the code. Embedded triple-quoted strings are a maintenance nightmare once they get longer than a paragraph. The pattern scales — adding a prompt is a new file, not a new constant in a growing module.

## 10. Rich CLI + Streamlit UI, not a React app

**Decision:** Typer + Rich for the CLI, Streamlit for a GUI. No custom frontend.

**Why:** The project's story is the AI pipeline and the evals. A week of React work on top wouldn't move those numbers. Streamlit gets us screenshots of a clean, working UI in an afternoon and is easy for recruiters to clone + run.

**What I'd do differently for production:** This wouldn't be a web UI at all — it'd be a Gmail add-on or a daily email sent to the user. A separate surface to check is a separate thing to forget to check.

---

## Things I explicitly chose not to build (and why)

- **Auto-archive or auto-label.** Too risky, and users already have filters for bulk routing.
- **Semantic search over the inbox.** A great feature, but out of scope for a week-long build and best done as a separate tool with its own index.
- **Multi-user/account support.** This is a personal tool, not a SaaS. One OAuth, one user.
- **Mobile app.** Never. If I wanted mobile, I'd send the morning brief as an actual email.
- **Integration with calendar/docs.** Meeting-prep is a different product. The "one product per git repo" principle saved me a weekend here.

---

## What I changed my mind on during the build

_(To be filled in as the project evolves. A live log of design decisions I walked back and what convinced me to walk them back.)_
