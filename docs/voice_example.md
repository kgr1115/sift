# Voice learning: a concrete example

> *A live run of `sift learn-voice` against my real sent mail, then the same draft rendered three ways — generic AI voice, learned voice with a latent bug, and learned voice after the bug was found and fixed. This is the most useful page in the repo if you want to see what "voice learning" actually changes about a draft.*

---

## What `sift learn-voice` produced

One Claude call against 23 of my 30 most recent sent messages (short acknowledgments get filtered out; see [`voice.py`](../src/sift/voice.py)). Runtime: ~33 seconds. The summary it returned, verbatim:

> Kyle writes in sentence case across all registers — no all-lowercase casual mode. His emails are consistently short to medium length (2–6 sentences is typical), regardless of whether the recipient is a recruiter, a church administrator, or HR. He never writes long paragraphs unless forced to answer multiple structured questions.
>
> **Openers:** Kyle almost always opens with "Hello!" or "Hello [First Name]!" — this is his most consistent quirk. It appears in casual, semi-professional, and professional contexts alike. For more formal/unknown recipients he uses "Dear [Title/Name]," or simply "Hello," without the exclamation point.
>
> **Closers:** He almost always closes with "Thank you," or "Thank you!" followed by a line break and "Kyle Rauch" (full name). In purely professional post-interview contexts he upgrades to "Sincerely, / Kyle Rauch."
>
> **Tone and register:** Warm and polite at all levels. Exclamation points frequent in greetings and sign-offs but rare in body sentences. Uses "No worries!" to dismiss apologies. Often explicitly acknowledges the other person's effort ("Thank you for reaching out," "I really appreciate it").
>
> **Structure:** Prose-only — no bullets in casual/semi-pro emails. Only uses numbered lists when explicitly answering numbered questions. No bold, no em-dashes, no block quotes in his own composing.
>
> **Notable quirks:** "Hello!" as a universal opener is the single most distinctive habit. Full-name sign-off ("Kyle Rauch") is nearly universal, even in one-line replies.

The learner also captures three verbatim style examples from real sent mail. Those aren't reproduced here (they contain other people's names) but they're the thing that actually moves the needle at draft time — a one-paragraph description teaches an LLM less than three short examples it can pattern-match against.

---

## The A/B: same thread, three drafts

The test case: Anthropic's automated billing system emailed me saying a $10.73 charge to my card failed. A legitimate-looking email but also a classic phishing shape. A reasonable draft should be "can you confirm this is real before I click any links."

### Version 1 — generic `DEFAULT_VOICE` (no voice learning)

```
Hi,

Just following up on my previous message — I want to update my payment info
but wanted to confirm this email is legitimate before clicking any links.
Can someone verify?

Thanks,
Kyle
```

Note the em-dash, the "Hi," opener, the "Thanks, / Kyle" sign-off. This is a perfectly fine draft — it's just not my voice. No one who knows me would read this and say "that sounds like Kyle."

### Version 2 — learned voice, pre-fix

```
Hello!

Thank you for reaching out. This is a legitimate billing notification from
Anthropic. The link in the email directing you to claude.ai is correct and
safe to use to update your payment information.

If you have any additional questions, feel free to let us know!

Sincerely,
Kyle Rauch
```

The style match is **correct** — "Hello!" opener, full-name sign-off, warm tone, no em-dashes. That's what the learned profile asked for.

The **content is catastrophic.** The draft is written as if I'm Anthropic customer service *replying to a confused customer*. I'd be cheerfully confirming the legitimacy of a charge against my own card.

This is the single most interesting failure mode I hit on this project. A better voice match made a latent prompt bug *worse*, because the draft now *sounded* confident enough to slip past a skim-read.

### Version 3 — learned voice, post-fix

```
Dear Anthropic Support,

I wanted to follow up on my previous email. I have not yet received a
response confirming the legitimacy of the billing notice I received from
failed-payments@mail.anthropic.com regarding a $10.73 charge to my
Mastercard ending in 3545.

Could someone please confirm this email is authentic so I can update my
payment information? I want to make sure I am not clicking a fraudulent
link before proceeding.

Thank you,
Kyle Rauch
```

Same voice match as v2 — `Dear [Team],` opener for an unknown support team, `Thank you, / Kyle Rauch` closer, no em-dashes, no bullets. But the framing is now correct: I'm the recipient asking Anthropic Billing to verify a possibly-fraudulent email against my card. Draftable as-is.

---

## What went wrong in v2, and what fixed it

The drafter's user message was, literally:

```
Draft a reply to this thread:

---
From: Anthropic, PBC <failed-payments@mail.anthropic.com>
Subject: $10.73 payment to Anthropic, PBC was unsuccessful

[body text addressed to "Dear Kyle"]
---
```

Pass that to an LLM and you've given it two conflicting signals: the system prompt says *"You are drafting on Kyle's behalf,"* but the user message just labels the email by sender and attaches a body that's *addressed to Kyle*. In a tie, the model often picks the side with more textual context — and that's the sender.

Three changes, all in [`drafter.py`](../src/sift/drafter.py) and [`prompts/draft.md`](../src/sift/prompts/draft.md), resolved it:

1. **Make the recipient explicit in the rendered thread.** The thread renderer now stamps Kyle's authenticated email as a `To:` line with a parenthetical: `(this is you — you are Kyle, replying TO the sender above)`. The LLM can no longer resolve the ambiguity against us.

2. **Rewrite the user-message opener.** From *"Draft a reply to this thread"* to *"Draft Kyle's reply to the email thread below. Kyle is the recipient (To:), and he is writing back to the sender (From:)."* Twice as many words, zero ambiguity.

3. **Add an anti-pattern to the system prompt.** The draft.md prompt now includes a paragraph specifically calling out the "your card was declined" failure mode: *"Do not write as if you are the original sender replying to yourself — you are Kyle, replying to them. If the incoming email says 'your card was declined,' Kyle's reply addresses his card, not a customer's."*

Collectively these changes don't cost a single extra token at runtime (the prompt additions are ~80 tokens on a prompt that already weighs >1000) and they eliminated the failure mode end-to-end against this exact thread.

---

## Two things I learned building this

**Voice imitation amplifies correctness bugs.** When the draft was obviously AI-generic, the sender-confusion was easier to catch — the draft sounded wrong, so I'd re-read it before sending. Once the voice was mine, skim-reading felt safe. This mirrors the broader pattern that higher-fidelity AI outputs raise the stakes of correctness bugs rather than lowering them, because the reader's defenses drop.

**The prompt fix wasn't "tune the instructions" — it was "stop giving the model an ambiguous input."** I initially tried rewriting the system prompt alone and the bug persisted across several variants. What actually worked was making the user-message *structurally* unambiguous about who's on which side of the exchange. That's a lesson I've seen in other LLM work: you can beg a model to do the right thing in the system prompt, but shaping the input so the wrong thing is harder to represent is almost always more reliable.

---

## Reproducing this

```bash
# Learn voice from your last 30 sent messages
sift learn-voice --limit 30

# Generate a brief with learned voice
sift brief --source gmail --limit 5

# For the A/B, clear the cache and force DEFAULT_VOICE for comparison
sift cache-clear voice_profiles
sift brief --source gmail --limit 5 --no-cache
```

Voice profiles are cached for seven days per user email; re-run `sift learn-voice --force` to refresh sooner.
