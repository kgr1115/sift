# Gmail Setup Walkthrough

One-time setup to let Sift read your inbox via the Gmail API. The whole thing takes 10-15 minutes.

We only request **read** and **compose-draft** scopes — the app never sends, archives, deletes, or modifies your actual inbox state. (Drafts land in your Gmail Drafts folder; you send them yourself.)

---

## 1. Create a Google Cloud project

1. Go to <https://console.cloud.google.com/> and sign in with your Google account.
2. In the top bar, click the project picker → **New Project**.
3. Name it something like `sift`. No organization. Click **Create**.
4. Wait for it to provision, then select it in the project picker.

## 2. Enable the Gmail API

1. From the left nav: **APIs & Services → Library**.
2. Search for **Gmail API**. Click it → **Enable**.

## 3. Configure the OAuth consent screen

Because this is a personal app running on your machine, you can keep it in **Testing** mode forever — no verification required.

1. Left nav: **APIs & Services → OAuth consent screen**.
2. User Type: **External**. Click **Create**.
3. **App information:**
   - App name: `Sift` (or whatever you like)
   - User support email: your address
   - Developer contact email: your address
4. **Scopes:** skip for now — we'll add them programmatically. Click **Save and Continue**.
5. **Test users:** add your own Google email. Click **Save and Continue**.
6. **Summary:** click **Back to Dashboard**.

## 4. Create an OAuth client

1. Left nav: **APIs & Services → Credentials**.
2. Click **Create Credentials → OAuth client ID**.
3. Application type: **Desktop app**.
4. Name: `Sift desktop`. Click **Create**.
5. Click **Download JSON** on the resulting credential. Save it as `credentials.json` in the root of this repo (the same folder as `pyproject.toml`).

> **`credentials.json` is in `.gitignore`.** Do not commit it.

## 5. First run — authorize the app

Authorize once with a dedicated command:

```bash
sift auth
```

Your browser opens to Google's OAuth screen. Sign in, review the requested scopes (you'll see **Read your email** and **Manage drafts and send emails** — we only use the draft half of the latter), and click **Allow**. The flow saves `token.json` next to `credentials.json` (also gitignored). Subsequent runs reuse the token silently.

Once authenticated you can:

```bash
# Pull recent threads and print the morning brief
sift brief --source gmail --limit 25

# Same, but also push AI drafts into your Gmail Drafts folder for review
sift brief --source gmail --push

# End-to-end "morning routine": classify, draft, push
sift push-drafts --limit 25

# Re-auth if your token expires or scopes change
sift auth --force
```

### What scopes Sift actually requests

| Scope | What it grants | Why Sift needs it |
| --- | --- | --- |
| `gmail.readonly` | Read messages, threads, labels | Fetching inbox for classification + drafting |
| `gmail.compose` | Create / modify Drafts (not send) | Landing AI-written replies in your Drafts folder |

Sift does **not** request `gmail.send`, `gmail.modify`, or any delete/archive scopes. You always send drafts yourself from the Gmail UI after reviewing them.

If you ever want to revoke access, go to <https://myaccount.google.com/permissions> and remove "Sift".

---

## Troubleshooting

**"This app isn't verified" warning on the OAuth screen.**
Expected. Because the app is in Testing mode and you're the only test user, Google flags it. Click **Advanced → Go to Sift (unsafe)**. It's safe — it's your own app running on your machine.

**`invalid_grant` when running later.**
Your refresh token likely expired (Testing-mode tokens expire after 7 days). Run `sift auth --force` to re-authenticate, or delete `token.json` and retry any Gmail command.

**"No cached Gmail token.json" when running tests.**
Expected — the integration smoke tests in `tests/test_gmail_client.py` are gated on a cached token and will skip if you haven't run `sift auth` yet. The pure-unit tests (MIME parser, header helpers, reply-MIME builder) still run.

**Rate limits.**
The Gmail API gives personal accounts a generous quota. If you hit limits, use the `--limit` flag on the fetch CLI to cap the number of threads per run.
