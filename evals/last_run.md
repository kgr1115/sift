# Latest Classifier Eval Run

**Overall accuracy:** 95.00% (38/40)

| Category     | Precision | Recall | F1    | Support |
|--------------|-----------|--------|-------|---------|
| urgent       | 0.83      | 1.00   | 0.91  |       5 |
| needs_reply  | 1.00      | 0.80   | 0.89  |      10 |
| fyi          | 0.92      | 1.00   | 0.96  |      11 |
| newsletter   | 1.00      | 1.00   | 1.00  |       8 |
| trash        | 1.00      | 1.00   | 1.00  |       6 |

## Misclassifications

- **t001** `needs_reply → urgent` (conf 0.85) — Quick question on the Q2 roadmap doc
- **t019** `needs_reply → fyi` (conf 0.72) — Looking forward to our chat