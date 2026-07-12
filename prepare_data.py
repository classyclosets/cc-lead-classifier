"""
prepare_data.py — Build training datasets from public email spam corpora
and your own labeled contact form submissions.

Output format: JSONL with `messages` array (system → user → assistant).
This matches the HuggingFace chat template format expected by train.py.
"""

import json
import csv
import re
import argparse
from pathlib import Path

SYSTEM_PROMPT = """You are a lead qualification assistant for a home services company. Analyze contact form submissions and classify them as VALID_LEAD or SPAM.

VALID_LEAD: A real person asking about the company's services (custom closets, wall beds, home organization, garage storage, home offices, pantries). They mention specific needs, rooms, projects, or ask for quotes. Even short inquiries like "need a quote" count.

SPAM: Marketing pitches, SEO offers, web design solicitations, guest post requests, link-building offers, unsolicited B2B sales pitches, gibberish, test submissions, or anyone selling TO the company rather than buying FROM it.

Respond with the classification on the first line, then a 1-2 sentence explanation of why."""


def make_chat_entry(user_text, label, label_detail=""):
    """Create one chat-format training example."""
    label_text = f"{label}\n\n{label_detail}" if label_detail else label
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": label_text},
        ]
    }


def from_labelled_csv(csv_path, text_col="email_text", label_col="label"):
    """
    Import from a two-column CSV: text, label.
    `label` should be 'spam' or 'ham' (or 'valid_lead').
    """
    entries = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            text = row[text_col]
            raw_label = row[label_col].strip().lower()
            if raw_label in ("spam",):
                entries.append(make_chat_entry(
                    f"Classify this content as VALID_LEAD or SPAM.\n\nEmail:\n{text}",
                    "SPAM",
                ))
            elif raw_label in ("ham", "valid_lead", "valid"):
                entries.append(make_chat_entry(
                    f"Classify this content as VALID_LEAD or SPAM.\n\nEmail:\n{text}",
                    "VALID_LEAD",
                ))
    return entries


def from_form_submissions(jsonl_path):
    """
    Import from a JSONL of contact form submissions.
    Each line: {"name": ..., "email": ..., "message": ..., "spam": true/false}

    Scrubs emails and phone numbers from the training text.
    """
    import re
    entries = []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            parts = []
            if row.get("name"):
                parts.append(f"Name: {row['name']}")
            if row.get("email"):
                parts.append(f"Email: {row['email']}")
            if row.get("phone"):
                parts.append(f"Phone: {row['phone']}")
            if row.get("message"):
                parts.append(f"Message: {row['message']}")

            submission = "\n".join(parts)
            is_spam = row.get("spam", False)
            label = "SPAM" if is_spam else "VALID_LEAD"
            detail = row.get("label_detail", "")

            entries.append(make_chat_entry(
                f"Classify this contact form submission:\n\n{submission}",
                label,
                detail,
            ))
    return entries


def split_train_eval(entries, eval_frac=0.10, seed=42):
    """Deterministic train/eval split."""
    import random
    rng = random.Random(seed)
    rng.shuffle(entries)
    split_idx = int(len(entries) * (1 - eval_frac))
    return entries[:split_idx], entries[split_idx:]


def redact_pii(entries):
    """
    Replace personally identifiable information with placeholders.
    Operates on both user and assistant message content.
    """
    for entry in entries:
        for msg in entry["messages"]:
            text = msg["content"]
            # Emails
            text = re.sub(r'[\w.+-]+@[\w-]+\.[\w.-]+', '[EMAIL]', text)
            # Phone numbers (various formats)
            text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
            text = re.sub(r'\b\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
            # Name field (Name: followed by anything until newline)
            text = re.sub(r'Name: .+', 'Name: [NAME]', text)
            msg["content"] = text
    return entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare training data for lead classifier LoRA."
    )
    parser.add_argument(
        "--from-csv", type=str,
        help="Path to a labelled CSV (text_col, label_col)."
    )
    parser.add_argument(
        "--from-forms", type=str,
        help="Path to a JSONL of contact form submissions."
    )
    parser.add_argument(
        "--out-dir", type=str, default="data",
        help="Output directory (default: data/)."
    )
    parser.add_argument(
        "--eval-frac", type=float, default=0.10,
        help="Fraction held out for eval (default: 0.10)."
    )
    parser.add_argument(
        "--redact", action="store_true",
        help="Replace PII (names, emails, phones) with [NAME]/[EMAIL]/[PHONE] placeholders."
    )
    args = parser.parse_args()

    all_entries = []

    if args.from_csv:
        print(f"Loading CSV: {args.from_csv}")
        all_entries.extend(from_labelled_csv(args.from_csv))

    if args.from_forms:
        print(f"Loading forms: {args.from_forms}")
        all_entries.extend(from_form_submissions(args.from_forms))

    if not all_entries:
        print("ERROR: No data sources provided. Use --from-csv or --from-forms.")
        exit(1)

    print(f"Total examples: {len(all_entries)}")

    train, eval_data = split_train_eval(all_entries, eval_frac=args.eval_frac)
    print(f"Train: {len(train)}  |  Eval: {len(eval_data)}")

    if args.redact:
        print("Redacting PII...")
        train = redact_pii(train)
        eval_data = redact_pii(eval_data)
        print("Redaction complete.")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{args.out_dir}/train.jsonl", "w") as f:
        for entry in train:
            f.write(json.dumps(entry) + "\n")

    with open(f"{args.out_dir}/eval.jsonl", "w") as f:
        for entry in eval_data:
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote {args.out_dir}/train.jsonl and {args.out_dir}/eval.jsonl")