I trained a spam classifier on Llama 3.2 using QLoRA. It cost $0.05 and took 27 minutes.

Here's the architecture, the data, and the code.

—

THE PROBLEM

My company's contact forms get the usual noise — SEO pitches, guest post offers, bogus emails. Every one wastes someone's time in the CRM.

I wanted a model that could tell the difference between "I need a quote for a walk-in closet" and "Hello Sir/Madam I offer digital marketing services."

—

THE APPROACH

Standard approach: train a BERT classifier. Attach a classification head. Get 95%+ accuracy in milliseconds.

I didn't do that.

I did QLoRA fine-tuning on Llama 3.2 3B Instruct — a text generation model — and used it as a classifier via constrained generation.

Why?

Because a text-generation model runs on HuggingFace's serverless inference API. No GPU to manage. No container to orchestrate. Zero infrastructure. At my volume (5-20 leads/day), I'm inside the free tier forever.

The tradeoff: 1-2 seconds per inference instead of milliseconds. For a contact form, nobody notices.

—

THE SPECIFICS

Base model: Llama 3.2 3B Instruct, 4-bit quantized (bitsandbytes NF4)
Method: QLoRA (rank 16, alpha 32, all 7 attention + FFN projections)
Training data: 1,574 examples — 787 spam, 787 valid leads. Mix of public email spam corpora (Enron, VoltageVagabond) + my own labeled form submissions
Hardware: RTX 3080 on Vast.ai — $0.08/hr
Training time: 27 minutes (2 epochs, effective batch size 8)
Adapter size: 97 MB

—

RESULTS

Evaluated on 86 held-out leads from real form submissions:

• 97.7% overall accuracy
• SPAM recall: 93.8% (15/16)
• VALID_LEAD recall: 98.6% (69/70)

Two errors on 86 examples. Both edge cases I'd have gotten wrong manually.

—

DEPLOYMENT

Merged the adapter into the base model, pushed to HuggingFace Hub, wired into my Next.js form handler via the serverless inference API. Forward pass only, temperature 0.1, return the first generated token.

8-second timeout. Fail-open — if the model is down, the lead passes through unflagged. Never block a real lead because your classifier is cold-starting.

—

This is the thing that's undersold about LoRA fine-tuning:

You don't need a dataset of 100K examples. You don't need an A100. You don't need to be an ML researcher.

You need a labeled dataset of ~1,500 examples, a rented GPU for half an hour, and a HuggingFace account.

The repo has the full training script, data prep, evaluation, and deployment code. TUTORIAL.md is step-by-step — designed to be followed by Cursor, Copilot, or any AI coding agent.

Everything is in the first comment.