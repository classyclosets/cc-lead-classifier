# CC Lead Classifier

A LoRA fine-tuned LLM that classifies contact form submissions as `VALID_LEAD` or `SPAM`.

**97.7% accuracy. 97MB adapter. $0.05 to train.**

---

## What This Is

A [QLoRA](https://arxiv.org/abs/2305.14314) fine-tune of [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) trained on 1,574 examples (787 spam, 787 valid leads).

The model reads a contact form submission and returns a classification:

```
Input: "Name: Jane | Email: jane@email.com | Message: I need a quote for a walk-in closet"
Output: "VALID_LEAD — Specific service request from a real person with clear intent."
```

```
Input: "Subject: Boost Your SEO Rankings | Message: We can position your brand above competitors..."
Output: "SPAM — Unsolicited marketing pitch from someone selling services, not buying."
```

---

## Why This Exists

Contact forms collect spam. SEO pitches, guest post offers, link-building solicitations — all of it lands in your CRM and wastes someone's time.

A classifier that catches this before it hits the CRM costs essentially nothing to build and run.

This repo contains everything needed to:

1. **Train** your own version on your data
2. **Understand** the full pipeline (data → training → eval → deploy)
3. **Deploy** inference via HuggingFace's free serverless API

---

## Quick Start

```bash
# 1. Install
git clone https://github.com/classyclosets/cc-lead-classifier.git
cd cc-lead-classifier
pip install -r requirements.txt

# 2. Prepare your data (see TUTORIAL.md for options)
python prepare_data.py --from-forms your_leads.jsonl --out-dir data

# 3. Train (~30 min on RTX 3080)
python train.py

# 4. Evaluate
# See TUTORIAL.md Step 4 for eval script
```

**Full tutorial:** [`TUTORIAL.md`](./TUTORIAL.md) — step-by-step instructions designed to be followed by humans and AI agents alike.

---

## Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | 97.7% (84/86) |
| SPAM Detection | 93.8% |
| VALID_LEAD Detection | 98.6% |
| Adapter Size | 97 MB |
| Training Time (RTX 3080) | 27 minutes |
| Training Cost (Vast.ai) | $0.05 |

---

## Training Configuration

| Param | Value |
|-------|-------|
| Base Model | Llama 3.2 3B Instruct (4-bit) |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| Dropout | 0.05 |
| Target Modules | q, k, v, o, gate, up, down |
| Epochs | 2 |
| Learning Rate | 2e-4 |
| Optimizer | AdamW 8-bit |
| Effective Batch Size | 8 |

---

## Deployment

### HuggingFace Serverless API (free)

```javascript
const response = await fetch(
  'https://api-inference.huggingface.co/models/YOUR_USERNAME/cc-lead-classifier-merged',
  {
    method: 'POST',
    headers: { Authorization: `Bearer ${HF_TOKEN}` },
    body: JSON.stringify({
      inputs: "Classify: Name: Jane\nEmail: jane@email.com\nMessage: I need a closet quote",
      parameters: { max_new_tokens: 60, temperature: 0.1 }
    })
  }
);
```

Full integration example: [`classify-lead.js`](./classify-lead.js)

---

## Files

| File | What It Does |
|------|-------------|
| `train.py` | Training script — load model, apply LoRA, run SFT |
| `prepare_data.py` | Convert your labeled data to training format |
| `classify-lead.js` | Production Node.js integration (Vercel / Cloudflare Worker) |
| `TUTORIAL.md` | Complete step-by-step guide |
| `requirements.txt` | Python dependencies |

---

## FAQ

**Can I use this with my own form data?** Yes. Label a few hundred examples and run `prepare_data.py`. The model adapts quickly — 200-300 examples is enough for decent results.

**Do I need a GPU?** For training, yes (≥12GB VRAM). For inference, no — the model runs on HuggingFace's free serverless tier or any CPU via Ollama + GGUF.

**Is the training data included?** Sample data only (scrubbed of PII). The training data used real contact form submissions which aren't public. See `prepare_data.py` for how to build your own or use public email spam datasets.

**Why classification via text generation?** It lets you use HuggingFace's serverless inference API with zero infrastructure. The tradeoff is ~1-2s per inference instead of milliseconds.

---

## License

MIT