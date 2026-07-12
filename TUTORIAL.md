# Tutorial: Build a Lead Form Spam Classifier with QLoRA

**Goal:** Fine-tune Llama 3.2 3B to classify contact form submissions as `VALID_LEAD` or `SPAM`.

**Time:** ~30 minutes on a single GPU.

**Cost:** ~$0.05 on rented hardware.

**Output:** A 97MB LoRA adapter that runs inference on HuggingFace's free tier.

---

## Prerequisites

- Python 3.10+
- CUDA-capable GPU with ≥12GB VRAM (RTX 3080 or better)
- HuggingFace account (free)

If you don't have a GPU, rent one:
- [Vast.ai](https://vast.ai) — RTX 3080 ≈ $0.08/hr
- [RunPod](https://runpod.io) — RTX 4090 ≈ $0.34/hr

---

## Step 1: Clone and Install

```bash
git clone https://github.com/classyclosets/cc-lead-classifier.git
cd cc-lead-classifier

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (takes ~2 min)
pip install -r requirements.txt
```

Expected output: successful install with no errors.

---

## Step 2: Prepare Training Data

You need labeled examples in chat format. Each example is a JSON object with a `messages` array:

```json
{
  "messages": [
    {"role": "system", "content": "You are a lead qualification assistant..."},
    {"role": "user", "content": "Classify this contact form submission:\n\nName: [NAME]\nEmail: [EMAIL]\nPhone: [PHONE]\nMessage: I need a quote for a walk-in closet"},
    {"role": "assistant", "content": "VALID_LEAD\n\nSpecific service request from a real person."}
  ]
}
```

### Option A: Use your own labeled data

If you have a CSV with `email_text` and `label` columns (where label is `spam` or `ham`):

```bash
python prepare_data.py --from-csv your_data.csv --out-dir data
```

If you have a JSONL of contact form submissions with a `spam` boolean field:

```bash
python prepare_data.py --from-forms your_forms.jsonl --out-dir data
```

### Option B: Use public datasets

These HuggingFace datasets are ready to use. Load them directly in `train.py` or convert with `prepare_data.py`:

- `VoltageVagabond/spam-email-dataset` — 4K chat-format examples with reasoning chains
- `sohamchougule-07/EnronSpam` — 31K emails in instruction format

```bash
# Example: download EnronSpam and convert
python -c "
from datasets import load_dataset
ds = load_dataset('sohamchougule-07/EnronSpam')
# Save to CSV for prepare_data.py
ds['train'].to_csv('enron_spam.csv', index=False)
"
python prepare_data.py --from-csv enron_spam.csv --out-dir data
```

### Verify the data

```bash
wc -l data/train.jsonl data/eval.jsonl
# Should show: train examples + eval examples
```

Sample data is provided in `data/sample_train.jsonl` and `data/sample_eval.jsonl` for format reference.

---

## Step 3: Train

```bash
python train.py
```

### What happens during training

1. **Load** — Downloads Llama 3.2 3B Instruct in 4-bit (~1.8GB)
2. **Apply LoRA** — Attaches trainable rank-16 adapters (97M params / 3.2B total = 3%)
3. **Train** — 2 epochs, batch size 2 × gradient accumulation 4 = effective batch 8
4. **Save** — Writes adapter to `outputs/adapter/`

### Expected output

```
Loading base model...
Applying LoRA adapter...
Loading datasets...
Train: 1574 examples  |  Eval: 86 examples
Starting training...
{'loss': 0.42, 'learning_rate': 1.8e-4, 'epoch': 1.0}
{'loss': 0.12, 'learning_rate': 2.0e-5, 'epoch': 1.5}
{'eval_loss': 0.08, ...}
...
Saving adapter...
Adapter saved to outputs/adapter
Done.
```

### Training time

| GPU | Examples | Time |
|-----|----------|------|
| RTX 3080 (10GB) | 1,574 | ~27 min |
| RTX 4090 (24GB) | 1,574 | ~12 min |
| A100 (40GB) | 1,574 | ~8 min |

### Hyperparameters

| Param | Value | Why |
|-------|-------|-----|
| `r` (rank) | 16 | Good enough for classification. Increase to 32-64 if training on >5K examples. |
| `alpha` | 32 | Scales adapter contribution. 2× rank is standard. |
| `dropout` | 0.05 | Prevents overfitting on small datasets. |
| `learning_rate` | 2e-4 | Higher than full fine-tuning since base weights are frozen. |
| `epochs` | 2 | Classification converges fast. Watch eval loss — stop if it rises. |
| `target_modules` | All 7 | Full LoRA coverage. For very small datasets (<500 examples), reduce to `q_proj, v_proj` only. |

---

## Step 4: Evaluate

```bash
python -c "
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset

# Load adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/Llama-3.2-3B-Instruct',
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    lora_alpha=32,
    lora_dropout=0.05,
)
model.load_adapter('outputs/adapter')

# Eval
eval_set = load_dataset('json', data_files={'eval': 'data/eval.jsonl'})['eval']
correct, total = 0, 0

for example in eval_set:
    text = tokenizer.apply_chat_template(example['messages'][:2], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=60, temperature=0.1, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    predicted = response.strip().split('\n')[0]
    actual = example['messages'][-1]['content'].strip().split('\n')[0]
    if predicted == actual:
        correct += 1
    total += 1

print(f'Accuracy: {correct}/{total} = {correct/total*100:.1f}%')
"
```

### Expected results (this config)

```
Accuracy: 84/86 = 97.7%
```

---

## Step 5: Push to HuggingFace Hub

### Push the adapter (LoRA weights only)

```bash
huggingface-cli login
# Enter your HF token

python -c "
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/Llama-3.2-3B-Instruct',
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    lora_alpha=32, lora_dropout=0.05,
)
model.load_adapter('outputs/adapter')

# Push to your namespace
model.push_to_hub('YOUR_USERNAME/cc-lead-classifier')
tokenizer.push_to_hub('YOUR_USERNAME/cc-lead-classifier')
"
```

### Option: Merge and push (full model, no separate adapter needed)

```bash
python -c "
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/Llama-3.2-3B-Instruct',
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    lora_alpha=32, lora_dropout=0.05,
)
model.load_adapter('outputs/adapter')

# Merge LoRA into base weights
merged = model.merge_and_unload()

# Push merged model
merged.push_to_hub('YOUR_USERNAME/cc-lead-classifier-merged')
tokenizer.push_to_hub('YOUR_USERNAME/cc-lead-classifier-merged')
"
```

**Adapter vs merged:**
- Adapter: 97MB. Requires base model at inference time. More flexible.
- Merged: 6.4GB. Self-contained. Works with HF Inference API directly.

---

## Step 6: Deploy Inference

### Option A: HuggingFace Serverless Inference API (free tier)

The merged model on HuggingFace Hub automatically gets a serverless inference endpoint.

```javascript
// classify-lead.js — Node.js integration example
const HF_API_URL = 'https://api-inference.huggingface.co/models/YOUR_USERNAME/cc-lead-classifier-merged';

async function classifyLead(fields) {
  const prompt = `<|system|>
You are a lead qualification assistant. Classify submissions as VALID_LEAD or SPAM.
VALID_LEAD: Real person asking about services (closets, organization, etc).
SPAM: Marketing pitches, SEO offers, gibberish, B2B solicitations.
Respond with classification on first line.</s>
<|user|>
Classify this contact form submission:

Name: ${fields.name}
Email: ${fields.email}
Message: ${fields.message}</s>
<|assistant|>
`;

  const response = await fetch(HF_API_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${process.env.HF_API_TOKEN}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      inputs: prompt,
      parameters: { max_new_tokens: 60, temperature: 0.1, return_full_text: false },
    }),
    signal: AbortSignal.timeout(8000),
  });

  const result = await response.json();
  const text = Array.isArray(result) ? result[0]?.generated_text || '' : '';
  return text.trim().split('\n')[0].startsWith('SPAM') ? 'SPAM' : 'VALID_LEAD';
}
```

Full production example: `classify-lead.js` in this repo.

### Option B: Self-hosted with Ollama

```bash
# Convert to GGUF (requires llama.cpp)
python -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/Llama-3.2-3B-Instruct',
    max_seq_length=2048, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    lora_alpha=32, lora_dropout=0.05,
)
model.load_adapter('outputs/adapter')
merged = model.merge_and_unload()
merged.save_pretrained_gguf('cc-lead-classifier', tokenizer, quantization_method='q4_k_m')
"

# Create Ollama model
echo 'FROM ./cc-lead-classifier-unsloth.Q4_K_M.gguf' > Modelfile
ollama create cc-lead-classifier -f Modelfile

# Run inference
ollama run cc-lead-classifier "Classify: Name: Jane\nEmail: jane@email.com\nMessage: I need a closet quote"
```

---

## FAQ

### My eval loss goes up after epoch 1 — what do I do?

Your model is overfitting. Reduce epochs to 1, increase dropout to 0.1, or reduce rank to 8. For very small datasets (<500 examples), use `target_modules=['q_proj','v_proj']` only.

### Can I train on a free Colab GPU?

Sometimes. The T4 (16GB) can handle this config, but Colab may disconnect during the 30-minute run. Vast.ai RTX 3080 at $0.08/hr is more reliable and costs pennies.

### How do I add new spam types?

Add more examples to `data/train.jsonl` in the same format and re-run `train.py`. The LoRA adapts quickly — 50-100 new examples is plenty.

### What if my model classifies everything as VALID_LEAD?

Your dataset is probably imbalanced. Aim for at least 40/60 split between spam and valid. 50/50 is ideal.

### Can I use a different base model?

Yes. Change `MODEL_NAME` in `train.py` to any Llama/Mistral/Qwen model supported by Unsloth. The LoRA configuration works across all of them.

### Why classification via text generation instead of a classification head?

A classification head (adding a linear layer on top) is faster at inference but requires the model to be loaded and running somewhere. Text generation means your model works on HuggingFace's serverless API with zero infrastructure. The tradeoff is ~1-2 seconds per inference vs milliseconds.

---

## File Reference

| File | Purpose |
|------|---------|
| `train.py` | Single-script training loop |
| `prepare_data.py` | Convert CSVs and form JSONL to training format |
| `classify-lead.js` | Production Node.js integration (Cloudflare Worker / Vercel) |
| `requirements.txt` | Python dependencies |
| `data/sample_train.jsonl` | 16 training examples (format reference) |
| `data/sample_eval.jsonl` | 5 eval examples (format reference) |

## License

MIT — use this for whatever you want.