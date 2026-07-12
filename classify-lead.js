/**
 * classify-lead.js — Production integration example for CC Lead Classifier.
 *
 * This is a standalone template. Drop it into any Node.js project,
 * set HF_API_TOKEN in your env, and call classifyLead().
 *
 * The model runs on HuggingFace's serverless Inference API.
 * At low volume (< 1000 calls/day) this is inside the free tier.
 *
 * Design: fail-open. If the model is unavailable, the lead passes through.
 * Never block a real lead because your classifier is cold-starting.
 */

const HF_API_URL = 'https://api-inference.huggingface.co/models/YOUR_USERNAME/cc-lead-classifier-merged';

const SYSTEM_PROMPT = `You are a lead qualification assistant. Analyze contact form submissions and classify them as VALID_LEAD or SPAM.

VALID_LEAD: A real person asking about the company's products or services. They mention specific needs, projects, or ask for quotes.

SPAM: Marketing pitches, SEO offers, web design solicitations, guest post requests, link-building offers, unsolicited B2B sales pitches, gibberish, or test submissions.

Respond with the classification on the first line, then a 1-sentence reason.`;

function buildPrompt(fields) {
  const parts = [];
  if (fields.name) parts.push(`Name: ${fields.name}`);
  if (fields.email) parts.push(`Email: ${fields.email}`);
  if (fields.phone) parts.push(`Phone: ${fields.phone}`);
  if (fields.message) parts.push(`Message: ${fields.message}`);

  const submission = parts.join('\n');
  return `<|system|>\n${SYSTEM_PROMPT}</s>\n<|user|>\nClassify this contact form submission:\n\n${submission}</s>\n<|assistant|>\n`;
}

/**
 * Classify a lead as VALID_LEAD or SPAM.
 *
 * @param {Object} fields — { name, email, phone, message }
 * @returns {Object} — { isSpam: boolean, classification: string, reason: string }
 */
export async function classifyLead(fields) {
  const hfToken = process.env.HF_API_TOKEN;
  if (!hfToken) {
    console.warn('[Classifier] HF_API_TOKEN not set — skipping');
    return { isSpam: false, classification: null, reason: 'no_token' };
  }

  try {
    const response = await fetch(HF_API_URL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${hfToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        inputs: buildPrompt(fields),
        parameters: {
          max_new_tokens: 60,
          temperature: 0.1,
          return_full_text: false,
        },
      }),
      signal: AbortSignal.timeout(8000),
    });

    if (!response.ok) {
      console.warn('[Classifier] API error:', response.status);
      return { isSpam: false, classification: null, reason: `api_status_${response.status}` };
    }

    const result = await response.json();
    const text = Array.isArray(result) ? result[0]?.generated_text || '' : '';

    if (!text) {
      console.warn('[Classifier] Empty response');
      return { isSpam: false, classification: null, reason: 'empty_response' };
    }

    const classification = text.trim().split('\n')[0].trim();
    const isSpam = classification.startsWith('SPAM');

    return { isSpam, classification, reason: 'model_classified' };

  } catch (error) {
    console.error('[Classifier] Error:', error.message);
    // Fail open — never block a lead because the classifier is down
    return { isSpam: false, classification: null, reason: 'error' };
  }
}

// ── Example usage ──────────────────────────────────────────────────────────
//
//   import { classifyLead } from './classify-lead.js';
//
//   const result = await classifyLead({
//     name: 'Jane Smith',
//     email: 'jane@example.com',
//     message: 'I need a quote for a walk-in closet',
//   });
//
//   if (result.isSpam) {
//     // Flag or discard this lead
//   }