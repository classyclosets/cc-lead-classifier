import { createClient } from '@supabase/supabase-js';
import { applySpamStatus } from '../../../utils/adminLeadActions';

const HF_API_URL = 'https://api-inference.huggingface.co/models/Overup/cc-lead-classifier-merged';

const SYSTEM_PROMPT = `You are a lead qualification assistant for a home services company. Analyze contact form submissions and classify them as VALID_LEAD or SPAM.

VALID_LEAD: A real person asking about custom closets, wall beds, home organization, garage storage, home offices, or pantries. They mention specific needs, rooms, projects, or ask for quotes and consultations.

SPAM: Marketing pitches, SEO offers, web design solicitations, guest post requests, link-building offers, unsolicited B2B sales pitches, gibberish, or test submissions.

Respond with the classification on the first line, then a 1-sentence reason.`;

function buildPrompt(fields) {
  const parts = [];
  const name = `${fields.first_name || ''} ${fields.last_name || ''}`.trim();
  if (name) parts.push(`Name: ${name}`);
  if (fields.email) parts.push(`Email: ${fields.email}`);
  if (fields.phone) parts.push(`Phone: ${fields.phone}`);
  if (fields.showroom) parts.push(`Showroom: ${fields.showroom}`);
  const loc = [fields.city, fields.state].filter(Boolean).join(', ');
  if (loc) parts.push(`Location: ${loc}`);
  if (fields.message) parts.push(`Message: ${fields.message}`);

  const submission = parts.join('\n');
  return `<|system|>\n${SYSTEM_PROMPT}</s>\n<|user|>\nClassify this contact form submission:\n\n${submission}</s>\n<|assistant|>\n`;
}

/**
 * Classify a lead using the HuggingFace model and auto-mark spam.
 * Fail-open: if the model is unavailable, the lead passes through.
 */
async function classifyAndAutoMark(supabase, fields, visitorId) {
  const hfToken = process.env.HF_API_TOKEN;
  if (!hfToken) {
    console.warn('[Classify Lead] HF_API_TOKEN not set — skipping classification');
    return { is_spam: false, classification: null, reason: 'no_token' };
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
        parameters: { max_new_tokens: 60, temperature: 0.1, return_full_text: false },
      }),
      signal: AbortSignal.timeout(8000), // 8s timeout — don't block the form
    });

    if (!response.ok) {
      console.warn('[Classify Lead] HF API error:', response.status);
      return { is_spam: false, classification: null, reason: `hf_status_${response.status}` };
    }

    const result = await response.json();
    const text = Array.isArray(result) ? result[0]?.generated_text || '' : '';

    if (!text) {
      console.warn('[Classify Lead] Empty response from HF');
      return { is_spam: false, classification: null, reason: 'empty_response' };
    }

    const classification = text.trim().split('\n')[0].trim();
    const isSpam = classification.startsWith('SPAM');

    console.log('[Classify Lead] Result:', { visitorId, classification, isSpam });

    // Auto-mark spam visitors
    if (isSpam && visitorId) {
      try {
        await applySpamStatus(supabase, {
          visitor_id: visitorId,
          is_spam: true,
          marked_by: 'ai-classifier',
        });
        console.log('[Classify Lead] Auto-marked as spam:', visitorId);
      } catch (markError) {
        console.error('[Classify Lead] Failed to auto-mark spam:', markError);
      }
    }

    return {
      is_spam: isSpam,
      classification,
      reason: 'model_classified',
    };
  } catch (error) {
    console.error('[Classify Lead] Error:', error.message);
    // Fail open — never block a lead because the classifier is down
    return { is_spam: false, classification: null, reason: `error:${error.message}` };
  }
}

/**
 * POST /api/admin/classify-lead
 * Body: { visitor_id, first_name, last_name, email, phone, message, showroom, city, state }
 * Returns: { is_spam, classification, reason }
 */
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const {
    visitor_id,
    first_name,
    last_name,
    email,
    phone,
    message,
    showroom,
    city,
    state,
  } = req.body || {};

  if (!email || !visitor_id) {
    return res.status(400).json({ error: 'Missing email or visitor_id' });
  }

  try {
    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL,
      process.env.SUPABASE_SERVICE_ROLE_KEY
    );

    const result = await classifyAndAutoMark(supabase, req.body, visitor_id);

    return res.status(200).json(result);
  } catch (error) {
    console.error('[Classify Lead] Handler error:', error);
    // Fail open
    return res.status(200).json({
      is_spam: false,
      classification: null,
      reason: 'handler_error',
    });
  }
}
