import axios from 'axios';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { text, mode } = req.body;

  if (!text || !mode) {
    return res.status(400).json({ error: 'Missing text or mode' });
  }

  let prompt = '';

  switch (mode) {
    case 'summarize':
      prompt = `Summarize the following text concisely: "${text}"`;
      break;
    case 'rewrite':
      prompt = `Rewrite the following text: "${text}"`;
      break;
    case 'expand':
      prompt = `Expand on the following text: "${text}"`;
      break;
    case 'simplify':
      prompt = `Simplify the following text: "${text}"`;
      break;
    case 'professional':
      prompt = `Make the following text sound more professional: "${text}"`;
      break;
    case 'casual':
      prompt = `Make the following text sound more casual: "${text}"`;
      break;
    default:
      prompt = `Process the following text: "${text}"`;
  }

  try {
    const response = await axios.post(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${process.env.GEMINI_API_KEY}`,
      {
        contents: [{ role: "user", parts: [{ text: prompt }] }]
      },
      {
        headers: { 'Content-Type': 'application/json' }
      }
    );

    const result = response.data?.candidates?.[0]?.content?.parts?.[0]?.text || 'No response from Gemini';
    return res.status(200).json({ result });

  } catch (err) {
    console.error("API error:", err.response?.data || err.message);
    
    if (err.response) {
      return res.status(500).json({ 
        error: 'Failed to process text with Gemini API',
        details: err.response.data
      });
    } else {
      return res.status(500).json({ 
        error: 'Backend error when trying to process text',
        message: err.message
      });
    }
  }
}
