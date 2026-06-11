const https = require('https');
const fs = require('fs');
const path = require('path');

const apiKey = "AIzaSyCQOdynrywuy2-Et-uKlfyB08IzJN5lfas";
const model = "gemini-3-flash";

// Helper to load prompts.json
function loadPrompts() {
  const promptsPath = path.join(__dirname, 'src', 'prompts', 'prompts.json');
  return JSON.parse(fs.readFileSync(promptsPath, 'utf8'));
}

// Map emotion to visual asset
const MIMO_ASSETS = [
  "Alert", "Angry", "Approved", "Celebrate", "Chill", "Cooking", "Cool",
  "Determined", "Error", "Excited", "Giggle", "Happy", "Hello", "Loading",
  "Love", "Proud", "Relax", "Sad", "Sleepy", "Sassy", "Shopping", "Travel",
  "Sorry", "Success", "Taunting", "Thankful", "Thinking", "Working", "Worried"
];

function coerceMimoAsset(raw) {
  if (!raw) return null;
  const trimmed = raw.trim();
  if (MIMO_ASSETS.includes(trimmed)) return trimmed;
  for (const asset of MIMO_ASSETS) {
    if (asset.toLowerCase() === trimmed.toLowerCase()) return asset;
  }
  return null;
}

function callGemini(systemPrompt, userPrompt) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify({
      systemInstruction: {
        parts: [
          {
            text: systemPrompt
          }
        ]
      },
      contents: [
        {
          parts: [
            {
              text: userPrompt
            }
          ]
        }
      ],
      generationConfig: {
        responseMimeType: "application/json",
        responseSchema: {
          type: "object",
          properties: {
            response: { type: "string" },
            mimo_emotion: { type: "string" }
          },
          required: ["response", "mimo_emotion"]
        }
      }
    });

    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;

    const options = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(payload)
      }
    };

    const req = https.request(url, options, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => body += chunk);
      res.on('end', () => {
        if (res.statusCode !== 200) {
          reject(new Error(`API Error (status ${res.statusCode}): ${body}`));
          return;
        }
        try {
          const parsed = JSON.parse(body);
          resolve(parsed);
        } catch (e) {
          reject(e);
        }
      });
    });

    req.on('error', (e) => reject(e));
    req.write(payload);
    req.end();
  });
}

async function runTest() {
  const prompts = loadPrompts();

  // Test Case 1: Expense with dan_doi persona
  const slangPool1 = prompts.emotions.dan_doi.slang_pool || [];
  const slangInstruction1 = slangPool1.length ? `[SLANG GỢI Ý]: Bạn có thể sử dụng các từ slang sau để phản hồi tự nhiên: ${slangPool1.join(', ')}.` : '';

  const systemPrompt1 = [
    prompts.emotions.dan_doi.system,
    prompts.common.style,
    prompts.common.response_rules,
    prompts.common.context_diversity_rule,
    slangInstruction1
  ].filter(Boolean).join(' ');

  const assetListStr = MIMO_ASSETS.join(', ');
  const outputInstruction = `\nYÊU CẦU ĐẦU RA JSON: Trả về đúng 2 trường:\n1. "response": Câu thoại (tối đa 30 từ).\n2. "mimo_emotion": Dựa vào ngữ nghĩa của đầu vào và reponse, hãy chọn ĐÚNG 1 tên trong danh sách (PascalCase): ${assetListStr}.`;

  const userPrompt1 = [
    prompts.emotions.dan_doi.user,
    prompts.common.response_rules,
    prompts.common.record_expense_rules,
    "LOẠI GIAO DỊCH: Chi tiêu (Expense). Phản hồi phải nói tiền RA / chi / mua — TUYỆT ĐỐI KHÔNG nói thu nhập, lương về, tiền vào ví. Có thể nhắc ngân sách nếu CONTEXT_META có cảnh báo.",
    "Món hoặc hạng mục: trà sữa. Số tiền: 45,000đ. Kiểu cảnh báo (chỉ Expense): NONE.",
    "CONTEXT_META (phối hợp ≥1 yếu tố):\n- Thời điểm: chiều_tối\n- Sức khoẻ ví: ⚠️ Cẩn thận\n- Loại giao dịch: Chi tiêu (Expense)\n- Ngân sách còn lại: 1,200,000đ",
    outputInstruction
  ].filter(Boolean).join(' ');

  console.log("==========================================");
  console.log("TEST CASE 1: mua trà sữa hết 45k (dan_doi)");
  console.log("==========================================");
  console.log("\n>>> System Prompt:\n", systemPrompt1);
  console.log("\n>>> User Prompt:\n", userPrompt1);

  try {
    const res = await callGemini(systemPrompt1, userPrompt1);
    console.log("\n>>> Raw API Response:", JSON.stringify(res, null, 2));
    const textRes = res.candidates[0].content.parts[0].text;
    const parsedJson = JSON.parse(textRes);
    console.log("\n>>> Parsed Gemini Output JSON:", parsedJson);
    console.log(">>> Resolved mimo_emotion:", coerceMimoAsset(parsedJson.mimo_emotion));
  } catch (e) {
    console.error("Test Case 1 failed:", e.message);
  }

  // Test Case 2: Income with vui persona
  const slangPool2 = prompts.emotions.vui.slang_pool || [];
  const slangInstruction2 = slangPool2.length ? `[SLANG GỢI Ý]: Bạn có thể sử dụng các từ slang sau để phản hồi tự nhiên: ${slangPool2.join(', ')}.` : '';

  const systemPrompt2 = [
    prompts.emotions.vui.system,
    prompts.common.style,
    prompts.common.response_rules,
    prompts.common.context_diversity_rule,
    slangInstruction2
  ].filter(Boolean).join(' ');

  const userPrompt2 = [
    prompts.emotions.vui.user,
    prompts.common.response_rules,
    prompts.common.record_income_rules,
    "LOẠI GIAO DỊCH: Thu nhập (Income). Phản hồi phải nói tiền VÀO ví / thu / nhận — TUYỆT ĐỐI KHÔNG nói chi tiêu, mua, tiêu xài, ngân sách cạn, ét ô ét vì mất tiền.",
    "Món hoặc hạng mục: lương. Số tiền: 15,000,000đ. Kiểu cảnh báo (chỉ Expense): NONE.",
    "CONTEXT_META (phối hợp ≥1 yếu tố):\n- Thời điểm: buổi_trưa\n- Hôm nay là ngày nhận lương 💸\n- Loại giao dịch: Thu nhập (Income)",
    outputInstruction
  ].filter(Boolean).join(' ');

  console.log("\n\n==========================================");
  console.log("TEST CASE 2: lương về 15tr (vui)");
  console.log("==========================================");
  console.log("\n>>> System Prompt:\n", systemPrompt2);
  console.log("\n>>> User Prompt:\n", userPrompt2);

  try {
    const res = await callGemini(systemPrompt2, userPrompt2);
    console.log("\n>>> Raw API Response:", JSON.stringify(res, null, 2));
    const textRes = res.candidates[0].content.parts[0].text;
    const parsedJson = JSON.parse(textRes);
    console.log("\n>>> Parsed Gemini Output JSON:", parsedJson);
    console.log(">>> Resolved mimo_emotion:", coerceMimoAsset(parsedJson.mimo_emotion));
  } catch (e) {
    console.error("Test Case 2 failed:", e.message);
  }
}

runTest();
