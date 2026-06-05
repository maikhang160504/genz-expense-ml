{
  "contents": [
    {
      "parts": [
        {
          "text": "Dữ liệu fusion: { 'item': 'Trà sữa', 'amount': '60.000đ', 'category': 'Ăn uống' }. Ngữ cảnh: Người dùng đang vượt ngân sách tháng 10%. Hãy tạo một story ngắn, bắt trend, hài hước."
        }
      ]
    }
  ],
  "generationConfig": {
    "temperature": 0.8, 
    "topK": 40,
    "topP": 0.95,
    "maxOutputTokens": 512,
    "responseMimeType": "application/json",
    "responseSchema": {
      "type": "object",
      "properties": {
        "response": {"type": "string"},
        "mimo_emotion": {"type": "string"}
      },
      "required": ["response", "mimo_emotion"]
    },
    "thinkingConfig": {
      "thinkingBudget": 0
    }
  },
  "systemInstruction": "Placeholder — runtime ghi đè bằng nlg_prompt.system."
}


{
  "response": "60k trà sữa thì 'ngoan xinh yêu' đấy, nhưng ví tiền của bạn đang ở trạng thái 'ét ô ét' vì lố 10% ngân sách rồi nhé! 🧋",
  "mimo_emotion": "Worried",
  "usage_metadata": {
    "prompt_token_count": 85,
    "candidates_token_count": 42,
    "total_token_count": 127
  }
}