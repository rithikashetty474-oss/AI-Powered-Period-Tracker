# Health Assistant Chatbot

A complete embedding-based chatbot system for menstrual wellness with vector search and LLM fallback.

## Features

✅ **Embedding-based Vector Search**: Uses sentence transformers to find the best matching Q&A pairs  
✅ **Fallback LLM Responses**: Rule-based intelligent responses when dataset doesn't match  
✅ **Safety First**: Includes medical disclaimers and avoids diagnosis  
✅ **Easy Integration**: Simple Flask API endpoint  
✅ **Optimized**: Cached embeddings for fast responses  

## Installation

```bash
pip install -r requirements.txt
```

The chatbot requires:
- `sentence-transformers` - For embeddings
- `torch` - PyTorch backend
- `numpy` - Vector operations

## Dataset Format

The chatbot expects `data/health_faq.csv` with the following format:

```csv
question,answer
What is a normal menstrual cycle?,A normal cycle is 21-35 days...
How long does a period last?,Most periods last 3-7 days...
```

## Usage

### Basic Usage

```python
from models.health_chatbot import get_chatbot

# Initialize chatbot (lazy loads on first call)
chatbot = get_chatbot()

# Get answer
result = chatbot.get_answer("What is a normal cycle length?")
print(result['answer'])
print(f"Confidence: {result['confidence']}")
print(f"Source: {result['source']}")  # 'dataset', 'keyword_match', or 'fallback_llm'
```

### API Endpoint

The chatbot is integrated into the main Flask app at `/api/chat`:

```bash
# POST request
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a normal period length?"}'

# Response
{
  "success": true,
  "reply": "Most periods last between 3 to 7 days...",
  "confidence": 0.85,
  "source": "dataset",
  "safety_disclaimer": "⚠️ Important: This information is for general wellness..."
}
```

### Standalone API Server

For a standalone API server, use `chatbot_api_example.py`:

```bash
python chatbot_api_example.py
```

This starts a server on port 5001 with `/chat` endpoint.

## Functions

### `load_dataset()`
Loads Q&A pairs from CSV file. Returns `(questions, answers)` lists.

### `embed_dataset(force_recompute=False)`
Creates embeddings for all questions. Caches results for faster subsequent loads.

### `get_answer(user_input, similarity_threshold=0.5)`
Main function to get answer:
- Tries embedding-based vector search first
- Falls back to keyword matching
- Uses rule-based LLM fallback if no match

Returns dictionary with:
- `answer`: The response text
- `confidence`: Similarity score (0.0-1.0)
- `source`: Where answer came from ('dataset', 'keyword_match', 'fallback_llm')
- `safety_disclaimer`: Medical disclaimer text

## Configuration

### Similarity Threshold
Adjust the minimum similarity for dataset matches:

```python
result = chatbot.get_answer("question", similarity_threshold=0.6)  # Higher = stricter matching
```

### Model Selection
Change the embedding model in `HealthChatbot.__init__()`:

```python
chatbot = HealthChatbot(model_name='all-mpnet-base-v2')  # Larger, more accurate
```

Available models:
- `all-MiniLM-L6-v2` (default) - Fast, lightweight
- `all-mpnet-base-v2` - More accurate, slower
- `paraphrase-MiniLM-L6-v2` - Good for Q&A

## Safety Features

- **Medical Disclaimers**: All responses include safety warnings
- **No Diagnosis**: Avoids medical diagnosis or treatment recommendations
- **General Wellness Only**: Focuses on general information and wellness tips
- **Healthcare Provider Guidance**: Encourages consulting professionals for medical concerns

## Performance

- **First Load**: ~5-10 seconds (downloads model, computes embeddings)
- **Subsequent Loads**: <1 second (uses cached embeddings)
- **Response Time**: <100ms per query (with cached embeddings)

## Frontend Integration

The chatbot works with the existing `ai_chat.html` frontend. The JavaScript already calls `/api/chat`:

```javascript
fetch('/api/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: userMessage})
})
.then(res => res.json())
.then(data => {
    if (data.success) {
        displayMessage(data.reply);
    }
});
```

## Troubleshooting

### Model Download Issues
If sentence-transformers fails to download:
- Check internet connection
- Model downloads automatically on first use (~80MB)
- Can manually download: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`

### Embeddings Not Working
If embeddings fail:
- Check `sentence-transformers` and `torch` are installed
- Chatbot automatically falls back to keyword matching
- Check console for error messages

### Low Confidence Scores
- Lower `similarity_threshold` (default 0.5)
- Add more Q&A pairs to dataset
- Use a larger embedding model

## File Structure

```
models/
  health_chatbot.py      # Main chatbot class
data/
  health_faq.csv         # Q&A dataset
  chatbot_embeddings.pkl # Cached embeddings
app.py                   # Flask app with /api/chat endpoint
chatbot_api_example.py   # Standalone API example
```

## License

Part of the AI Period Tracker project.



