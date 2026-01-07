"""
Standalone Flask API Example for Health Chatbot
This is a minimal example showing how to use the chatbot API
"""
from flask import Flask, request, jsonify
from models.health_chatbot import get_chatbot

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat API endpoint
    Accepts: { "message": "user text" }
    Returns: { "reply": "assistant answer", "confidence": 0.0-1.0, "source": "dataset|fallback" }
    """
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'Message required'
            }), 400
        
        # Get chatbot instance
        chatbot = get_chatbot()
        
        # Get answer
        result = chatbot.get_answer(message)
        
        return jsonify({
            'success': True,
            'reply': result['answer'],
            'confidence': result.get('confidence', 0.0),
            'source': result.get('source', 'fallback'),
            'safety_disclaimer': result.get('safety_disclaimer', '')
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'reply': 'I apologize, but I encountered an error processing your message.'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    # Initialize chatbot
    print("Initializing Health Chatbot...")
    chatbot = get_chatbot()
    print("Chatbot ready! Starting server...")
    
    app.run(debug=True, host='0.0.0.0', port=5001)



