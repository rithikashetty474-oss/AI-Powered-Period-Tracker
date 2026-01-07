"""
Quick test script for the Health Chatbot
Run this to test the chatbot before integrating into the main app
"""
from models.health_chatbot import get_chatbot

def test_chatbot():
    """Test the chatbot with sample questions"""
    print("=" * 60)
    print("Health Chatbot Test")
    print("=" * 60)
    
    # Initialize chatbot
    print("\n1. Initializing chatbot...")
    chatbot = get_chatbot()
    print("   ✓ Chatbot initialized!")
    
    # Test questions
    test_questions = [
        "What is a normal menstrual cycle length?",
        "How long does a period usually last?",
        "What causes period cramps?",
        "How can I relieve period pain?",
        "What is ovulation?",
        "Tell me about my cycle",  # Should use fallback
        "What are some random health tips?"  # Should use fallback
    ]
    
    print("\n2. Testing questions...\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"Question {i}: {question}")
        result = chatbot.get_answer(question)
        print(f"   Answer: {result['answer'][:100]}...")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Source: {result['source']}")
        print()
    
    print("=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == '__main__':
    test_chatbot()



