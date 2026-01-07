"""
Health Assistant Chatbot
Uses Gemini API (if available) or embeddings + vector search for Q&A matching with LLM fallback
"""
import os
import csv
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Check for Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Gemini API will not be available.")

# Check for embeddings
try:
    from sentence_transformers import SentenceTransformer
    import torch
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Using fallback mode.")


class HealthChatbot:
    """Health Assistant Chatbot with Gemini API (if available) or embedding-based search"""
    
    def __init__(self, dataset_path: str = None, model_name: str = 'all-MiniLM-L6-v2', gemini_model: str = 'gemini-pro'):
        """
        Initialize the chatbot
        
        Args:
            dataset_path: Path to health_faq.csv
            model_name: Sentence transformer model name (lightweight default)
            gemini_model: Gemini model name (default: 'gemini-pro')
        """
        self.dataset_path = dataset_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data', 'health_faq.csv'
        )
        self.model_name = model_name
        self.gemini_model_name = gemini_model
        self.embeddings_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data', 'chatbot_embeddings.pkl'
        )
        
        self.questions = []
        self.answers = []
        self.question_embeddings = None
        self.model = None
        self.gemini_model = None
        self.use_gemini = False
        
        # Conversation history storage (per user_id)
        self.conversation_history = {}  # {user_id: [{"role": "user"/"assistant", "content": "..."}, ...]}
        self.max_history_length = 20  # Maximum messages to keep before summarizing
        self.summarize_threshold = 15  # Start summarizing when reaching this many messages
        
        # Safety disclaimer
        self.safety_disclaimer = (
            "⚠️ **Important**: This information is for general wellness purposes only "
            "and is not a substitute for professional medical advice, diagnosis, or treatment. "
            "Always consult with a healthcare provider for medical concerns."
        )
        
        # Initialize Gemini API if available
        if GEMINI_AVAILABLE:
            gemini_api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if gemini_api_key:
                try:
                    genai.configure(api_key=gemini_api_key)
                    self.gemini_model = genai.GenerativeModel(self.gemini_model_name)
                    self.use_gemini = True
                    print(f"✓ Gemini API initialized with model: {self.gemini_model_name}")
                except Exception as e:
                    print(f"Error initializing Gemini API: {e}")
                    print("Falling back to embedding-based search")
            else:
                print("Gemini API key not found. Set GEMINI_API_KEY environment variable to use Gemini.")
                print("Falling back to embedding-based search")
        
        # Initialize embedding model if Gemini is not available
        if not self.use_gemini and EMBEDDINGS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                print(f"Loaded embedding model: {model_name}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
    
    def load_dataset(self) -> Tuple[List[str], List[str]]:
        """
        Load Q&A pairs from CSV file
        
        Returns:
            Tuple of (questions, answers) lists
        """
        questions = []
        answers = []
        
        if not os.path.exists(self.dataset_path):
            print(f"Dataset not found at {self.dataset_path}")
            return questions, answers
        
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Support multiple CSV formats
                    if 'question' in row and 'answer' in row:
                        questions.append(row['question'].strip())
                        answers.append(row['answer'].strip())
                    elif 'Question' in row and 'Answer' in row:
                        questions.append(row['Question'].strip())
                        answers.append(row['Answer'].strip())
                    elif len(row) >= 2:
                        # Assume first column is question, second is answer
                        cols = list(row.values())
                        if len(cols) >= 2:
                            questions.append(cols[0].strip())
                            answers.append(cols[1].strip())
        
        except Exception as e:
            print(f"Error loading dataset: {e}")
        
        self.questions = questions
        self.answers = answers
        print(f"Loaded {len(questions)} Q&A pairs from dataset")
        return questions, answers
    
    def embed_dataset(self, force_recompute: bool = False) -> np.ndarray:
        """
        Create embeddings for all questions in the dataset
        
        Args:
            force_recompute: If True, recompute even if cached embeddings exist
            
        Returns:
            Numpy array of embeddings
        """
        if not EMBEDDINGS_AVAILABLE or self.model is None:
            print("Embeddings not available, using keyword-based fallback")
            return None
        
        # Load cached embeddings if available
        if not force_recompute and os.path.exists(self.embeddings_path):
            try:
                with open(self.embeddings_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if (cached_data.get('questions') == self.questions and 
                        cached_data.get('model_name') == self.model_name):
                        print("Loading cached embeddings...")
                        self.question_embeddings = cached_data['embeddings']
                        return self.question_embeddings
            except Exception as e:
                print(f"Error loading cached embeddings: {e}")
        
        # Compute embeddings
        if not self.questions:
            self.load_dataset()
        
        if not self.questions:
            print("No questions to embed")
            return None
        
        print(f"Computing embeddings for {len(self.questions)} questions...")
        self.question_embeddings = self.model.encode(
            self.questions,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Cache embeddings
        try:
            os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump({
                    'embeddings': self.question_embeddings,
                    'questions': self.questions,
                    'model_name': self.model_name
                }, f)
            print("Embeddings cached successfully")
        except Exception as e:
            print(f"Error caching embeddings: {e}")
        
        return self.question_embeddings
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def _keyword_match(self, user_input: str, threshold: float = 0.3) -> Optional[Tuple[str, float]]:
        """
        Fallback keyword-based matching when embeddings aren't available
        
        Returns:
            Tuple of (answer, confidence) or None
        """
        if not self.questions:
            return None
        
        user_lower = user_input.lower()
        best_match = None
        best_score = 0.0
        
        for i, question in enumerate(self.questions):
            question_lower = question.lower()
            question_words = set(question_lower.split())
            user_words = set(user_lower.split())
            
            # Calculate Jaccard similarity
            intersection = len(question_words & user_words)
            union = len(question_words | user_words)
            if union > 0:
                score = intersection / union
                
                # Bonus for exact substring matches
                if user_lower in question_lower or question_lower in user_lower:
                    score += 0.2
                
                if score > best_score:
                    best_score = score
                    best_match = i
        
        if best_score >= threshold and best_match is not None:
            return (self.answers[best_match], best_score)
        
        return None
    
    def _summarize_conversation(self, history: List[Dict], user_id: str = None) -> str:
        """
        Summarize conversation history when it gets too long
        
        Args:
            history: List of conversation messages
            user_id: User ID for context
            
        Returns:
            Summary string
        """
        if not history or len(history) < 5:
            return None
        
        try:
            # Take first few messages to summarize
            messages_to_summarize = history[:self.summarize_threshold]
            
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in messages_to_summarize
            ])
            
            summary_prompt = f"""Summarize the following conversation about menstrual health and wellness. Keep it concise (2-3 sentences) and focus on key topics discussed:

{conversation_text}

Summary:"""
            
            if self.gemini_model:
                response = self.gemini_model.generate_content(
                    summary_prompt,
                    generation_config={
                        'temperature': 0.3,
                        'max_output_tokens': 200,
                    }
                )
                return response.text.strip()
        except Exception as e:
            print(f"Error summarizing conversation: {e}")
        
        # Fallback: simple summary
        topics = []
        for msg in messages_to_summarize[:5]:
            if msg['role'] == 'user':
                topics.append(msg['content'][:50])
        return f"Previous conversation covered: {', '.join(topics[:3])}..."
    
    def _get_gemini_answer(self, user_input: str, context: str = None, conversation_history: List[Dict] = None, user_id: str = None) -> Dict[str, any]:
        """
        Get answer using Gemini API with conversation history
        
        Args:
            user_input: User's question
            context: Optional context about user's cycle/mood data
            conversation_history: Previous conversation messages
            user_id: User ID for maintaining conversation history
            
        Returns:
            Dictionary with 'answer', 'confidence', 'source', 'safety_disclaimer'
        """
        if not self.gemini_model:
            return None
        
        try:
            # Enhanced system prompt with new guidelines
            system_prompt = """You are a helpful and empathetic AI wellness assistant specializing in menstrual health, cycle tracking, fertility, PMS, and general wellness.

CRITICAL GUIDELINES:
- Answer unlimited user questions clearly and helpfully
- Remember and reference the conversation history - continue the chat without resetting
- Always stay in context of menstrual health, cycle prediction, fertility, PMS, and general wellness
- NEVER give medical diagnoses - always recommend consulting healthcare providers for medical concerns
- NEVER hallucinate or make up information - only provide accurate, evidence-based information
- Give SHORT, SIMPLE, helpful answers unless the user explicitly asks for detailed explanations
- If conversation becomes long, you may summarize earlier parts but continue the conversation naturally
- Be supportive, non-judgmental, and understanding
- Focus on wellness, cycle tracking, mood patterns, and general health tips

Remember: Keep answers concise unless asked for detail. Stay focused on menstrual health topics."""
            
            # Build conversation context
            conversation_context = ""
            if conversation_history and len(conversation_history) > 0:
                # Filter out system messages and format conversation history
                filtered_history = [
                    msg for msg in conversation_history[-10:]  # Last 10 messages for context
                    if msg.get('role') in ['user', 'assistant']
                ]
                if filtered_history:
                    history_text = "\n".join([
                        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                        for msg in filtered_history
                    ])
                    conversation_context = f"\n\nPrevious conversation:\n{history_text}\n"
            
            # Add user context if available
            context_text = ""
            if context:
                context_text = f"\nUser's personal data: {context}\n"
            
            # Build full prompt
            full_prompt = system_prompt + context_text + conversation_context + f"\nUser: {user_input}\nAssistant:"
            
            # Generate response with updated config for shorter responses
            response = self.gemini_model.generate_content(
                full_prompt,
                generation_config={
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 40,
                    'max_output_tokens': 512,  # Reduced for shorter answers
                }
            )
            
            answer = response.text.strip()
            
            # Ensure answer is concise (if too long, truncate but keep it natural)
            if len(answer) > 500 and "?" not in user_input.lower():
                # Keep first sentence or two for short answers
                sentences = answer.split('.')
                if len(sentences) > 2:
                    answer = '. '.join(sentences[:2]) + '.'
            
            return {
                'answer': answer,
                'confidence': 0.9,  # High confidence for Gemini responses
                'source': 'gemini',
                'safety_disclaimer': self.safety_disclaimer
            }
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return None
    
    def get_conversation_history(self, user_id: str) -> List[Dict]:
        """Get conversation history for a user"""
        return self.conversation_history.get(user_id, [])
    
    def add_to_history(self, user_id: str, role: str, content: str):
        """Add a message to conversation history"""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].append({
            'role': role,
            'content': content
        })
        
        # Summarize if conversation gets too long
        if len(self.conversation_history[user_id]) > self.max_history_length:
            summary = self._summarize_conversation(self.conversation_history[user_id], user_id)
            if summary:
                # Keep recent messages and add summary
                recent_messages = self.conversation_history[user_id][-5:]  # Keep last 5
                self.conversation_history[user_id] = [
                    {'role': 'system', 'content': f'Previous conversation summary: {summary}'}
                ] + recent_messages
    
    def clear_history(self, user_id: str):
        """Clear conversation history for a user"""
        if user_id in self.conversation_history:
            del self.conversation_history[user_id]
    
    def get_answer(self, user_input: str, similarity_threshold: float = 0.5, user_context: str = None, user_id: str = None, conversation_history: List[Dict] = None) -> Dict[str, any]:
        """
        Get answer for user input using Gemini API (if available) or vector search/fallback
        
        Args:
            user_input: User's question
            similarity_threshold: Minimum similarity score (0-1) to match from dataset
            user_context: Optional context about user's cycle/mood data for Gemini
            user_id: User ID for maintaining conversation history
            conversation_history: Optional conversation history (if None, will use stored history)
            
        Returns:
            Dictionary with 'answer', 'confidence', 'source', 'safety_disclaimer'
        """
        if not user_input or not user_input.strip():
            return {
                'answer': 'Please ask me a question about menstrual health and wellness.',
                'confidence': 0.0,
                'source': 'fallback',
                'safety_disclaimer': self.safety_disclaimer
            }
        
        user_input = user_input.strip()
        
        # Get conversation history
        if user_id:
            if conversation_history is None:
                conversation_history = self.get_conversation_history(user_id)
            # Add current user message to history
            self.add_to_history(user_id, 'user', user_input)
        
        # Try Gemini API first if available
        if self.use_gemini and self.gemini_model:
            gemini_result = self._get_gemini_answer(
                user_input, 
                user_context, 
                conversation_history=conversation_history,
                user_id=user_id
            )
            if gemini_result:
                # Add assistant response to history
                if user_id:
                    self.add_to_history(user_id, 'assistant', gemini_result['answer'])
                return gemini_result
            # If Gemini fails, fall back to other methods
            print("Gemini API call failed, falling back to embedding search")
        
        # Try embedding-based search
        if self.model is not None and self.question_embeddings is not None:
            try:
                # Encode user input
                user_embedding = self.model.encode([user_input], convert_to_numpy=True)[0]
                
                # Calculate similarities
                similarities = []
                for q_embedding in self.question_embeddings:
                    similarity = self._cosine_similarity(user_embedding, q_embedding)
                    similarities.append(similarity)
                
                # Find best match
                best_idx = np.argmax(similarities)
                best_similarity = similarities[best_idx]
                
                if best_similarity >= similarity_threshold:
                    return {
                        'answer': self.answers[best_idx],
                        'confidence': float(best_similarity),
                        'source': 'dataset',
                        'safety_disclaimer': self.safety_disclaimer
                    }
            
            except Exception as e:
                print(f"Error in embedding search: {e}")
        
        # Fallback to keyword matching
        keyword_result = self._keyword_match(user_input)
        if keyword_result:
            answer, confidence = keyword_result
            return {
                'answer': answer,
                'confidence': float(confidence),
                'source': 'keyword_match',
                'safety_disclaimer': self.safety_disclaimer
            }
        
        # Final fallback: LLM-style response
        return self._get_fallback_answer(user_input)
    
    def _get_fallback_answer(self, user_input: str) -> Dict[str, any]:
        """
        Generate fallback answer when no dataset match is found
        Uses rule-based responses for common topics
        """
        user_lower = user_input.lower()
        
        # Cycle-related
        if any(word in user_lower for word in ['period', 'menstrual', 'cycle', 'bleeding', 'menses']):
            return {
                'answer': (
                    "A typical menstrual cycle lasts 21-35 days, with bleeding usually lasting 3-7 days. "
                    "Cycle length can vary, and it's normal for cycles to change over time. "
                    "Track your cycle to understand your patterns better. If you experience significant "
                    "changes or concerns, consult with a healthcare provider."
                ),
                'confidence': 0.3,
                'source': 'fallback_llm',
                'safety_disclaimer': self.safety_disclaimer
            }
        
        # Pain-related
        if any(word in user_lower for word in ['pain', 'cramp', 'ache', 'discomfort']):
            return {
                'answer': (
                    "Period pain (dysmenorrhea) is common and can often be managed with: "
                    "heat therapy (heating pad or warm bath), gentle exercise, over-the-counter "
                    "pain relievers (as directed), relaxation techniques, and maintaining a healthy diet. "
                    "If pain is severe or interferes with daily activities, please consult a healthcare provider."
                ),
                'confidence': 0.3,
                'source': 'fallback_llm',
                'safety_disclaimer': self.safety_disclaimer
            }
        
        # Mood-related
        if any(word in user_lower for word in ['mood', 'emotional', 'feeling', 'sad', 'anxious', 'irritable']):
            return {
                'answer': (
                    "Hormonal changes during your cycle can affect mood. Many people experience "
                    "mood changes before or during their period. Strategies that may help include: "
                    "regular exercise, adequate sleep, stress management techniques, maintaining a "
                    "balanced diet, and tracking your mood patterns. If mood changes are severe or "
                    "significantly impact your life, consider speaking with a healthcare provider."
                ),
                'confidence': 0.3,
                'source': 'fallback_llm',
                'safety_disclaimer': self.safety_disclaimer
            }
        
        # Ovulation/fertility
        if any(word in user_lower for word in ['ovulation', 'fertile', 'fertility', 'pregnancy', 'conceive']):
            return {
                'answer': (
                    "Ovulation typically occurs around day 14 of a 28-day cycle, but this can vary. "
                    "The fertile window is usually 5-6 days before and including ovulation day. "
                    "Tracking your cycle, basal body temperature, and cervical mucus can help identify "
                    "ovulation. For personalized fertility guidance, consult with a healthcare provider "
                    "or fertility specialist."
                ),
                'confidence': 0.3,
                'source': 'fallback_llm',
                'safety_disclaimer': self.safety_disclaimer
            }
        
        # General wellness
        return {
            'answer': (
                "I'm here to help with questions about menstrual health, cycle tracking, "
                "wellness tips, and general information. Could you provide more details about "
                "what you'd like to know? For specific medical concerns, please consult with "
                "a healthcare provider."
            ),
            'confidence': 0.2,
            'source': 'fallback_llm',
            'safety_disclaimer': self.safety_disclaimer
        }
    
    def initialize(self, force_recompute: bool = False):
        """Initialize the chatbot: load dataset and create embeddings"""
        print("Initializing Health Chatbot...")
        self.load_dataset()
        if self.model is not None:
            self.embed_dataset(force_recompute=force_recompute)
        print("Chatbot initialized successfully!")


# Global chatbot instance
_chatbot_instance = None

def get_chatbot() -> HealthChatbot:
    """Get or create global chatbot instance"""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = HealthChatbot()
        _chatbot_instance.initialize()
    return _chatbot_instance



