"""
AI-Powered Period Tracker - Flask Application
Main application file with all routes and API endpoints
"""
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.csv_manager import (
    get_user_by_username, get_user_by_id, create_user,
    get_user_cycles, get_latest_cycle, log_cycle,
    get_user_moods, log_mood, verify_password,
    save_prediction, get_latest_prediction,
    log_daily_activity, has_activity_today, get_user_activities,
    log_selfcare, get_selfcare_by_date, calculate_selfcare_streak
)
from models.cycle_lstm import predict_cycle
from models.mood_rf import analyze_mood_patterns, get_wellness_tips
from models.health_chatbot import get_chatbot

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'  # Change this in production!


# Helper functions
def require_login(f):
    """Decorator to require login"""
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function


def calculate_streak(user_id: str) -> int:
    """Calculate current daily activity streak"""
    from utils.csv_manager import get_user_activities
    
    activities = get_user_activities(user_id)
    if not activities:
        return 0
    
    # Get unique dates with any activity
    activity_dates = set()
    for act in activities:
        try:
            date_str = act.get('date', '')
            if date_str:
                activity_dates.add(date_str)
        except:
            continue
    
    if not activity_dates:
        return 0
    
    # Sort dates descending (most recent first)
    sorted_dates = sorted([datetime.strptime(d, '%Y-%m-%d').date() for d in activity_dates], reverse=True)
    
    if not sorted_dates:
        return 0
    
    streak = 0
    today = datetime.now().date()
    
    most_recent = sorted_dates[0]
    today = datetime.now().date()
    
    # If most recent activity is more than 1 day ago, streak is broken
    days_since_last = (today - most_recent).days
    if days_since_last > 1:
        return 0  # Streak broken - gap of more than 1 day
    
    # Count consecutive days backwards from most recent activity
    # Start from most recent and count backwards
    current_date = most_recent
    for activity_date in sorted_dates:
        if activity_date == current_date:
            streak += 1
            current_date = current_date - timedelta(days=1)
        elif activity_date < current_date:
            # We've passed the current date, check if this date is consecutive
            if activity_date == current_date:
                streak += 1
                current_date = current_date - timedelta(days=1)
            else:
                # Gap found, stop counting
                break
        # If activity_date > current_date, skip (already processed)
    
    return streak


def calculate_best_streak(user_id: str) -> int:
    """Calculate best streak ever achieved"""
    from utils.csv_manager import get_user_activities
    
    activities = get_user_activities(user_id)
    if not activities:
        return 0
    
    # Get unique dates
    activity_dates = set()
    for act in activities:
        try:
            date_str = act.get('date', '')
            if date_str:
                activity_dates.add(date_str)
        except:
            continue
    
    if not activity_dates:
        return 0
    
    # Sort dates ascending
    sorted_dates = sorted([datetime.strptime(d, '%Y-%m-%d').date() for d in activity_dates])
    
    if not sorted_dates:
        return 0
    
    best_streak = 1
    current_streak = 1
    
    for i in range(1, len(sorted_dates)):
        days_diff = (sorted_dates[i] - sorted_dates[i-1]).days
        if days_diff == 1:
            current_streak += 1
            best_streak = max(best_streak, current_streak)
        else:
            current_streak = 1
    
    return best_streak


# ==================== PAGE ROUTES ====================

@app.route('/')
def index():
    """Landing page"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            flash('Please fill in all fields', 'error')
            return render_template('login.html')
        
        user = get_user_by_username(username)
        if user and verify_password(password, user['password_hash']):
            session['user_id'] = user['id']
            session['username'] = user['username']
            # Log daily login activity for streak tracking
            log_daily_activity(user['id'], 'login')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        if not username or not email or not password:
            flash('Please fill in all fields', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return render_template('register.html')
        
        try:
            user = create_user(username, email, password)
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Registration successful!', 'success')
            return redirect(url_for('dashboard'))
        except ValueError as e:
            flash(str(e), 'error')
    
    return render_template('register.html')


@app.route('/logout')
def logout():
    """Logout"""
    session.clear()
    return redirect(url_for('index'))


@app.route('/dashboard')
@require_login
def dashboard():
    """Main dashboard"""
    user_id = session['user_id']
    username = session.get('username', 'User')
    
    # Log daily login activity (will skip if already exists for today)
    try:
        log_daily_activity(user_id, 'login')
    except Exception as e:
        print(f"Error logging activity: {e}")
    
    # Get user data
    cycles = get_user_cycles(user_id)
    moods = get_user_moods(user_id, limit=7)
    latest_cycle = get_latest_cycle(user_id)
    
    # Calculate stats
    current_day = 0
    next_period = None
    cycle_length = 28
    
    if latest_cycle:
        try:
            cycle_start = datetime.strptime(latest_cycle['start_date'], '%Y-%m-%d')
            today = datetime.now()
            current_day = (today - cycle_start).days
            
            cycle_length = int(latest_cycle.get('cycle_length', 28)) if latest_cycle.get('cycle_length') else 28
            next_period = cycle_start + timedelta(days=cycle_length)
        except:
            pass
    
    # Get prediction
    prediction = get_latest_prediction(user_id)
    if prediction:
        try:
            next_period = datetime.strptime(prediction['predicted_date'], '%Y-%m-%d')
        except:
            pass
    elif latest_cycle:
        # Fallback: calculate from latest cycle
        try:
            cycle_start = datetime.strptime(latest_cycle['start_date'], '%Y-%m-%d')
            cycle_length = int(latest_cycle.get('cycle_length', 28)) if latest_cycle.get('cycle_length') else 28
            next_period = cycle_start + timedelta(days=cycle_length)
        except:
            pass
    
    # Calculate streak
    streak = calculate_streak(user_id)
    
    # Get insights
    mood_analysis = analyze_mood_patterns(moods, cycles)
    wellness_tips = get_wellness_tips(moods, cycles)
    
    return render_template('dashboard.html',
                         username=username,
                         current_day=current_day,
                         next_period=next_period,
                         cycle_length=cycle_length,
                         streak=streak,
                         cycles=cycles[:5],
                         moods=moods,
                         mood_analysis=mood_analysis,
                         wellness_tips=wellness_tips)


@app.route('/cycle_calendar')
@require_login
def cycle_calendar():
    """Cycle calendar page"""
    user_id = session['user_id']
    cycles = get_user_cycles(user_id)
    return render_template('cycle_calendar.html', cycles=cycles)


@app.route('/mood_logging')
@require_login
def mood_logging():
    """Mood logging page"""
    user_id = session['user_id']
    moods = get_user_moods(user_id, limit=10)
    return render_template('mood_logging.html', moods=moods)


@app.route('/wellness_streaks')
@require_login
def wellness_streaks():
    """Wellness streaks page"""
    user_id = session['user_id']
    
    # Log daily login activity (will skip if already exists for today)
    try:
        log_daily_activity(user_id, 'login')
    except Exception as e:
        print(f"Error logging activity: {e}")
    
    streak = calculate_streak(user_id)
    moods = get_user_moods(user_id)
    
    # Calculate weekly stats
    today = datetime.now().date()
    week_start = today - timedelta(days=today.weekday())
    week_days = [week_start + timedelta(days=i) for i in range(7)]
    
    weekly_logs = {}
    for mood in moods:
        try:
            mood_date = datetime.strptime(mood['date'], '%Y-%m-%d').date()
            if mood_date in week_days:
                weekly_logs[mood_date] = mood
        except:
            pass
    
    return render_template('wellness_streaks.html', 
                         streak=streak, 
                         weekly_logs=weekly_logs,
                         week_days=week_days)


@app.route('/education_hub')
@require_login
def education_hub():
    """Education hub page"""
    return render_template('education_hub.html')


@app.route('/ai_chat')
@require_login
def ai_chat():
    """AI chat page"""
    return render_template('ai_chat.html')


@app.route('/pregnancy')
@require_login
def pregnancy_prediction():
    """Pregnancy/Fertility prediction page"""
    return render_template('pregnancy_prediction.html')


# ==================== API ENDPOINTS ====================

@app.route('/api/log-cycle', methods=['POST'])
@require_login
def api_log_cycle():
    """API endpoint to log cycle"""
    try:
        data = request.get_json()
        user_id = session['user_id']
        
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        cycle_length = data.get('cycle_length')
        period_length = data.get('period_length')
        
        if not start_date:
            return jsonify({'success': False, 'error': 'Start date required'}), 400
        
        cycle = log_cycle(user_id, start_date, end_date, cycle_length, period_length)
        
        # Log activity for streak
        log_daily_activity(user_id, 'cycle_log', start_date)
        
        return jsonify({'success': True, 'cycle': cycle})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/log-mood', methods=['POST'])
@require_login
def api_log_mood():
    """API endpoint to log mood"""
    try:
        data = request.get_json()
        user_id = session['user_id']
        
        date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        mood_emoji = data.get('mood_emoji')
        mood_text = data.get('mood_text', '')
        notes = data.get('notes', '')
        
        if not mood_emoji:
            return jsonify({'success': False, 'error': 'Mood emoji required'}), 400
        
        mood = log_mood(user_id, date, mood_emoji, mood_text, notes)
        
        # Log activity for streak
        log_daily_activity(user_id, 'mood_log', date)
        
        return jsonify({'success': True, 'mood': mood})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict-cycle', methods=['GET'])
@require_login
def api_predict_cycle():
    """API endpoint for cycle prediction"""
    try:
        user_id = session['user_id']
        cycles = get_user_cycles(user_id)
        
        if not cycles:
            return jsonify({
                'success': False,
                'error': 'Not enough cycle data. Please log at least one cycle.'
            }), 400
        
        predicted_date, confidence = predict_cycle(cycles)
        
        if predicted_date:
            # Save prediction
            save_prediction(user_id, predicted_date, confidence, 'lstm')
            
            return jsonify({
                'success': True,
                'predicted_date': predicted_date,
                'confidence': confidence
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not generate prediction'
            }), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analyze-mood', methods=['GET'])
@require_login
def api_analyze_mood():
    """API endpoint for mood analysis"""
    try:
        user_id = session['user_id']
        moods = get_user_moods(user_id)
        cycles = get_user_cycles(user_id)
        
        analysis = analyze_mood_patterns(moods, cycles)
        tips = get_wellness_tips(moods, cycles)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'tips': tips
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chat/clear', methods=['POST'])
@require_login
def api_clear_chat():
    """API endpoint to clear conversation history"""
    try:
        user_id = session.get('user_id')
        if user_id:
            chatbot = get_chatbot()
            chatbot.clear_history(str(user_id))
            return jsonify({'success': True, 'message': 'Conversation history cleared'})
        return jsonify({'success': False, 'error': 'User not found'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
@require_login
def api_chat():
    """
    API endpoint for Health Assistant Chatbot
    Uses embedding-based vector search with LLM fallback
    """
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({
                'success': False, 
                'error': 'Message required'
            }), 400
        
        # Get chatbot instance (initializes on first call)
        chatbot = get_chatbot()
        
        # Build user context for personalized responses
        user_context = None
        user_id = session.get('user_id')
        user_id_str = str(user_id) if user_id else None
        
        if user_id:
            cycles = get_user_cycles(user_id)
            moods = get_user_moods(user_id)
            
            # Build context string with user's cycle and mood data
            context_parts = []
            if cycles:
                try:
                    latest = get_latest_cycle(user_id)
                    if latest:
                        start = datetime.strptime(latest['start_date'], '%Y-%m-%d')
                        days_ago = (datetime.now() - start).days
                        cycle_length = int(latest.get('cycle_length', 28))
                        context_parts.append(f"User's last period started {days_ago} days ago. Average cycle length: {cycle_length} days.")
                except:
                    pass
            
            if moods and len(moods) > 0:
                recent_moods = moods[:5]  # Last 5 moods
                mood_summary = ", ".join([m.get('mood_text', m.get('mood_emoji', '')) for m in recent_moods if m.get('mood_text') or m.get('mood_emoji')])
                if mood_summary:
                    context_parts.append(f"Recent mood patterns: {mood_summary}")
            
            if context_parts:
                user_context = " ".join(context_parts)
        
        # Get conversation history for this user
        conversation_history = None
        if user_id_str:
            conversation_history = chatbot.get_conversation_history(user_id_str)
        
        # Get answer using Gemini (if available) or embedding search + fallback
        result = chatbot.get_answer(
            message, 
            user_context=user_context,
            user_id=user_id_str,
            conversation_history=conversation_history
        )
        
        return jsonify({
            'success': True,
            'reply': result['answer'],
            'confidence': result.get('confidence', 0.0),
            'source': result.get('source', 'fallback'),
            'safety_disclaimer': result.get('safety_disclaimer', '')
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred processing your message',
            'reply': 'I apologize, but I encountered an error. Please try rephrasing your question or try again later.'
        }), 500


@app.route('/api/chat-old', methods=['POST'])
@require_login
def api_chat_old():
    """Legacy API endpoint for AI chat (kept for reference)"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'success': False, 'error': 'Message required'}), 400
        
        user_id = session['user_id']
        moods = get_user_moods(user_id)
        cycles = get_user_cycles(user_id)
        
        # Simple rule-based chat responses
        message_lower = message.lower()
        
        # Cycle-related questions - more specific matching
        if any(word in message_lower for word in ['period', 'menstrual', 'bleeding', 'menses']):
            if cycles:
                latest = get_latest_cycle(user_id)
                if latest:
                    try:
                        start = datetime.strptime(latest['start_date'], '%Y-%m-%d')
                        days_ago = (datetime.now() - start).days
                        predicted_date, confidence = predict_cycle(cycles)
                        
                        # Calculate cycle length
                        cycle_lengths = []
                        for cycle in cycles[-6:]:
                            try:
                                cl = int(cycle.get('cycle_length', 28))
                                if 20 <= cl <= 45:
                                    cycle_lengths.append(cl)
                            except:
                                continue
                        avg_cycle = int(sum(cycle_lengths) / len(cycle_lengths)) if cycle_lengths else 28
                        
                        response = f"Your last period started {days_ago} day{'s' if days_ago != 1 else ''} ago. "
                        response += f"Your average cycle length is {avg_cycle} days. "
                        
                        if predicted_date:
                            pred_date = datetime.strptime(predicted_date, '%Y-%m-%d')
                            days_until = (pred_date - datetime.now()).days
                            response += f"Your next period is predicted to start around {predicted_date} (in about {days_until} days) with {int(confidence * 100)}% confidence."
                        else:
                            response += "Keep logging your cycles to get more accurate predictions!"
                    except Exception as e:
                        response = "I can help you track your cycle! Make sure to log your period start dates regularly."
                else:
                    response = "Start logging your periods to get cycle predictions and insights!"
            else:
                response = "I'd love to help you track your cycle! Log your period start date to get started. You can do this from the Calendar page."
        
        # When/next period questions
        elif any(word in message_lower for word in ['when', 'next', 'predict', 'coming', 'due']):
            if cycles:
                try:
                    predicted_date, confidence = predict_cycle(cycles)
                    if predicted_date:
                        pred_date = datetime.strptime(predicted_date, '%Y-%m-%d')
                        days_until = (pred_date - datetime.now()).days
                        if days_until < 0:
                            response = f"Based on your cycle patterns, your period was predicted for {predicted_date}. If it hasn't started yet, it should be coming soon!"
                        else:
                            response = f"Your next period is predicted to start on {predicted_date}, which is in {days_until} day{'s' if days_until != 1 else ''}. This prediction has {int(confidence * 100)}% confidence based on your cycle history."
                    else:
                        response = "I need more cycle data to make accurate predictions. Keep logging your periods!"
                except:
                    response = "I'm having trouble calculating your prediction. Make sure you've logged at least one complete cycle."
            else:
                response = "I need you to log at least one period to make predictions. Start tracking from the Calendar page!"
        
        # Ovulation/fertility questions
        elif any(word in message_lower for word in ['ovulation', 'ovulate', 'fertile', 'fertility', 'conceive', 'pregnant']):
            if cycles:
                latest = get_latest_cycle(user_id)
                if latest:
                    try:
                        start = datetime.strptime(latest['start_date'], '%Y-%m-%d')
                        cycle_length = int(latest.get('cycle_length', 28))
                        ovulation_day = cycle_length - 14
                        ovulation_date = start + timedelta(days=ovulation_day)
                        fertile_start = ovulation_date - timedelta(days=5)
                        fertile_end = ovulation_date + timedelta(days=1)
                        
                        today = datetime.now()
                        days_to_ovulation = (ovulation_date - today).days
                        
                        response = f"Based on your cycle, ovulation typically occurs around day {ovulation_day} of your cycle. "
                        response += f"Your fertile window is approximately {fertile_start.strftime('%B %d')} to {fertile_end.strftime('%B %d')}. "
                        
                        if days_to_ovulation > 0:
                            response += f"Your next ovulation is predicted in about {days_to_ovulation} days."
                        elif days_to_ovulation == 0:
                            response += "You're likely ovulating today or very soon!"
                        else:
                            response += "Your ovulation window has likely passed for this cycle."
                    except:
                        response = "I can help you track your fertile window! Make sure to log your period dates regularly for accurate calculations."
                else:
                    response = "Log your period dates to get fertility window predictions!"
            else:
                response = "To calculate your fertile window, I need you to log your period dates. The fertile window is typically 5-6 days around ovulation (usually day 14 of a 28-day cycle)."
        
        # Pain/symptoms questions
        elif any(word in message_lower for word in ['pain', 'cramp', 'ache', 'hurt', 'symptom', 'bloat', 'headache']):
            response = "Period symptoms like cramps, bloating, and headaches are common. Here are some tips:\n\n"
            response += "• Use a heating pad or hot water bottle on your lower abdomen\n"
            response += "• Try gentle exercise like walking or yoga\n"
            response += "• Stay hydrated and eat anti-inflammatory foods\n"
            response += "• Consider over-the-counter pain relievers (consult your doctor first)\n"
            response += "• Practice relaxation techniques like deep breathing\n\n"
            response += "If pain is severe or unusual, please consult a healthcare provider."
        
        # Mood-related questions
        elif any(word in message_lower for word in ['mood', 'feel', 'feeling', 'emotion', 'sad', 'happy', 'anxious', 'irritable']):
            if moods:
                analysis = analyze_mood_patterns(moods, cycles)
                response = "Based on your recent mood logs:\n\n"
                if analysis.get('insights'):
                    for insight in analysis['insights'][:2]:
                        response += f"• {insight}\n"
                if analysis.get('recommendations'):
                    response += "\nRecommendations:\n"
                    for rec in analysis['recommendations'][:2]:
                        response += f"• {rec}\n"
                if not analysis.get('insights') and not analysis.get('recommendations'):
                    response += "Keep tracking your mood daily to see patterns emerge!"
            else:
                response = "Start logging your mood daily to get personalized insights! Mood changes throughout your cycle are normal and tracking them can help you understand patterns."
        
        # Wellness/health questions
        elif any(word in message_lower for word in ['wellness', 'health', 'tip', 'advice', 'help', 'suggest', 'recommend']):
            tips = get_wellness_tips(moods, cycles)
            if tips:
                response = "Here are some wellness tips for you:\n\n"
                for i, tip in enumerate(tips[:3], 1):
                    response += f"{i}. {tip}\n"
            else:
                response = "General wellness tips:\n\n"
                response += "• Stay hydrated - aim for 8 glasses of water daily\n"
                response += "• Get 7-9 hours of sleep each night\n"
                response += "• Eat a balanced diet with fruits and vegetables\n"
                response += "• Exercise regularly, but listen to your body\n"
                response += "• Track your cycle and mood for personalized insights"
        
        # Cycle length questions
        elif any(word in message_lower for word in ['length', 'long', 'short', 'regular', 'irregular']):
            if cycles:
                cycle_lengths = []
                for cycle in cycles[-6:]:
                    try:
                        cl = int(cycle.get('cycle_length', 28))
                        if 20 <= cl <= 45:
                            cycle_lengths.append(cl)
                    except:
                        continue
                
                if cycle_lengths:
                    avg_length = int(sum(cycle_lengths) / len(cycle_lengths))
                    min_length = min(cycle_lengths)
                    max_length = max(cycle_lengths)
                    
                    response = f"Your cycle lengths range from {min_length} to {max_length} days, with an average of {avg_length} days. "
                    
                    if max_length - min_length > 7:
                        response += "Your cycles show some variation, which is normal. Most women have cycles between 21-35 days."
                    else:
                        response += "Your cycles are quite regular! A normal cycle is typically 21-35 days."
                else:
                    response = "I need more cycle data to analyze your cycle length patterns. Keep logging your periods!"
            else:
                response = "Log your period dates to track your cycle length. A normal cycle is typically 21-35 days."
        
        # Greetings
        elif any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']):
            response = "Hello! I'm your AI wellness assistant. I can help you with:\n\n"
            response += "• Cycle tracking and predictions\n"
            response += "• Fertility window calculations\n"
            response += "• Mood pattern analysis\n"
            response += "• Wellness tips and advice\n"
            response += "• Period symptom management\n\n"
            response += "What would you like to know?"
        
        # What can you do / help
        elif any(word in message_lower for word in ['what can you', 'what do you', 'capabilities', 'features']):
            response = "I can help you with:\n\n"
            response += "📅 **Cycle Tracking**: Predict your next period, track cycle length, and identify patterns\n"
            response += "🌱 **Fertility**: Calculate your fertile window and ovulation dates\n"
            response += "😊 **Mood Analysis**: Analyze your mood patterns and provide insights\n"
            response += "💡 **Wellness Tips**: Personalized health and wellness advice\n"
            response += "🩸 **Period Management**: Tips for managing symptoms and discomfort\n\n"
            response += "Just ask me anything about your cycle, mood, or health!"
        
        # Default response
        else:
            response = "I'm here to help with cycle tracking, mood analysis, and wellness tips! Try asking:\n\n"
            response += "• \"When is my next period?\"\n"
            response += "• \"When am I ovulating?\"\n"
            response += "• \"Tell me about my mood patterns\"\n"
            response += "• \"How can I manage period pain?\"\n"
            response += "• \"What wellness tips do you have?\"\n\n"
            response += "Or ask me anything else about your cycle or health!"
        
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dashboard-stats', methods=['GET'])
@require_login
def api_dashboard_stats():
    """API endpoint for dashboard statistics"""
    try:
        user_id = session['user_id']
        cycles = get_user_cycles(user_id)
        moods = get_user_moods(user_id, limit=7)
        latest_cycle = get_latest_cycle(user_id)
        streak = calculate_streak(user_id)
        
        stats = {
            'streak': streak,
            'total_cycles': len(cycles),
            'total_moods': len(moods),
            'current_day': 0,
            'next_period': None
        }
        
        if latest_cycle:
            try:
                cycle_start = datetime.strptime(latest_cycle['start_date'], '%Y-%m-%d')
                today = datetime.now()
                stats['current_day'] = (today - cycle_start).days
                
                cycle_length = int(latest_cycle.get('cycle_length', 28)) if latest_cycle.get('cycle_length') else 28
                stats['next_period'] = (cycle_start + timedelta(days=cycle_length)).strftime('%Y-%m-%d')
            except:
                pass
        
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/get-cycles', methods=['GET'])
@require_login
def api_get_cycles():
    """API endpoint to get user cycles"""
    try:
        user_id = session['user_id']
        cycles = get_user_cycles(user_id)
        return jsonify({'success': True, 'cycles': cycles})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/get-cycle-phase', methods=['GET'])
@require_login
def api_get_cycle_phase():
    """API endpoint to get cycle phase information including fertile window"""
    try:
        user_id = session['user_id']
        cycles = get_user_cycles(user_id)
        
        if not cycles:
            return jsonify({
                'success': False,
                'error': 'No cycle data available. Please log your period start date first.'
            }), 200  # Return 200 instead of 400 so frontend can handle it
        
        # Get latest cycle
        latest_cycle = get_latest_cycle(user_id)
        if not latest_cycle:
            return jsonify({
                'success': False,
                'error': 'No cycle data available. Please log your period start date first.'
            }), 200  # Return 200 instead of 400 so frontend can handle it
        
        # Calculate cycle parameters
        last_period_start = datetime.strptime(latest_cycle['start_date'], '%Y-%m-%d')
        cycle_length = int(latest_cycle.get('cycle_length', 28))
        period_length = int(latest_cycle.get('period_length', 5))
        
        # Calculate average cycle length from recent cycles
        if len(cycles) > 1:
            cycle_lengths = []
            for cycle in cycles[-6:]:  # Use last 6 cycles
                try:
                    cl = int(cycle.get('cycle_length', 28))
                    if 20 <= cl <= 45:
                        cycle_lengths.append(cl)
                except:
                    continue
            if cycle_lengths:
                cycle_length = int(sum(cycle_lengths) / len(cycle_lengths))
        
        # Fallback: if cycle_length is still not set or invalid, use default 28
        if not cycle_length or cycle_length < 20 or cycle_length > 45:
            cycle_length = 28
        
        # Calculate ovulation date (typically 14 days before next period)
        # Ovulation usually occurs around day 14 of a 28-day cycle
        ovulation_day = cycle_length - 14  # Days from period start
        ovulation_date = last_period_start + timedelta(days=ovulation_day)
        
        # Fertile window: 5 days before ovulation to 1 day after (6 days total)
        fertile_window_start = ovulation_date - timedelta(days=5)
        fertile_window_end = ovulation_date + timedelta(days=1)
        
        # Next period date
        next_period_date = last_period_start + timedelta(days=cycle_length)
        
        # Calculate days in current phase
        today = datetime.now()
        days_since_start = (today - last_period_start).days
        
        if days_since_start <= period_length:
            current_phase = 'Menstrual'
        elif days_since_start < ovulation_day - 2:
            current_phase = 'Follicular'
        elif days_since_start <= ovulation_day + 1:
            current_phase = 'Ovulation'
        else:
            current_phase = 'Luteal'
        
        return jsonify({
            'success': True,
            'phase_info': {
                'current_phase': current_phase,
                'days_in_phase': days_since_start,
                'next_period_date': next_period_date.strftime('%Y-%m-%d'),
                'ovulation_date': ovulation_date.strftime('%Y-%m-%d'),
                'fertile_window_start': fertile_window_start.strftime('%Y-%m-%d'),
                'fertile_window_end': fertile_window_end.strftime('%Y-%m-%d'),
                'days_to_next_period': (next_period_date - today).days,
                'days_to_ovulation': (ovulation_date - today).days
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/get-moods', methods=['GET'])
@require_login
def api_get_moods():
    """API endpoint to get user moods"""
    try:
        user_id = session['user_id']
        moods = get_user_moods(user_id, limit=10)
        return jsonify({'success': True, 'moods': moods})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict-fertility', methods=['POST'])
@require_login
def api_predict_fertility():
    """API endpoint to predict fertility based on user answers"""
    try:
        data = request.get_json()
        
        # Extract answers
        age = int(data.get('age', 25))
        cycle_regularity = data.get('cycle_regularity', 'unknown')
        cycle_length = data.get('cycle_length', 'unknown')
        trying_duration = data.get('trying_duration', 'not_trying')
        conditions = data.get('conditions', [])
        lifestyle = data.get('lifestyle', [])
        previous_pregnancy = data.get('previous_pregnancy', 'no')
        
        # Calculate fertility score (0-100)
        score = 50  # Base score
        
        # Age factor (optimal 20-35)
        if 20 <= age <= 30:
            score += 20
        elif 31 <= age <= 35:
            score += 15
        elif 36 <= age <= 40:
            score += 5
        elif age > 40:
            score -= 10
        elif age < 20:
            score += 10
        
        # Cycle regularity factor
        if cycle_regularity == 'very_regular':
            score += 15
        elif cycle_regularity == 'somewhat_regular':
            score += 8
        elif cycle_regularity == 'irregular':
            score -= 10
        elif cycle_regularity == 'unknown':
            score -= 5
        
        # Cycle length factor
        if cycle_length in ['25-28', '29-32']:
            score += 10
        elif cycle_length in ['21-24', '33-35']:
            score += 5
        elif cycle_length == '36+':
            score -= 5
        
        # Medical conditions factor
        if 'none' not in conditions:
            if 'pcos' in conditions:
                score -= 15
            if 'endometriosis' in conditions:
                score -= 12
            if 'thyroid' in conditions:
                score -= 8
        
        # Lifestyle factors
        if 'smoking' in lifestyle:
            score -= 15
        if 'alcohol' in lifestyle:
            score -= 5
        if 'exercise' in lifestyle:
            score += 8
        if 'stress' in lifestyle:
            score -= 8
        if 'healthy' in lifestyle:
            score += 10
        
        # Previous pregnancy factor
        if previous_pregnancy == 'yes_successful':
            score += 15
        elif previous_pregnancy == 'yes_miscarriage':
            score += 5
        
        # Trying duration factor
        if trying_duration == 'not_trying':
            score += 5  # Less pressure
        elif trying_duration == 'more_than_12':
            score -= 5  # May indicate issues
        
        # Clamp score between 0 and 100
        score = max(0, min(100, score))
        
        # Generate prediction text
        if score >= 75:
            prediction_text = "Your fertility indicators are excellent! You have a high likelihood of conception with proper timing and healthy lifestyle habits."
        elif score >= 60:
            prediction_text = "Your fertility indicators are good. With optimal timing and continued healthy habits, you have a favorable chance of conception."
        elif score >= 45:
            prediction_text = "Your fertility indicators are moderate. Consider tracking your cycle more closely and consulting with a healthcare provider for personalized guidance."
        else:
            prediction_text = "Your fertility indicators suggest some challenges. We recommend consulting with a fertility specialist for a comprehensive evaluation and personalized treatment plan."
        
        # Generate insights
        insights = []
        if 20 <= age <= 35:
            insights.append("You're in the optimal age range for fertility")
        elif age > 35:
            insights.append("Age may be a factor - consider consulting a specialist")
        
        if cycle_regularity == 'very_regular':
            insights.append("Regular cycles indicate good hormonal balance")
        elif cycle_regularity == 'irregular':
            insights.append("Irregular cycles may benefit from medical evaluation")
        
        if 'exercise' in lifestyle and 'healthy' in lifestyle:
            insights.append("Your healthy lifestyle supports fertility")
        
        if 'pcos' in conditions or 'endometriosis' in conditions:
            insights.append("Medical conditions may require specialized care")
        
        if trying_duration == 'more_than_12':
            insights.append("If trying for over 12 months, consider fertility evaluation")
        
        # Generate recommendations
        recommendations = []
        
        if cycle_regularity in ['irregular', 'unknown']:
            recommendations.append("Track your cycles for at least 3 months to identify patterns")
        
        if 'smoking' in lifestyle:
            recommendations.append("Quit smoking - it significantly impacts fertility")
        
        if 'stress' in lifestyle:
            recommendations.append("Practice stress management techniques (yoga, meditation, therapy)")
        
        if not ('exercise' in lifestyle):
            recommendations.append("Incorporate moderate exercise (3-4 times per week)")
        
        recommendations.append("Track ovulation using basal body temperature or ovulation predictor kits")
        recommendations.append("Have intercourse every 2-3 days during your fertile window")
        recommendations.append("Maintain a healthy BMI (18.5-24.9)")
        recommendations.append("Take prenatal vitamins with folic acid")
        
        if score < 60:
            recommendations.append("Consider scheduling a consultation with a reproductive endocrinologist")
        
        # Calculate fertile window if we have cycle data
        fertile_window = None
        user_id = session.get('user_id')
        if user_id:
            cycles = get_user_cycles(user_id)
            if cycles:
                latest = get_latest_cycle(user_id)
                if latest:
                    try:
                        last_period = datetime.strptime(latest['start_date'], '%Y-%m-%d')
                        avg_cycle = int(latest.get('cycle_length', 28))
                        ovulation_day = avg_cycle - 14
                        ovulation_date = last_period + timedelta(days=ovulation_day)
                        fertile_start = ovulation_date - timedelta(days=5)
                        fertile_end = ovulation_date + timedelta(days=1)
                        
                        fertile_window = f"Based on your cycle data, your next fertile window is approximately {fertile_start.strftime('%B %d')} to {fertile_end.strftime('%B %d, %Y')}"
                    except:
                        pass
        
        return jsonify({
            'success': True,
            'prediction': {
                'fertility_score': score,
                'prediction_text': prediction_text,
                'insights': insights if insights else ["Continue tracking your cycle and maintain healthy habits"],
                'recommendations': recommendations,
                'fertile_window': fertile_window
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/get-streaks', methods=['GET'])
@require_login
def api_get_streaks():
    """API endpoint to get streak data"""
    try:
        user_id = session['user_id']
        
        # Ensure today's activity is logged
        try:
            log_daily_activity(user_id, 'login')
        except:
            pass
        
        streak = calculate_streak(user_id)
        moods = get_user_moods(user_id)
        
        # Calculate best streak
        best_streak = calculate_best_streak(user_id)
        
        # Get week data - check for any activity (not just moods)
        today = datetime.now().date()
        week_start = today - timedelta(days=today.weekday())
        week_days = [week_start + timedelta(days=i) for i in range(7)]
        
        activities = get_user_activities(user_id)
        weekly_logs = []
        activity_dates = set()
        for act in activities:
            try:
                act_date = datetime.strptime(act['date'], '%Y-%m-%d').date()
                activity_dates.add(act_date)
            except:
                pass
        
        for day in week_days:
            if day in activity_dates:
                weekly_logs.append(day.strftime('%Y-%m-%d'))
        
        # Calculate achievements and rewards
        achievements = []
        
        # Streak-based achievements
        if streak >= 1:
            achievements.append({'name': 'Getting Started', 'description': '1 day streak', 'icon': 'fas fa-seedling', 'unlocked': True, 'reward': '🌟'})
        if streak >= 3:
            achievements.append({'name': 'Three Day Champion', 'description': '3 day streak', 'icon': 'fas fa-medal', 'unlocked': True, 'reward': '🏅'})
        if streak >= 7:
            achievements.append({'name': 'Week Warrior', 'description': '7 day streak', 'icon': 'fas fa-trophy', 'unlocked': True, 'reward': '🏆'})
        if streak >= 14:
            achievements.append({'name': 'Two Week Hero', 'description': '14 day streak', 'icon': 'fas fa-star', 'unlocked': True, 'reward': '⭐'})
        if streak >= 30:
            achievements.append({'name': 'Monthly Master', 'description': '30 day streak', 'icon': 'fas fa-crown', 'unlocked': True, 'reward': '👑'})
        if streak >= 60:
            achievements.append({'name': 'Two Month Legend', 'description': '60 day streak', 'icon': 'fas fa-gem', 'unlocked': True, 'reward': '💎'})
        if streak >= 100:
            achievements.append({'name': 'Century Club', 'description': '100 day streak', 'icon': 'fas fa-fire', 'unlocked': True, 'reward': '🔥'})
        
        # Activity-based achievements
        total_activities = len(get_user_activities(user_id))
        if total_activities >= 10:
            achievements.append({'name': 'Active Tracker', 'description': '10 total activities', 'icon': 'fas fa-chart-line', 'unlocked': True, 'reward': '📊'})
        if total_activities >= 50:
            achievements.append({'name': 'Dedicated Logger', 'description': '50 total activities', 'icon': 'fas fa-book', 'unlocked': True, 'reward': '📚'})
        if len(moods) >= 10:
            achievements.append({'name': 'Mood Tracker', 'description': 'Logged 10 moods', 'icon': 'fas fa-smile', 'unlocked': True, 'reward': '😊'})
        if len(cycles) >= 3:
            achievements.append({'name': 'Cycle Master', 'description': 'Logged 3 cycles', 'icon': 'fas fa-calendar-alt', 'unlocked': True, 'reward': '📅'})
        
        total_activities = len(get_user_activities(user_id))
        cycles = get_user_cycles(user_id)
        
        return jsonify({
            'success': True,
            'current_streak': streak,
            'best_streak': best_streak,
            'total_logs': total_activities,
            'achievements_count': len(achievements),
            'achievements': achievements,
            'week_data': weekly_logs
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/save-selfcare', methods=['POST'])
@require_login
def api_save_selfcare():
    """API endpoint to save self-care tasks"""
    try:
        data = request.get_json()
        user_id = session['user_id']
        date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        tasks = {
            'meditation': data.get('meditation', False),
            'walk': data.get('walk', False),
            'water': data.get('water', False),
            'sleep': data.get('sleep', False),
            'supplements': data.get('supplements', False)
        }
        
        entry = log_selfcare(user_id, date, tasks)
        
        # Calculate completion percentage
        completed = sum(1 for v in tasks.values() if v)
        percentage = (completed / 5) * 100
        
        # Log activity for streak if all tasks completed
        if completed == 5:
            log_daily_activity(user_id, 'selfcare_complete', date)
        
        return jsonify({
            'success': True,
            'entry': entry,
            'percentage': percentage,
            'completed': completed,
            'total': 5
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/get-selfcare', methods=['GET'])
@require_login
def api_get_selfcare():
    """API endpoint to get self-care data for today"""
    try:
        user_id = session['user_id']
        date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        entry = get_selfcare_by_date(user_id, date)
        
        if entry:
            tasks = {
                'meditation': entry.get('meditation') == '1',
                'walk': entry.get('walk') == '1',
                'water': entry.get('water') == '1',
                'sleep': entry.get('sleep') == '1',
                'supplements': entry.get('supplements') == '1'
            }
            completed = sum(1 for v in tasks.values() if v)
            percentage = (completed / 5) * 100
        else:
            tasks = {
                'meditation': False,
                'walk': False,
                'water': False,
                'sleep': False,
                'supplements': False
            }
            completed = 0
            percentage = 0
        
        streak = calculate_selfcare_streak(user_id)
        
        return jsonify({
            'success': True,
            'tasks': tasks,
            'percentage': percentage,
            'completed': completed,
            'total': 5,
            'streak': streak
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Initialize chatbot on startup (optional - can also lazy load)
    try:
        print("Initializing Health Chatbot...")
        chatbot = get_chatbot()
        print("Health Chatbot ready!")
    except Exception as e:
        print(f"Warning: Could not initialize chatbot: {e}")
        print("Chatbot will use fallback mode only.")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)

