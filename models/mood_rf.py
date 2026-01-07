"""
Mood Analysis Model - Random Forest & Rule-based
Analyzes mood patterns and provides wellness insights
"""
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import os

# Try to import scikit-learn (optional)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not available. Using rule-based analysis only.")


class MoodAnalyzer:
    """Mood pattern analyzer using Random Forest and rule-based methods"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
    
    def analyze_mood_patterns(self, moods: List[Dict], cycles: List[Dict] = None) -> Dict:
        """
        Analyze mood patterns and return insights
        Returns dictionary with patterns, trends, and recommendations
        """
        if not moods:
            return {
                'pattern': 'insufficient_data',
                'trend': 'neutral',
                'insights': ['Start logging your mood to get personalized insights!'],
                'recommendations': []
            }
        
        # Rule-based analysis (always available)
        insights = []
        recommendations = []
        
        # Analyze recent moods
        recent_moods = moods[:7] if len(moods) >= 7 else moods
        mood_counts = {}
        for mood in recent_moods:
            emoji = mood.get('mood_emoji', '😐')
            mood_counts[emoji] = mood_counts.get(emoji, 0) + 1
        
        # Determine dominant mood
        if mood_counts:
            dominant_mood = max(mood_counts, key=mood_counts.get)
            dominant_count = mood_counts[dominant_mood]
            total = len(recent_moods)
            
            if dominant_count / total >= 0.5:
                insights.append(f"Your dominant mood this week: {self._get_mood_name(dominant_mood)}")
        
        # Check for negative trend
        negative_emojis = ['😢', '😰', '😡', '😴']
        negative_count = sum(mood_counts.get(emoji, 0) for emoji in negative_emojis)
        
        if negative_count / len(recent_moods) >= 0.4:
            insights.append("You've been experiencing more negative moods recently")
            recommendations.append("Consider practicing mindfulness or talking to someone you trust")
            recommendations.append("Try light exercise or spending time in nature")
        
        # Check for positive trend
        positive_emojis = ['😊', '😍']
        positive_count = sum(mood_counts.get(emoji, 0) for emoji in positive_emojis)
        
        if positive_count / len(recent_moods) >= 0.6:
            insights.append("You've been in a positive mood lately! Keep it up!")
            recommendations.append("Maintain your current wellness routine")
        
        # Cycle-related analysis
        if cycles:
            try:
                latest_cycle = cycles[-1]
                cycle_start = datetime.strptime(latest_cycle['start_date'], '%Y-%m-%d')
                today = datetime.now()
                days_since_start = (today - cycle_start).days
                
                # PMS period (typically 5-7 days before period)
                if 21 <= days_since_start <= 28:
                    insights.append("You're in the premenstrual phase - mood changes are normal")
                    recommendations.append("Be gentle with yourself during this time")
                    recommendations.append("Consider light exercise and healthy snacks")
            except:
                pass
        
        # Determine overall trend
        if negative_count > positive_count:
            trend = 'declining'
        elif positive_count > negative_count:
            trend = 'improving'
        else:
            trend = 'stable'
        
        return {
            'pattern': 'analyzed',
            'trend': trend,
            'insights': insights if insights else ['Keep tracking your mood for more insights'],
            'recommendations': recommendations if recommendations else ['Continue logging daily moods']
        }
    
    def _get_mood_name(self, emoji: str) -> str:
        """Get mood name from emoji"""
        mood_map = {
            '😊': 'Happy',
            '😢': 'Sad',
            '😴': 'Tired',
            '😰': 'Anxious',
            '😡': 'Angry',
            '😍': 'Excited',
            '😐': 'Neutral'
        }
        return mood_map.get(emoji, 'Unknown')
    
    def get_wellness_tips(self, moods: List[Dict], cycles: List[Dict] = None) -> List[str]:
        """Get personalized wellness tips based on mood and cycle data"""
        tips = []
        
        if not moods:
            return [
                "Start your wellness journey by logging your mood daily",
                "Track your cycle to understand patterns in your body",
                "Stay hydrated and get enough sleep"
            ]
        
        # Analyze recent mood
        recent_moods = moods[:3] if len(moods) >= 3 else moods
        recent_emojis = [m.get('mood_emoji', '😐') for m in recent_moods]
        
        # Tips based on mood
        if '😴' in recent_emojis:
            tips.append("You've been feeling tired - prioritize 7-9 hours of sleep")
            tips.append("Consider light exercise to boost energy levels")
        
        if '😰' in recent_emojis or '😡' in recent_emojis:
            tips.append("Try deep breathing exercises or meditation")
            tips.append("Take breaks throughout the day to manage stress")
        
        if '😊' in recent_emojis or '😍' in recent_emojis:
            tips.append("Great mood! Keep up your current wellness routine")
        
        # Cycle-based tips
        if cycles:
            try:
                latest_cycle = cycles[-1]
                cycle_start = datetime.strptime(latest_cycle['start_date'], '%Y-%m-%d')
                today = datetime.now()
                days_since_start = (today - cycle_start).days
                
                if days_since_start <= 5:
                    tips.append("During your period: Rest, hydrate, and use heat therapy if needed")
                elif 6 <= days_since_start <= 13:
                    tips.append("Follicular phase: Great time for new activities and exercise")
                elif 14 <= days_since_start <= 16:
                    tips.append("Ovulation: Peak energy - perfect for challenging workouts")
                elif 17 <= days_since_start <= 28:
                    tips.append("Luteal phase: Focus on self-care and gentle activities")
            except:
                pass
        
        # Default tips if none generated
        if not tips:
            tips = [
                "Maintain a consistent sleep schedule",
                "Stay hydrated - aim for 8 glasses of water daily",
                "Include fruits and vegetables in your meals",
                "Take time for activities you enjoy"
            ]
        
        return tips[:5]  # Return top 5 tips
    
    def predict_mood_trend(self, moods: List[Dict]) -> str:
        """Predict mood trend for next few days (simple rule-based)"""
        if len(moods) < 3:
            return "stable"
        
        recent = moods[:5]
        older = moods[5:10] if len(moods) >= 10 else []
        
        if not older:
            return "stable"
        
        # Simple trend calculation
        recent_avg = self._mood_score_avg(recent)
        older_avg = self._mood_score_avg(older)
        
        if recent_avg > older_avg + 0.3:
            return "improving"
        elif recent_avg < older_avg - 0.3:
            return "declining"
        else:
            return "stable"
    
    def _mood_score_avg(self, moods: List[Dict]) -> float:
        """Calculate average mood score (1-7 scale)"""
        mood_scores = {
            '😊': 6, '😍': 7, '😐': 4,
            '😴': 3, '😰': 2, '😡': 1, '😢': 2
        }
        
        scores = [mood_scores.get(m.get('mood_emoji', '😐'), 4) for m in moods]
        return sum(scores) / len(scores) if scores else 4.0


def analyze_mood_patterns(moods: List[Dict], cycles: List[Dict] = None) -> Dict:
    """Convenience function for mood analysis"""
    analyzer = MoodAnalyzer()
    return analyzer.analyze_mood_patterns(moods, cycles)


def get_wellness_tips(moods: List[Dict], cycles: List[Dict] = None) -> List[str]:
    """Convenience function for wellness tips"""
    analyzer = MoodAnalyzer()
    return analyzer.get_wellness_tips(moods, cycles)



