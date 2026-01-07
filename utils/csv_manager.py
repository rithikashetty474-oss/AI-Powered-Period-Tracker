"""
CSV Manager - Utility functions for reading/writing CSV files
"""
import csv
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Optional

# Data directory path
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# CSV file paths
USERS_CSV = os.path.join(DATA_DIR, 'users.csv')
CYCLES_CSV = os.path.join(DATA_DIR, 'cycles.csv')
MOODS_CSV = os.path.join(DATA_DIR, 'moods.csv')
PREDICTIONS_CSV = os.path.join(DATA_DIR, 'predictions.csv')
DAILY_ACTIVITY_CSV = os.path.join(DATA_DIR, 'daily_activity.csv')
SELFCARE_CSV = os.path.join(DATA_DIR, 'selfcare.csv')


def ensure_data_dir():
    """Ensure data directory exists"""
    os.makedirs(DATA_DIR, exist_ok=True)


def hash_password(password: str) -> str:
    """Hash a password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a hash"""
    return hash_password(password) == password_hash


def read_csv(filepath: str) -> List[Dict]:
    """Read CSV file and return list of dictionaries"""
    ensure_data_dir()
    if not os.path.exists(filepath):
        return []
    
    data = []
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return data


def write_csv(filepath: str, data: List[Dict], mode: str = 'w'):
    """Write data to CSV file"""
    ensure_data_dir()
    if not data:
        return
    
    try:
        with open(filepath, mode, newline='', encoding='utf-8') as f:
            if mode == 'w' or os.path.getsize(filepath) == 0:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
            else:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writerows(data)
    except Exception as e:
        print(f"Error writing {filepath}: {e}")


def append_csv(filepath: str, row: Dict):
    """Append a single row to CSV file"""
    ensure_data_dir()
    file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
    
    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"Error appending to {filepath}: {e}")


# User functions
def get_user_by_username(username: str) -> Optional[Dict]:
    """Get user by username"""
    users = read_csv(USERS_CSV)
    for user in users:
        if user.get('username') == username:
            return user
    return None


def get_user_by_id(user_id: str) -> Optional[Dict]:
    """Get user by ID"""
    users = read_csv(USERS_CSV)
    for user in users:
        if user.get('id') == str(user_id):
            return user
    return None


def create_user(username: str, email: str, password: str) -> Dict:
    """Create a new user"""
    users = read_csv(USERS_CSV)
    new_id = str(len(users) + 1) if users else '1'
    
    # Check if username or email already exists
    for user in users:
        if user.get('username') == username or user.get('email') == email:
            raise ValueError("Username or email already exists")
    
    new_user = {
        'id': new_id,
        'username': username,
        'email': email,
        'password_hash': hash_password(password),
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    append_csv(USERS_CSV, new_user)
    return new_user


# Cycle functions
def get_user_cycles(user_id: str) -> List[Dict]:
    """Get all cycles for a user"""
    cycles = read_csv(CYCLES_CSV)
    return [c for c in cycles if c.get('user_id') == str(user_id)]


def get_latest_cycle(user_id: str) -> Optional[Dict]:
    """Get the most recent cycle for a user"""
    cycles = get_user_cycles(user_id)
    if not cycles:
        return None
    return max(cycles, key=lambda x: x.get('start_date', ''))


def log_cycle(user_id: str, start_date: str, end_date: str = None, 
              cycle_length: int = None, period_length: int = None) -> Dict:
    """Log a new cycle"""
    cycles = read_csv(CYCLES_CSV)
    new_id = str(len(cycles) + 1) if cycles else '1'
    
    new_cycle = {
        'id': new_id,
        'user_id': str(user_id),
        'start_date': start_date,
        'end_date': end_date or '',
        'cycle_length': str(cycle_length) if cycle_length else '',
        'period_length': str(period_length) if period_length else '',
        'logged_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    append_csv(CYCLES_CSV, new_cycle)
    return new_cycle


# Mood functions
def get_user_moods(user_id: str, limit: int = None) -> List[Dict]:
    """Get all moods for a user, optionally limited"""
    moods = read_csv(MOODS_CSV)
    user_moods = [m for m in moods if m.get('user_id') == str(user_id)]
    user_moods.sort(key=lambda x: x.get('date', ''), reverse=True)
    if limit:
        return user_moods[:limit]
    return user_moods


def log_mood(user_id: str, date: str, mood_emoji: str, 
             mood_text: str = '', notes: str = '') -> Dict:
    """Log a new mood entry"""
    moods = read_csv(MOODS_CSV)
    new_id = str(len(moods) + 1) if moods else '1'
    
    new_mood = {
        'id': new_id,
        'user_id': str(user_id),
        'date': date,
        'mood_emoji': mood_emoji,
        'mood_text': mood_text,
        'notes': notes,
        'logged_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    append_csv(MOODS_CSV, new_mood)
    return new_mood


# Prediction functions
def save_prediction(user_id: str, predicted_date: str, 
                   confidence: float, model_type: str = 'lstm') -> Dict:
    """Save a prediction"""
    predictions = read_csv(PREDICTIONS_CSV)
    new_id = str(len(predictions) + 1) if predictions else '1'
    
    new_prediction = {
        'id': new_id,
        'user_id': str(user_id),
        'predicted_date': predicted_date,
        'confidence': str(confidence),
        'model_type': model_type,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    append_csv(PREDICTIONS_CSV, new_prediction)
    return new_prediction


def get_user_predictions(user_id: str) -> List[Dict]:
    """Get all predictions for a user"""
    predictions = read_csv(PREDICTIONS_CSV)
    return [p for p in predictions if p.get('user_id') == str(user_id)]


def get_latest_prediction(user_id: str) -> Optional[Dict]:
    """Get the most recent prediction for a user"""
    predictions = get_user_predictions(user_id)
    if not predictions:
        return None
    return max(predictions, key=lambda x: x.get('created_at', ''))


# Daily Activity functions for streak tracking
def log_daily_activity(user_id: str, activity_type: str = 'login', date: str = None) -> Dict:
    """Log daily activity (login, mood_log, cycle_log, etc.)"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    activities = read_csv(DAILY_ACTIVITY_CSV)
    
    # Check if activity already logged for this date
    existing = None
    for act in activities:
        if (act.get('user_id') == str(user_id) and 
            act.get('date') == date and 
            act.get('activity_type') == activity_type):
            existing = act
            break
    
    if existing:
        return existing  # Already logged
    
    new_id = str(len(activities) + 1) if activities else '1'
    
    new_activity = {
        'id': new_id,
        'user_id': str(user_id),
        'date': date,
        'activity_type': activity_type,  # login, mood_log, cycle_log, etc.
        'logged_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    append_csv(DAILY_ACTIVITY_CSV, new_activity)
    return new_activity


def get_user_activities(user_id: str, limit: int = None) -> List[Dict]:
    """Get all activities for a user"""
    activities = read_csv(DAILY_ACTIVITY_CSV)
    user_activities = [a for a in activities if a.get('user_id') == str(user_id)]
    user_activities.sort(key=lambda x: x.get('date', ''), reverse=True)
    if limit:
        return user_activities[:limit]
    return user_activities


def get_daily_activities(user_id: str, date: str = None) -> List[Dict]:
    """Get activities for a specific date"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    activities = read_csv(DAILY_ACTIVITY_CSV)
    return [a for a in activities if a.get('user_id') == str(user_id) and a.get('date') == date]


def has_activity_today(user_id: str) -> bool:
    """Check if user has any activity today"""
    today = datetime.now().strftime('%Y-%m-%d')
    activities = get_daily_activities(user_id, today)
    return len(activities) > 0


# Self-Care functions
def log_selfcare(user_id: str, date: str, tasks: Dict) -> Dict:
    """Log self-care tasks for a date"""
    selfcare_data = read_csv(SELFCARE_CSV)
    
    # Check if entry exists for this date
    existing = None
    for entry in selfcare_data:
        if entry.get('user_id') == str(user_id) and entry.get('date') == date:
            existing = entry
            break
    
    new_entry = {
        'id': existing.get('id') if existing else str(len(selfcare_data) + 1) if selfcare_data else '1',
        'user_id': str(user_id),
        'date': date,
        'meditation': '1' if tasks.get('meditation') else '0',
        'walk': '1' if tasks.get('walk') else '0',
        'water': '1' if tasks.get('water') else '0',
        'sleep': '1' if tasks.get('sleep') else '0',
        'supplements': '1' if tasks.get('supplements') else '0',
        'logged_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if existing:
        # Update existing entry
        for i, entry in enumerate(selfcare_data):
            if entry.get('id') == existing.get('id'):
                selfcare_data[i] = new_entry
                break
        write_csv(SELFCARE_CSV, selfcare_data)
    else:
        append_csv(SELFCARE_CSV, new_entry)
    
    return new_entry


def get_selfcare_by_date(user_id: str, date: str = None) -> Optional[Dict]:
    """Get self-care data for a specific date"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    selfcare_data = read_csv(SELFCARE_CSV)
    for entry in selfcare_data:
        if entry.get('user_id') == str(user_id) and entry.get('date') == date:
            return entry
    return None


def get_user_selfcare(user_id: str, limit: int = None) -> List[Dict]:
    """Get all self-care entries for a user"""
    selfcare_data = read_csv(SELFCARE_CSV)
    user_selfcare = [s for s in selfcare_data if s.get('user_id') == str(user_id)]
    user_selfcare.sort(key=lambda x: x.get('date', ''), reverse=True)
    if limit:
        return user_selfcare[:limit]
    return user_selfcare


def calculate_selfcare_streak(user_id: str) -> int:
    """Calculate self-care streak (days with all 5 tasks completed)"""
    selfcare_data = get_user_selfcare(user_id)
    if not selfcare_data:
        return 0
    
    streak = 0
    today = datetime.now().date()
    
    # Sort by date descending
    sorted_entries = sorted(
        [s for s in selfcare_data if s.get('date')],
        key=lambda x: x.get('date', ''),
        reverse=True
    )
    
    for entry in sorted_entries:
        try:
            entry_date = datetime.strptime(entry['date'], '%Y-%m-%d').date()
            days_diff = (today - entry_date).days
            
            # Check if all tasks completed
            all_completed = (
                entry.get('meditation') == '1' and
                entry.get('walk') == '1' and
                entry.get('water') == '1' and
                entry.get('sleep') == '1' and
                entry.get('supplements') == '1'
            )
            
            if all_completed:
                if days_diff == streak:
                    streak += 1
                elif days_diff > streak:
                    break
            else:
                break
        except:
            continue
    
    return streak



