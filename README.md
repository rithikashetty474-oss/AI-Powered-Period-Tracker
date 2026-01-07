# AI-Powered Period Tracker 🌸

A beautiful, modern web application for tracking menstrual cycles with AI-powered predictions, mood analysis, and wellness insights.

## Features

- 🧠 **AI-Powered Cycle Predictions** - LSTM neural network learns your cycle patterns
- 😊 **Mood Tracking** - Log daily moods with emoji-based interface
- 📅 **Interactive Calendar** - Visualize your cycle with color-coded calendar
- 🔥 **Wellness Streaks** - Gamified tracking to keep you motivated
- 🤖 **AI Chat Assistant** - Get instant answers about your health
- 📚 **Education Hub** - Curated articles about menstrual health
- 📊 **Personalized Insights** - AI analyzes your patterns and provides tips

## Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Backend**: Flask (Python)
- **Storage**: CSV files
- **AI Models**: 
  - LSTM for cycle prediction
  - Random Forest/Rule-based for mood analysis

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd aiperiod
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Mac/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:5000`

## Project Structure

```
aiperiod/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── static/
│   ├── css/
│   │   └── style.css     # Modern, responsive stylesheet
│   └── js/
│       └── main.js       # Interactive JavaScript
│
├── templates/
│   ├── base.html         # Base template with navigation
│   ├── index.html        # Landing page
│   ├── login.html        # Login page
│   ├── register.html     # Registration page
│   ├── dashboard.html    # Main dashboard
│   ├── cycle_calendar.html  # Interactive cycle calendar
│   ├── mood_logging.html    # Emoji mood selector
│   ├── wellness_streaks.html # Gamified streak tracker
│   ├── education_hub.html   # Article cards layout
│   └── ai_chat.html         # AI chat interface
│
├── models/
│   ├── cycle_lstm.py     # LSTM model for cycle prediction
│   └── mood_rf.py        # Random Forest/rule-based mood analysis
│
├── utils/
│   └── csv_manager.py    # CSV read/write utilities
│
└── data/
    ├── users.csv         # User accounts
    ├── cycles.csv        # Cycle history
    ├── moods.csv         # Mood logs
    └── predictions.csv   # AI predictions cache
```

## Usage

### First Time Setup

1. **Register an Account**
   - Click "Get Started" on the landing page
   - Create your account with username, email, and password

2. **Log Your First Period**
   - Go to Calendar page
   - Click "Log Period Start"
   - Enter your period start date

3. **Start Tracking Your Mood**
   - Visit Mood Logging page
   - Select how you're feeling
   - Add optional notes
   - Save your mood

### Features Guide

#### Dashboard
- View your current cycle day
- See your logging streak
- Get AI-generated wellness tips
- Quick access to all features

#### Cycle Calendar
- Visual calendar showing your cycle
- Color-coded days (period, predicted, fertile window)
- Log period start/end dates
- Navigate between months

#### Mood Logging
- Emoji-based mood selection
- Daily mood tracking
- View mood history
- Pattern analysis

#### Wellness Streaks
- Track consecutive logging days
- View achievements and badges
- Weekly statistics
- Motivational messages

#### AI Chat
- Ask questions about your cycle
- Get wellness tips
- Mood pattern insights
- Health advice

#### Education Hub
- Browse health articles
- Search by category
- Learn about menstrual health
- FAQs and tips

## API Endpoints

### Authentication
- `POST /api/login` - User login
- `POST /api/register` - User registration
- `GET /logout` - User logout

### Data Logging
- `POST /api/log-cycle` - Log period start/end
- `POST /api/log-mood` - Log daily mood

### AI Features
- `GET /api/predict-cycle` - Get cycle prediction
- `GET /api/analyze-mood` - Analyze mood patterns
- `POST /api/chat` - AI chat responses

### Statistics
- `GET /api/dashboard-stats` - Get dashboard statistics

## AI Models

### Cycle Prediction (LSTM)
- Uses historical cycle data
- Learns individual patterns
- Predicts next period date
- Provides confidence scores

### Mood Analysis (Random Forest/Rule-based)
- Analyzes mood patterns
- Identifies trends
- Provides personalized tips
- Cycle-related insights

## Customization

### Changing Colors
Edit `static/css/style.css` and modify CSS variables:
```css
:root {
    --primary-pink: #FF6B9D;
    --secondary-purple: #9B59B6;
    /* ... */
}
```

### Adding Articles
Edit `templates/education_hub.html` and add articles to the JavaScript array.

### Modifying AI Responses
Edit the `api_chat()` function in `app.py` to customize chat responses.

## Security Notes

⚠️ **Important**: This is a development version. For production:

1. Change `app.secret_key` in `app.py` to a secure random string
2. Use a proper database instead of CSV files
3. Implement proper password hashing (bcrypt recommended)
4. Add CSRF protection
5. Use HTTPS
6. Implement rate limiting
7. Add input validation and sanitization

## Troubleshooting

### Port Already in Use
If port 5000 is busy, change it in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### CSV File Errors
Make sure the `data/` directory exists and is writable.

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Future Enhancements

- [ ] Database integration (PostgreSQL/MySQL)
- [ ] User authentication with JWT
- [ ] Mobile app version
- [ ] Advanced ML models
- [ ] Export data functionality
- [ ] Reminders and notifications
- [ ] Social features
- [ ] Integration with health apps

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Support

For issues, questions, or suggestions, please open an issue on the repository.

---

Made with ❤️ for women's wellness



