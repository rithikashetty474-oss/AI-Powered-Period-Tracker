# Gemini API Setup Guide

This guide will help you set up the Gemini API for the AI chat feature.

## Step 1: Get Your Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey) or [Google Cloud Console](https://console.cloud.google.com/)
2. Sign in with your Google account
3. Create a new API key for Gemini
4. Copy your API key (you'll need it in the next step)

## Step 2: Set Up the API Key

You have two options:

### Option A: Environment Variable (Recommended)

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

**Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY=your_api_key_here
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Option B: Create a .env File

1. Create a file named `.env` in the project root directory
2. Add the following line:
```
GEMINI_API_KEY=your_api_key_here
```

**Note:** Make sure `.env` is in your `.gitignore` file to keep your API key secure!

## Step 3: Install Dependencies

Make sure you have the required packages:

```bash
pip install google-generativeai python-dotenv
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Step 4: Restart the Application

After setting up the API key, restart your Flask application:

```bash
python app.py
```

## Verification

When you start the app, you should see one of these messages:

- ✅ **"✓ Gemini API initialized with model: gemini-pro"** - Gemini is working!
- ⚠️ **"Gemini API key not found..."** - Check your API key setup
- ⚠️ **"Falling back to embedding-based search"** - Using fallback mode

## How It Works

- **With Gemini API Key**: The chatbot uses Google's Gemini AI for intelligent, conversational responses
- **Without API Key**: The chatbot falls back to the embedding-based search system (still functional!)

## Troubleshooting

### "ModuleNotFoundError: No module named 'google.generativeai'"
- Run: `pip install google-generativeai`

### "Gemini API key not found"
- Make sure you've set the `GEMINI_API_KEY` environment variable or created a `.env` file
- Restart your application after setting the key

### API Errors
- Check that your API key is valid and has not expired
- Ensure you have internet connectivity
- Check Google's API status page for any service issues

## Security Notes

⚠️ **Important**: Never commit your API key to version control!
- Add `.env` to your `.gitignore` file
- Don't share your API key publicly
- Rotate your key if it's accidentally exposed

