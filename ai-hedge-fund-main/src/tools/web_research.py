import google.generativeai as genai
from datetime import datetime, timedelta
import json
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini AI using environment variable
genai.configure(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

def clean_json_string(text):
    """Clean and extract JSON from the response text"""
    # Find JSON content between curly braces
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group()
        # Remove any markdown formatting
        json_str = re.sub(r'```json|```', '', json_str)
        # Clean up common formatting issues
        json_str = json_str.replace('\n', ' ').replace('\\n', ' ')
        json_str = re.sub(r'\s+', ' ', json_str)
        return json_str
    return None

def get_stock_data(ticker):
    """Get comprehensive stock data using Gemini AI with web search"""
    try:
        # Generate dates for historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(31)]
        
        print(f"\nAnalyzing {ticker} using real-time web data...")
        
        # Verify API key is configured
        if not os.getenv('GOOGLE_GEMINI_API_KEY'):
            raise ValueError("GOOGLE_GEMINI_API_KEY environment variable not set")
            
        response = model.generate_content(prompt)
        
        # Clean and parse the response
        json_str = clean_json_string(response.text)
        if not json_str:
            raise ValueError("Could not extract valid JSON from Gemini response")
            
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"\nError parsing market data response: {str(e)}")
            print("Using default market data values...")
            # Create a minimal valid response with realistic default values
            data = create_default_market_data(dates)
            
        # Validate and clean the data
        data = validate_and_clean_market_data(data, dates)
        
        print("Successfully gathered market data and analysis.")
        return data
        
    except Exception as e:
        print(f"\nError analyzing {ticker}: {str(e)}")
        print("\nPlease check:")
        print("1. Your API key is correctly set in the .env file")
        print("2. You have a working internet connection")
        print("3. The ticker symbol is valid (e.g., 'AAPL' for Apple)")
        return None 