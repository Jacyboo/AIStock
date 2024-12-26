import google.generativeai as genai
from agents.state import AgentState
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Gemini API using environment variable
genai.configure(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

def market_data_agent(state: AgentState):
    """Responsible for gathering and preprocessing market data"""
    messages = state["messages"]
    data = state["data"]

    # Set default dates
    end_date = data["end_date"] or datetime.now().strftime('%Y-%m-%d')
    if not data["start_date"]:
        # Calculate 3 months before end_date
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        start_date = end_date_obj.replace(month=end_date_obj.month - 3) if end_date_obj.month > 3 else \
            end_date_obj.replace(year=end_date_obj.year - 1, month=end_date_obj.month + 9)
        start_date = start_date.strftime('%Y-%m-%d')
    else:
        start_date = data["start_date"]

    # Use the manually input data
    prices = data["manual_data"]["prices"]
    financial_metrics = data["manual_data"]["financial_metrics"]
    insider_trades = data["manual_data"]["insider_trades"]
    market_cap = data["manual_data"]["market_cap"]

    # Update the state with the data
    data.update({
        "prices": prices,
        "financial_metrics": financial_metrics,
        "insider_trades": insider_trades,
        "market_cap": market_cap,
        "start_date": start_date,
        "end_date": end_date
    })

    return {
        "messages": messages,
        "data": data,
    }