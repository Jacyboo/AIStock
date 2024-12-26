from langchain_core.messages import HumanMessage
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

from agents.state import AgentState, show_agent_reasoning

# Load environment variables
load_dotenv()

# Configure the Gemini API using environment variable
genai.configure(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

def normalize_confidence(value):
    """Normalize confidence value to be between 0 and 1"""
    try:
        if isinstance(value, str):
            # Remove % if present and convert to float
            value = float(value.rstrip('%')) / 100
        if isinstance(value, (int, float)):
            # Ensure value is between 0 and 1
            return min(max(abs(value), 0.0), 1.0)
    except (ValueError, TypeError):
        pass
    return 0.0

##### Portfolio Management Agent #####
def portfolio_management_agent(state):
    """Makes final trading decisions and generates orders"""
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]

    def get_agent_message(messages, agent_name):
        """Safely get agent message"""
        try:
            msg = next((msg for msg in messages if msg.name == agent_name), None)
            if msg:
                return msg
        except:
            pass
        return None

    # Get all agent messages
    technical_message = get_agent_message(state["messages"], "technical_analyst") or \
                       get_agent_message(state["messages"], "technical_analyst_agent")
    fundamentals_message = get_agent_message(state["messages"], "fundamentals_agent")
    sentiment_message = get_agent_message(state["messages"], "sentiment_agent")
    risk_message = get_agent_message(state["messages"], "risk_management_agent")

    # Create default message if any are missing
    default_message = HumanMessage(
        content=json.dumps({
            "signal": "neutral",
            "confidence": 0.5,
            "reasoning": "No analysis available"
        })
    )

    technical_message = technical_message or default_message
    fundamentals_message = fundamentals_message or default_message
    sentiment_message = sentiment_message or default_message
    risk_message = risk_message or default_message

    # Create the prompt
    system_prompt = """You are a portfolio manager making final trading decisions.
    Your job is to make a trading decision based on the team's analysis while strictly adhering
    to risk management constraints.

    RISK MANAGEMENT CONSTRAINTS:
    - You MUST NOT exceed the max_position_size specified by the risk manager
    - You MUST follow the trading_action (buy/sell/hold) recommended by risk management
    - These are hard constraints that cannot be overridden by other signals

    When weighing the different signals for direction and timing:
    1. Fundamental Analysis (50% weight)
       - Primary driver of trading decisions
       - Should determine overall direction
    
    2. Technical Analysis (35% weight)
       - Secondary confirmation
       - Helps with entry/exit timing
    
    3. Sentiment Analysis (15% weight)
       - Final consideration
       - Can influence sizing within risk limits
    
    The decision process should be:
    1. First check risk management constraints
    2. Then evaluate fundamental outlook
    3. Use technical analysis for timing
    4. Consider sentiment for final adjustment
    
    Provide the following in your output:
    - "action": "buy" | "sell" | "hold",
    - "quantity": <positive integer>
    - "confidence": <float between 0 and 1>
    - "agent_signals": <list of agent signals including agent name, signal (bullish | bearish | neutral), and their confidence>
    - "reasoning": <concise explanation of the decision including how you weighted the signals>

    Trading Rules:
    - Never exceed risk management position limits
    - Only buy if you have available cash
    - Only sell if you have shares to sell
    - Quantity must be ≤ current position for sells
    - Quantity must be ≤ max_position_size from risk management"""

    human_prompt = f"""Based on the team's analysis below, make your trading decision.

    Technical Analysis Trading Signal: {technical_message.content}
    Fundamental Analysis Trading Signal: {fundamentals_message.content}
    Sentiment Analysis Trading Signal: {sentiment_message.content}
    Risk Management Trading Signal: {risk_message.content}

    Here is the current portfolio:
    Portfolio:
    Cash: {portfolio['cash']:.2f}
    Current Position: {portfolio['stock']} shares

    Only include the action, quantity, reasoning, confidence, and agent_signals in your output as JSON.  Do not include any JSON markdown.

    Remember, the action must be either buy, sell, or hold.
    You can only buy if you have available cash.
    You can only sell if you have shares in the portfolio to sell."""

    # Generate response using Gemini
    response = model.generate_content([system_prompt, human_prompt])
    result = response.text

    # Parse the response and normalize confidence values
    try:
        decision = json.loads(result)
        if 'confidence' in decision:
            decision['confidence'] = normalize_confidence(decision['confidence'])
        if 'agent_signals' in decision:
            for signal in decision['agent_signals']:
                if 'confidence' in signal:
                    signal['confidence'] = normalize_confidence(signal['confidence'])
        result = json.dumps(decision)
    except json.JSONDecodeError:
        result = json.dumps({
            "action": "hold",
            "quantity": 0,
            "confidence": 0.5,
            "reasoning": "Error parsing portfolio management decision",
            "agent_signals": []
        })

    # Create the portfolio management message
    message = HumanMessage(
        content=result,
        name="portfolio_management",
    )

    # Print the decision if the flag is set
    if show_reasoning:
        show_agent_reasoning(message.content, "Portfolio Management Agent")

    return {"messages": state["messages"] + [message]}