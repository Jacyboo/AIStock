from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from agents.fundamentals import fundamentals_agent
from agents.market_data import market_data_agent
from agents.portfolio_manager import portfolio_management_agent
from agents.technicals import technical_analyst_agent
from agents.risk_manager import risk_management_agent
from agents.sentiment import sentiment_agent
from agents.state import AgentState

import argparse
import json
from datetime import datetime, timedelta
import os
from tools.web_research import get_stock_data
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
if not api_key:
    print("\nError: GOOGLE_GEMINI_API_KEY environment variable not set!")
    print("Please follow these steps:")
    print("1. Create a .env file in the project root directory")
    print("2. Add your Gemini API key to the file:")
    print("   GOOGLE_GEMINI_API_KEY=your-api-key-here")
    print("3. Make sure to get your API key from: https://ai.google.dev/")
    exit(1)

# Initialize Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

def save_output(state, ticker, timestamp):
    """Save analysis results to a well-formatted text file"""
    
    # Create outputs directory if it doesn't exist using Path
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Format filename with timestamp using Path
    filename = output_dir / f"{ticker}_analysis_{timestamp}.txt"
    
    def get_agent_message(messages, agent_name):
        """Safely get and parse an agent's message"""
        try:
            msg = next((msg.content for msg in messages if msg.name == agent_name), None)
            if msg:
                return json.loads(msg)
        except Exception:
            pass
        return {
            "signal": "N/A",
            "confidence": 0,
            "reasoning": "No analysis available"
        }
    
    with open(filename, "w", encoding="utf-8") as f:  # Add encoding for cross-platform compatibility
        # Write header
        f.write(f"=== Investment Analysis Report for {ticker} ===\n")
        f.write(f"Generated on: {timestamp}\n\n")
        
        # Get all agent messages
        risk = get_agent_message(state["messages"], "risk_management_agent")
        tech = get_agent_message(state["messages"], "technical_analyst")
        fund = get_agent_message(state["messages"], "fundamentals_agent")
        sent = get_agent_message(state["messages"], "sentiment_agent")
        port = get_agent_message(state["messages"], "portfolio_management")
        
        # Write overall decision section
        f.write("=== OVERALL DECISION ===\n")
        action = port.get('action', 'N/A').upper()
        confidence = format_confidence(port.get('confidence', 0))
        quantity = port.get('quantity', 0)
        position_size = min(risk.get('position_size', 0) * 100, 100)  # Cap at 100%
        
        f.write(f"DECISION: {action}\n")
        f.write(f"CONFIDENCE: {confidence:.1f}%\n")
        if action in ['BUY', 'SELL']:
            f.write(f"QUANTITY: {quantity}\n")
        f.write(f"POSITION SIZE: {position_size:.1f}%\n\n")
        
        f.write("SUPPORTING SIGNALS:\n")
        f.write(f"- Technical: {tech.get('signal', 'N/A')} ({format_confidence(tech.get('confidence', 0)):.1f}%)\n")
        f.write(f"- Fundamental: {fund.get('signal', 'N/A')} ({format_confidence(fund.get('confidence', 0)):.1f}%)\n")
        f.write(f"- Sentiment: {sent.get('signal', 'N/A')} ({format_confidence(sent.get('confidence', 0)):.1f}%)\n")
        f.write(f"- Risk Level: {risk.get('signal', 'N/A')} ({format_confidence(risk.get('confidence', 0)):.1f}%)\n\n")
        
        f.write("REASONING:\n")
        f.write(f"{port.get('reasoning', 'N/A')}\n\n")
        
        # Write detailed analysis sections
        f.write("=== DETAILED ANALYSIS ===\n\n")
        
        # Write technical analysis section
        f.write("Technical Analysis:\n")
        f.write(f"Signal: {tech.get('signal', 'N/A')}\n")
        f.write(f"Confidence: {format_confidence(tech.get('confidence', 0)):.1f}%\n")
        f.write(f"Reasoning: {tech.get('reasoning', 'N/A')}\n\n")
        
        # Write fundamental analysis section
        f.write("Fundamental Analysis:\n")
        f.write(f"Signal: {fund.get('signal', 'N/A')}\n")
        f.write(f"Confidence: {format_confidence(fund.get('confidence', 0)):.1f}%\n")
        f.write(f"Reasoning: {fund.get('reasoning', 'N/A')}\n\n")
        
        # Write sentiment analysis section
        f.write("Sentiment Analysis:\n")
        f.write(f"Signal: {sent.get('signal', 'N/A')}\n")
        f.write(f"Confidence: {format_confidence(sent.get('confidence', 0)):.1f}%\n")
        f.write(f"Reasoning: {sent.get('reasoning', 'N/A')}\n\n")
        
        # Write risk assessment section
        f.write("Risk Assessment:\n")
        f.write(f"Risk Signal: {risk.get('signal', 'N/A')}\n")
        f.write(f"Risk Confidence: {format_confidence(risk.get('confidence', 0)):.1f}%\n")
        f.write(f"Position Size: {position_size:.1f}%\n")
        f.write(f"Reasoning: {risk.get('reasoning', 'N/A')}\n\n")
        
        # Write portfolio management details
        f.write("Portfolio Management Details:\n")
        f.write(f"Action: {port.get('action', 'N/A')}\n")
        f.write(f"Quantity: {port.get('quantity', 0)}\n")
        f.write(f"Confidence: {format_confidence(port.get('confidence', 0)):.1f}%\n")
        f.write(f"Reasoning: {port.get('reasoning', 'N/A')}\n")
    
    return filename

def format_confidence(value):
    """Format confidence value as percentage, capped at 100%"""
    try:
        if isinstance(value, str):
            # Remove % if present and convert to float
            value = float(value.rstrip('%')) / 100
        if isinstance(value, (int, float)):
            # Cap at 100% and ensure minimum of 0%
            value = min(max(abs(float(value)), 0.0), 1.0)
            return value * 100
    except (ValueError, TypeError):
        pass
    return 0.0

def format_output(messages):
    """Format the output messages into a clear structure"""
    output = {
        "summary": {},
        "detailed_analysis": {
            "technical": {},
            "fundamental": {},
            "sentiment": {},
            "risk": {},
            "final_decision": {}
        }
    }
    
    # Process each message
    for msg in messages:
        try:
            # Try to parse as JSON first
            try:
                content = json.loads(msg.content)
            except json.JSONDecodeError:
                # If not JSON, use the content as is
                content = {"message": msg.content}
            
            # Format confidence as percentage if present
            if 'confidence' in content:
                content['confidence'] = format_confidence(content['confidence'])
            
            if msg.name == "technical_analyst_agent":
                output["detailed_analysis"]["technical"] = content
            elif msg.name == "fundamentals_agent":
                output["detailed_analysis"]["fundamental"] = content
            elif msg.name == "sentiment_agent":
                output["detailed_analysis"]["sentiment"] = content
            elif msg.name == "risk_management_agent":
                output["detailed_analysis"]["risk"] = content
            elif msg.name == "portfolio_management":
                # Format confidence in agent signals
                if 'agent_signals' in content:
                    for signal in content['agent_signals']:
                        if 'confidence' in signal:
                            signal['confidence'] = format_confidence(signal['confidence'])
                
                output["detailed_analysis"]["final_decision"] = content
                # Add summary from final decision
                output["summary"] = {
                    "action": content.get("action", ""),
                    "quantity": content.get("quantity", 0),
                    "confidence": content.get("confidence", "0%"),
                    "reasoning": content.get("reasoning", "")
                }
        except Exception as e:
            # If anything goes wrong, store the error
            output["detailed_analysis"][msg.name] = {
                "error": f"Failed to process message: {str(e)}",
                "raw_content": msg.content
            }
    
    # If no final decision was found, add placeholder summary
    if not output["summary"]:
        output["summary"] = {
            "action": "unknown",
            "quantity": 0,
            "confidence": "0%",
            "reasoning": "Failed to generate final decision"
        }
    
    return output

##### Run the Hedge Fund #####
def run_hedge_fund(ticker: str, start_date: str, end_date: str, portfolio: dict, manual_data: dict, show_reasoning: bool = False):
    final_state = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Make a trading decision based on the provided data.",
                )
            ],
            "data": {
                "ticker": ticker,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
                "manual_data": manual_data
            },
            "metadata": {
                "show_reasoning": show_reasoning,
            }
        },
    )
    
    # Format and save the output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = save_output(final_state, ticker, timestamp)
    
    def get_agent_signal(messages, agent_name):
        """Get signal from agent message"""
        try:
            msg = next((msg.content for msg in messages if msg.name == agent_name), None)
            if msg:
                signal = json.loads(msg)
                return signal
        except:
            pass
        return None
    
    # Get all agent signals
    tech = get_agent_signal(final_state["messages"], "technical_analyst")
    fund = get_agent_signal(final_state["messages"], "fundamentals_agent")
    sent = get_agent_signal(final_state["messages"], "sentiment_agent")
    risk = get_agent_signal(final_state["messages"], "risk_management_agent")
    port = get_agent_signal(final_state["messages"], "portfolio_management")
    
    # Calculate position size once
    position_size = risk.get('position_size', 0) * 100 if risk else 0
    
    # Print summary to console
    print("\n=== INVESTMENT ANALYSIS SUMMARY ===")
    print(f"Ticker: {ticker}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print overall decision
    if port:
        action = port.get('action', 'N/A').upper()
        confidence = port.get('confidence', '0%')
        quantity = port.get('quantity', 0)
        
        print("\n=== OVERALL DECISION ===")
        print(f"DECISION: {action}")
        print(f"CONFIDENCE: {confidence}")
        if action in ['BUY', 'SELL']:
            print(f"QUANTITY: {quantity}")
        print(f"POSITION SIZE: {position_size:.1f}%")
        
        print("\nSUPPORTING SIGNALS:")
        if tech:
            print(f"- Technical: {tech.get('signal', 'N/A')} ({tech.get('confidence', 'N/A')})")
        if fund:
            print(f"- Fundamental: {fund.get('signal', 'N/A')} ({fund.get('confidence', 'N/A')})")
        if sent:
            print(f"- Sentiment: {sent.get('signal', 'N/A')} ({sent.get('confidence', 'N/A')})")
        if risk:
            print(f"- Risk Level: {risk.get('signal', 'N/A')} ({risk.get('confidence', 'N/A')})")
        
        print(f"\nREASONING:")
        print(port.get('reasoning', 'N/A'))
    else:
        print("\nNo portfolio management decision available")
    
    # Print detailed analysis
    if show_reasoning:
        print("\n=== DETAILED ANALYSIS ===")
        
        if tech:
            print("\nTechnical Analysis:")
            print(f"Signal: {tech.get('signal', 'N/A')}")
            print(f"Confidence: {tech.get('confidence', 'N/A')}")
            print(f"Reasoning: {tech.get('reasoning', 'N/A')}")
        
        if fund:
            print("\nFundamental Analysis:")
            print(f"Signal: {fund.get('signal', 'N/A')}")
            print(f"Confidence: {fund.get('confidence', 'N/A')}")
            print(f"Reasoning: {fund.get('reasoning', 'N/A')}")
        
        if sent:
            print("\nSentiment Analysis:")
            print(f"Signal: {sent.get('signal', 'N/A')}")
            print(f"Confidence: {sent.get('confidence', 'N/A')}")
            print(f"Reasoning: {sent.get('reasoning', 'N/A')}")
        
        if risk:
            print("\nRisk Assessment:")
            print(f"Signal: {risk.get('signal', 'N/A')}")
            print(f"Confidence: {risk.get('confidence', 'N/A')}")
            print(f"Position Size: {position_size:.1f}%")
            print(f"Reasoning: {risk.get('reasoning', 'N/A')}")
    
    print(f"\nFull analysis saved to: {output_file}")
    
    return final_state["messages"][-1].content

# Define the new workflow
workflow = StateGraph(AgentState)

# Add nodes for each agent
workflow.add_node("market_data", market_data_agent)
workflow.add_node("technical_analyst", technical_analyst_agent)
workflow.add_node("fundamentals_agent", fundamentals_agent)
workflow.add_node("sentiment_agent", sentiment_agent)
workflow.add_node("risk_management_agent", risk_management_agent)
workflow.add_node("portfolio_management", portfolio_management_agent)

# Define the workflow
workflow.set_entry_point("market_data")

# Market data feeds into analysis agents
workflow.add_edge("market_data", "technical_analyst")
workflow.add_edge("market_data", "fundamentals_agent")
workflow.add_edge("market_data", "sentiment_agent")

# Analysis agents feed into risk management
workflow.add_edge("technical_analyst", "risk_management_agent")
workflow.add_edge("fundamentals_agent", "risk_management_agent")
workflow.add_edge("sentiment_agent", "risk_management_agent")

# Risk management feeds into portfolio management
workflow.add_edge("risk_management_agent", "portfolio_management")

# Portfolio management is the final step
workflow.add_edge("portfolio_management", END)

app = workflow.compile()

# Add this at the bottom of the file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the hedge fund trading system')
    parser.add_argument('--show-reasoning', action='store_true', help='Show reasoning from each agent')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD). Defaults to 3 months before end date')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD). Defaults to today')
    
    args = parser.parse_args()
    
    while True:
        # Get ticker input from user
        ticker = input("\nEnter stock ticker symbol (e.g., AAPL): ").upper().strip()
        
        if ticker == "":
            print("Ticker symbol cannot be empty. Please try again.")
            continue
            
        print(f"\nFetching real-time data for {ticker}...")
        manual_data = get_stock_data(ticker)
        
        if not manual_data:
            retry = input("\nWould you like to try another ticker? (y/n): ").lower().strip()
            if retry != 'y':
                print("Exiting program.")
                exit(0)
            continue
        
        print("Successfully fetched stock data!")
        break
    
    # Validate dates if provided
    if args.start_date:
        try:
            datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Start date must be in YYYY-MM-DD format")
    
    if args.end_date:
        try:
            datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("End date must be in YYYY-MM-DD format")
    
    # Get portfolio settings from environment variables
    portfolio = {
        "cash": float(os.getenv('INITIAL_CASH', 100000.0)),
        "stock": int(os.getenv('INITIAL_STOCK', 0))
    }
    
    # Use environment variable for show_reasoning if not provided as argument
    show_reasoning = args.show_reasoning or os.getenv('SHOW_REASONING', 'false').lower() == 'true'
    
    result = run_hedge_fund(
        ticker=ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        portfolio=portfolio,
        manual_data=manual_data,
        show_reasoning=show_reasoning
    )