from langchain_core.messages import HumanMessage
import google.generativeai as genai
from agents.state import AgentState, show_agent_reasoning
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Gemini API using environment variable
api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_GEMINI_API_KEY environment variable not set")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

##### Fundamental Agent #####
def fundamentals_agent(state: AgentState):
    """Analyzes fundamental data and generates trading signals."""
    data = state["data"]
    show_reasoning = state["metadata"]["show_reasoning"]
    
    # Get the financial metrics
    metrics = data["financial_metrics"]
    market_cap = data["market_cap"]
    
    # Initialize signals list
    signals = []
    reasoning = []
    
    # 1. Profitability Analysis
    profitability_score = 0
    if metrics["return_on_equity"] > 0.15:  # Strong ROE above 15%
        profitability_score += 1
    if metrics["net_margin"] > 0.20:  # Healthy profit margins
        profitability_score += 1
    if metrics["operating_margin"] > 0.15:  # Strong operating efficiency
        profitability_score += 1
    
    signals.append('bullish' if profitability_score >= 2 else 'bearish' if profitability_score == 0 else 'neutral')
    reasoning.append(f"Profitability: ROE={metrics['return_on_equity']:.1%}, Net Margin={metrics['net_margin']:.1%}, Op Margin={metrics['operating_margin']:.1%}")
    
    # 2. Growth Analysis
    growth_score = 0
    if metrics["revenue_growth"] > 0.10:  # 10% revenue growth
        growth_score += 1
    if metrics["earnings_growth"] > 0.10:  # 10% earnings growth
        growth_score += 1
    if metrics["book_value_growth"] > 0.10:  # 10% book value growth
        growth_score += 1
    
    signals.append('bullish' if growth_score >= 2 else 'bearish' if growth_score == 0 else 'neutral')
    reasoning.append(f"Growth: Revenue={metrics['revenue_growth']:.1%}, Earnings={metrics['earnings_growth']:.1%}, Book Value={metrics['book_value_growth']:.1%}")
    
    # 3. Financial Health
    health_score = 0
    if metrics["current_ratio"] > 1.5:  # Strong liquidity
        health_score += 1
    if metrics["debt_to_equity"] < 0.5:  # Conservative debt levels
        health_score += 1
    if metrics["free_cash_flow_per_share"] > metrics["earnings_per_share"] * 0.8:  # Strong FCF conversion
        health_score += 1
    
    signals.append('bullish' if health_score >= 2 else 'bearish' if health_score == 0 else 'neutral')
    reasoning.append(f"Financial Health: Current Ratio={metrics['current_ratio']:.1f}, D/E={metrics['debt_to_equity']:.1f}, FCF/Share=${metrics['free_cash_flow_per_share']:.2f}")
    
    # 4. Valuation Analysis
    valuation_score = 0
    if metrics["price_to_earnings_ratio"] < 25:  # Reasonable P/E ratio
        valuation_score += 1
    if metrics["price_to_book_ratio"] < 3:  # Reasonable P/B ratio
        valuation_score += 1
    if metrics["price_to_sales_ratio"] < 5:  # Reasonable P/S ratio
        valuation_score += 1
    
    signals.append('bullish' if valuation_score >= 2 else 'bearish' if valuation_score == 0 else 'neutral')
    reasoning.append(f"Valuation: P/E={metrics['price_to_earnings_ratio']:.1f}, P/B={metrics['price_to_book_ratio']:.1f}, P/S={metrics['price_to_sales_ratio']:.1f}")
    
    # 5. Cash Flow Analysis
    if "financial_line_items" in data and data["financial_line_items"]:
        fcf = data["financial_line_items"][0].get("free_cash_flow", 0)
        fcf_yield = fcf / market_cap if market_cap > 0 else 0
        if fcf_yield > 0.05:  # FCF yield > 5%
            signals.append('bullish')
            reasoning.append(f"Cash Flow: Strong FCF yield of {fcf_yield:.1%}")
        elif fcf_yield < 0.02:  # FCF yield < 2%
            signals.append('bearish')
            reasoning.append(f"Cash Flow: Low FCF yield of {fcf_yield:.1%}")
        else:
            signals.append('neutral')
            reasoning.append(f"Cash Flow: Moderate FCF yield of {fcf_yield:.1%}")
    else:
        signals.append('neutral')
        reasoning.append("Cash Flow: Insufficient data for analysis")
    
    # Count the signals
    bullish_signals = signals.count('bullish')
    bearish_signals = signals.count('bearish')
    neutral_signals = signals.count('neutral')
    
    # Determine overall signal
    if bullish_signals > bearish_signals:
        overall_signal = 'bullish'
    elif bearish_signals > bullish_signals:
        overall_signal = 'bearish'
    else:
        overall_signal = 'neutral'
    
    # Calculate confidence level
    total_signals = len(signals)
    max_signals = max(bullish_signals, bearish_signals, neutral_signals)
    confidence = max_signals / total_signals
    
    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence * 100)}%",
        "reasoning": " | ".join(reasoning)
    }
    
    # Create the fundamental analysis message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="fundamentals_agent",
    )
    
    # Print the reasoning if the flag is set
    if show_reasoning:
        show_agent_reasoning(message_content, "Fundamental Analysis Agent")
    
    return {
        "messages": [message],
        "data": data,
    }