import math

from langchain_core.messages import HumanMessage

from agents.state import AgentState, show_agent_reasoning
from tools.api import prices_to_df

import json
import ast

##### Risk Management Agent #####
def parse_confidence(conf_str):
    """Parse confidence value from either string percentage or float"""
    if isinstance(conf_str, str):
        return float(conf_str.replace('%', '')) / 100.0
    elif isinstance(conf_str, (int, float)):
        return float(conf_str)
    else:
        return 0.5  # Default confidence if invalid format

def risk_management_agent(state):
    """Evaluates risk levels and adjusts position sizes"""
    show_reasoning = state["metadata"]["show_reasoning"]
    
    # Get the signals from other agents
    messages = state["messages"]
    agent_signals = {}
    
    for msg in messages:
        if msg.name in ["technical_analyst", "fundamentals_agent", "sentiment_agent"]:
            signal_data = json.loads(msg.content)
            agent_signals[msg.name] = signal_data
    
    # Risk assessment based on agent signals
    bearish_count = sum(1 for signal in agent_signals.values() if signal['signal'] == 'bearish')
    bullish_count = sum(1 for signal in agent_signals.values() if signal['signal'] == 'bullish')
    
    # Calculate average confidence of signals
    confidences = [parse_confidence(signal['confidence']) for signal in agent_signals.values()]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
    
    # Determine position size and confidence based on signal agreement
    if bearish_count >= 2:
        position_size = 0.0  # Exit position
        signal = "bearish"
        confidence = min(avg_confidence + 0.2, 0.95)  # Boost confidence but cap at 95%
    elif bullish_count >= 2:
        position_size = 1.0  # Full position
        signal = "bullish"
        confidence = min(avg_confidence + 0.2, 0.95)  # Boost confidence but cap at 95%
    else:
        # Mixed signals - position size based on bullish vs bearish ratio
        signal = "neutral"
        position_size = 0.5  # Half position
        confidence = max(0.5, avg_confidence)  # At least 50% confidence
    
    # Format the reasoning
    reasoning = f"Signal Agreement: {bullish_count} bullish, {bearish_count} bearish | "
    reasoning += f"Average Signal Confidence: {avg_confidence*100:.1f}% | "
    reasoning += f"Position Size: {position_size*100:.1f}%"
    
    result = {
        "signal": signal,
        "confidence": confidence * 100,  # Convert to percentage
        "position_size": position_size,
        "reasoning": reasoning
    }
    
    # Create the risk management message
    message = HumanMessage(
        content=json.dumps(result),
        name="risk_management_agent",
    )
    
    # Print the decision if the flag is set
    if show_reasoning:
        show_agent_reasoning(message.content, "Risk Management Agent")
    
    return {"messages": state["messages"] + [message]}

