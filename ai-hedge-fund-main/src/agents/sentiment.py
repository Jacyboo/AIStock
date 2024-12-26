from langchain_core.messages import HumanMessage
import json

from agents.state import show_agent_reasoning

def sentiment_agent(state):
    """Analyzes market sentiment and news impact"""
    show_reasoning = state["metadata"]["show_reasoning"]
    market_sentiment = state["data"]["manual_data"]["market_sentiment"]
    
    # Extract sentiment data
    overall_sentiment = market_sentiment["overall_sentiment"]
    confidence = min(float(market_sentiment["confidence"]), 1.0) * 100  # Ensure confidence is capped at 100%
    recent_news = market_sentiment["recent_news"]
    market_trends = market_sentiment["market_trends"]
    upcoming_events = market_sentiment["upcoming_events"]
    analyst_ratings = market_sentiment["analyst_ratings"]
    
    # Calculate bullish vs bearish signals from news
    bullish_signals = len([
        news for news in recent_news 
        if news["impact"] in ["very_positive", "positive"] or "bullish" in news["summary"].lower()
    ])
    bearish_signals = len([
        news for news in recent_news 
        if news["impact"] in ["very_negative", "negative"] or "bearish" in news["summary"].lower()
    ])
    
    # Calculate analyst sentiment
    total_ratings = sum(analyst_ratings.values())
    if total_ratings > 0:
        buy_ratio = (analyst_ratings["strong_buy"] + analyst_ratings["buy"]) / total_ratings
        sell_ratio = (analyst_ratings["strong_sell"] + analyst_ratings["sell"]) / total_ratings
    else:
        buy_ratio = sell_ratio = 0.33
    
    # Determine final sentiment with more decisive thresholds
    if overall_sentiment in ["very_bullish", "bullish"] or (bullish_signals > bearish_signals and buy_ratio > 0.6):
        signal = "bullish"
        confidence = max(confidence, 75)  # Minimum 75% confidence for strong signals
    elif overall_sentiment in ["very_bearish", "bearish"] or (bearish_signals > bullish_signals and sell_ratio > 0.6):
        signal = "bearish"
        confidence = max(confidence, 75)  # Minimum 75% confidence for strong signals
    else:
        signal = "neutral"
        confidence = 50  # Neutral signals get 50% confidence
    
    # Format the reasoning
    reasoning = f"Market Sentiment: {overall_sentiment} | "
    reasoning += f"News Analysis: {bullish_signals} bullish, {bearish_signals} bearish | "
    reasoning += f"Analyst Ratings: Buy={analyst_ratings['buy'] + analyst_ratings['strong_buy']}, "
    reasoning += f"Hold={analyst_ratings['hold']}, "
    reasoning += f"Sell={analyst_ratings['sell'] + analyst_ratings['strong_sell']}"
    if upcoming_events:
        reasoning += f" | Upcoming Events: {', '.join(upcoming_events[:2])}"  # Show only top 2 events
    
    result = {
        "signal": signal,
        "confidence": confidence,
        "reasoning": reasoning
    }
    
    # Create the sentiment message
    message = HumanMessage(
        content=json.dumps(result),
        name="sentiment_agent",
    )
    
    # Print the decision if the flag is set
    if show_reasoning:
        show_agent_reasoning(message.content, "Sentiment Analysis Agent")
    
    return {"messages": state["messages"] + [message]}
