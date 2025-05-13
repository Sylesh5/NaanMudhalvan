import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def display_welcome():
    print("""
    ====================================
    VIRTUAL ASSISTANT INTENT RECOGNITION
    ====================================
    This program classifies user queries into different intent categories.
    It will first train a model, then demonstrate how it understands commands.
    """)

def load_training_data():
    """Create and display the training dataset"""
    data = {
        "text": [
            "what's the weather today",
            "how's the weather in new york",
            "play some music",
            "play my favorite song",
            "set an alarm for 7 am",
            "remind me to call mom at 5 pm",
            "what's on my calendar",
            "what are my appointments for tomorrow",
            "tell me a joke",
            "make me laugh",
            "order pizza",
            "find me a restaurant",
            "good morning",
            "hello",
            "hi there"
        ],
        "intent": [
            "weather",
            "weather",
            "play_music",
            "play_music",
            "set_alarm",
            "set_reminder",
            "check_calendar",
            "check_calendar",
            "tell_joke",
            "tell_joke",
            "order_food",
            "find_restaurant",
            "greeting",
            "greeting",
            "greeting"
        ]
    }
    
    df = pd.DataFrame(data)
    print("\nTRAINING DATA EXAMPLES:")
    print(df.sample(5).to_string(index=False))  # Show random samples
    print(f"\nTotal training examples: {len(df)}")
    print("Intent categories:", df['intent'].unique())
    return df

def train_model(df):
    """Train and evaluate the intent recognition model"""
    print("\nTRAINING THE MODEL...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['intent'], test_size=0.2, random_state=42
    )
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC())
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nMODEL EVALUATION:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return model

def recognize_intent_interactive(model):
    """Interactive mode for testing the model"""
    print("\nINTERACTIVE MODE: (type 'quit' to exit)")
    print("Try phrases like: 'what's the weather', 'play some jazz', 'set an alarm'")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        if not user_input.strip():
            print("Please enter a command")
            continue
            
        result = recognize_intent(model, user_input)
        display_result(result)

def recognize_intent(model, text):
    """Predict intent with confidence scores"""
    # Get prediction probabilities
    decision_scores = model.decision_function([text])[0]
    prob_scores = np.exp(decision_scores) / np.sum(np.exp(decision_scores))
    confidence = np.max(prob_scores)
    
    # Get top 3 possible intents
    top_3_indices = np.argsort(prob_scores)[-3:][::-1]
    top_intents = model.classes_[top_3_indices]
    top_confidences = prob_scores[top_3_indices]
    
    return {
        "text": text,
        "top_intent": model.predict([text])[0],
        "confidence": float(confidence),
        "possible_intents": list(zip(top_intents, top_confidences))
    }

def display_result(result):
    """Display the recognition results in a user-friendly way"""
    print("\nRESULT:")
    print(f"Input: '{result['text']}'")
    print(f"Main Intent: {result['top_intent']} (confidence: {result['confidence']:.1%})")
    
    print("\nOther Possible Intents:")
    for intent, confidence in result['possible_intents'][1:]:
        print(f"- {intent}: {confidence:.1%} confidence")
    
    print("\nSUGGESTED RESPONSES:")
    suggest_response(result['top_intent'])

def suggest_response(intent):
    """Provide example responses for each intent"""
    responses = {
        "weather": "I can check the weather. Would you like the forecast for your current location?",
        "play_music": "Playing music from your library. Any specific genre or artist?",
        "set_alarm": "Alarm set. Would you like to give it a name?",
        "set_reminder": "I'll remind you. Should I set any additional details?",
        "check_calendar": "Here are your upcoming events...",
        "tell_joke": "Why don't scientists trust atoms? Because they make up everything!",
        "order_food": "I can help with food delivery. What cuisine would you like?",
        "find_restaurant": "Finding restaurants near you... Any dietary preferences?",
        "greeting": "Hello! How can I assist you today?"
    }
    print(responses.get(intent, "I'm not sure how to respond to that."))

def main():
    display_welcome()
    df = load_training_data()
    model = train_model(df)
    recognize_intent_interactive(model)
    print("\nThank you for using the Intent Recognition System!")

if __name__ == "__main__":
    main()