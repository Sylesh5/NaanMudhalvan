import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from googletrans import Translator
from textblob import TextBlob
import spacy
import warnings
warnings.filterwarnings('ignore')

# Load language models (run these commands first in terminal if needed):
# python -m spacy download en_core_web_sm
# python -m spacy download es_core_news_sm
# python -m spacy download fr_core_news_sm

class MultilingualChatbot:
    def __init__(self):
        self.translator = Translator()
        self.nlp = {
            'en': spacy.load('en_core_web_sm'),
            'es': spacy.load('es_core_news_sm'),
            'fr': spacy.load('fr_core_news_sm')
        }
        self.context = {}
        self.setup_intent_recognition()
        self.setup_qa_pairs()
        
    def setup_intent_recognition(self):
        # Multilingual training data (English, Spanish, French)
        data = {
            "text": [
                # English
                "what's the weather today", "how's the weather in new york",
                "play some music", "play my favorite song",
                "set an alarm for 7 am", "remind me to call mom",
                "what's on my calendar", "my appointments for tomorrow",
                "tell me a joke", "make me laugh",
                "order pizza", "find me a restaurant",
                
                # Spanish
                "qué tiempo hace hoy", "cómo está el clima en madrid",
                "pon música", "reproduce mi canción favorita",
                "pon una alarma a las 7", "recuérdame llamar a mamá",
                
                # French
                "quel temps fait-il aujourd'hui", "météo à paris",
                "joue de la musique", "mets ma chanson préférée"
            ],
            "intent": [
                "weather", "weather",
                "play_music", "play_music",
                "set_alarm", "set_reminder",
                "check_calendar", "check_calendar",
                "tell_joke", "tell_joke",
                "order_food", "find_restaurant",
                
                "weather", "weather",
                "play_music", "play_music",
                "set_alarm", "set_reminder",
                
                "weather", "weather",
                "play_music", "play_music"
            ]
        }
        
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LinearSVC())
        ])
        self.model.fit(data["text"], data["intent"])
    
    def setup_qa_pairs(self):
        self.qa_responses = {
            "weather": {
                "en": "I can check the weather. For which location?",
                "es": "Puedo consultar el clima. ¿Para qué ubicación?",
                "fr": "Je peux vérifier la météo. Pour quel endroit?"
            },
            "play_music": {
                "en": "Playing music. Any specific genre or artist?",
                "es": "Reproduciendo música. ¿Algún género o artista específico?",
                "fr": "Lecture de musique. Un genre ou artiste spécifique?"
            },
            "tell_joke": {
                "en": "Why don't scientists trust atoms? Because they make up everything!",
                "es": "¿Por qué los científicos no confían en los átomos? ¡Porque lo componen todo!",
                "fr": "Pourquoi les scientifiques ne font-ils pas confiance aux atomes? Parce qu'ils composent tout!"
            }
            # Add more responses as needed
        }
    
    def detect_language(self, text):
        try:
            lang = self.translator.detect(text).lang
            return lang if lang in ['en', 'es', 'fr'] else 'en'
        except:
            return 'en'
    
    def analyze_sentiment(self, text, lang='en'):
        analysis = TextBlob(text)
        if lang != 'en':
            analysis = TextBlob(str(self.translator.translate(text, dest='en').text))
        return analysis.sentiment.polarity
    
    def process_input(self, text, user_id='default'):
        # Detect language
        lang = self.detect_language(text)
        
        # Sentiment analysis
        sentiment = self.analyze_sentiment(text, lang)
        
        # Intent recognition
        intent = self.model.predict([text])[0]
        
        # Get response
        response = self.generate_response(intent, lang, sentiment, user_id)
        
        # Update context
        self.update_context(user_id, intent, lang, sentiment)
        
        return {
            "response": response,
            "language": lang,
            "intent": intent,
            "sentiment": "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral",
            "context": self.context.get(user_id, {})
        }
    
    def generate_response(self, intent, lang, sentiment, user_id):
        # Check if we're in a multi-turn conversation
        ctx = self.context.get(user_id, {})
        
        if intent in self.qa_responses:
            base_response = self.qa_responses[intent].get(lang, self.qa_responses[intent]['en'])
            
            # Add sentiment-aware phrasing
            if sentiment > 0.3:
                if lang == 'en': base_response = "Great! " + base_response
                elif lang == 'es': base_response = "¡Genial! " + base_response
                elif lang == 'fr': base_response = "Super! " + base_response
            elif sentiment < -0.3:
                if lang == 'en': base_response = "I understand this is frustrating. " + base_response
                elif lang == 'es': base_response = "Entiendo que esto es frustrante. " + base_response
                elif lang == 'fr': base_response = "Je comprends que c'est frustrant. " + base_response
            
            return base_response
        
        return self.qa_responses['default'].get(lang, "I didn't understand that.")
    
    def update_context(self, user_id, intent, lang, sentiment):
        if user_id not in self.context:
            self.context[user_id] = {
                'previous_intents': [],
                'language': lang,
                'sentiment_history': []
            }
        
        self.context[user_id]['previous_intents'].append(intent)
        self.context[user_id]['sentiment_history'].append(sentiment)
        self.context[user_id]['language'] = lang
        
        # Keep only last 5 interactions
        self.context[user_id]['previous_intents'] = self.context[user_id]['previous_intents'][-5:]
        self.context[user_id]['sentiment_history'] = self.context[user_id]['sentiment_history'][-5:]

# Interactive Demo
if __name__ == "__main__":
    bot = MultilingualChatbot()
    print("="*50)
    print("MULTILINGUAL CHATBOT SYSTEM")
    print("="*50)
    print("Supported languages: English, Spanish, French")
    print("Try phrases like:")
    print("- What's the weather today?")
    print("- ¿Qué tiempo hace en Madrid?")
    print("- Quel temps fait-il à Paris?")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        if not user_input.strip():
            continue
            
        response = bot.process_input(user_input)
        print("\nBot:")
        print(f"Response: {response['response']}")
        print(f"Detected Language: {response['language']}")
        print(f"Intent: {response['intent']}")
        print(f"Sentiment: {response['sentiment']}")
        print("-"*50)