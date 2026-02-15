from typing import Dict, Any
from .base import BaseFeatureGenerator
from app.dataclasses import Email
from sentence_transformers import SentenceTransformer
import numpy as np
import re

class SpamFeatureGenerator(BaseFeatureGenerator):
    """Generates spam detection features from email content"""
    
    def generate_features(self, email: Email) -> Dict[str, Any]:
        # Extract email content from dataclass
        subject = email.subject
        body = email.body
        all_text = f"{subject} {body}".lower()
        
        # Spam word detection
        spam_words = ['free', 'winner', 'congratulations', 'click here', 'limited time', 
                      'act now', 'urgent', 'special offer', 'guaranteed', 'no risk',
                      'cash', 'money back', 'amazing', 'incredible', 'unbeatable']
        
        has_spam_words = int(any(word in all_text for word in spam_words))
        
        return {"has_spam_words": has_spam_words}
    
    @property
    def feature_names(self) -> list[str]:
        return ["has_spam_words"]


class AverageWordLengthFeatureGenerator(BaseFeatureGenerator):
    """Generates average word length feature from email content"""
    
    def generate_features(self, email: Email) -> Dict[str, Any]:
        # Extract email content from dataclass
        subject = email.subject
        body = email.body
        all_text = f"{subject} {body}"
        
        # Split into words and calculate average length
        words = all_text.split()
        if not words:
            average_word_length = 0.0
        else:
            total_length = sum(len(word) for word in words)
            average_word_length = total_length / len(words)
        
        return {"average_word_length": average_word_length}
    
    @property
    def feature_names(self) -> list[str]:
        return ["average_word_length"]


class EmailEmbeddingsFeatureGenerator(BaseFeatureGenerator):
    """Generates embedding features using sentence transformers"""

    # Class-level model instance (loaded once and shared)
    _model = None

    @classmethod
    def _get_model(cls):
        """Lazy load the sentence transformer model"""
        if cls._model is None:
            # Use a small, efficient model (80MB, fast inference)
            cls._model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._model

    def generate_features(self, email: Email) -> Dict[str, Any]:
        # Extract email content from dataclass
        subject = email.subject
        body = email.body

        # Combine subject and body for embedding
        email_text = f"{subject} {body}"

        # Generate embedding using sentence transformer
        model = self._get_model()
        embedding = model.encode(email_text, convert_to_numpy=True)

        # Convert to list for JSON serialization
        embedding_list = embedding.tolist()
        
        return {"average_embedding": embedding_list}

    @property
    def feature_names(self) -> list[str]:
        return ["average_embedding"]


class RawEmailFeatureGenerator(BaseFeatureGenerator):
    """Extracts raw email data as features"""
    
    def generate_features(self, email: Email) -> Dict[str, Any]:
        # Extract email content from dataclass
        subject = email.subject
        body = email.body
        
        return {
            "email_subject": subject,
            "email_body": body
        }
    
    @property
    def feature_names(self) -> list[str]:
        return ["email_subject", "email_body"]
        
        
class NonTextCharacterFeatureGenerator(BaseFeatureGenerator):
    """Counts non-alphanumeric characters in subject + body"""
    
    def generate_features(self, email:Email) -> Dict[str, Any]:
        text = f"{email.subject}{email.body}"
        count = len(re.findall(r"[^a-zA-Z0-9]", text))
        return {"non_text_char_count": count}
        
        
    @property
    def feature_names(self) -> list[str]:
        return ["non_text_char_count"]
    

# TODO: Lab Assignment - Part 0 of 2
# Extend the embedding feature generator to include the email body as well as the subject


# TODO: LAB ASSIGNMENT - Part 1 of 2
# Create a NonTextCharacterFeatureGenerator class that counts non-alphanumeric characters
# (punctuation, symbols, etc.) in the email subject and body text.
# 
# Requirements:
# 1. Inherit from BaseFeatureGenerator
# 2. Implement generate_features(self, email: Email) -> Dict[str, Any]:
#    - Count non-alphanumeric characters in both subject and body
#    - Return a dictionary with "non_text_char_count" as the key
# 3. Implement the feature_names property to return ["non_text_char_count"]
# 4. Follow the same pattern as the other generators above
# 
# Hint: Use email.subject and email.body to access the text
# Hint: You can use regex or string methods to count non-alphanumeric characters