from typing import Dict, Any, List
from .base import BaseFeatureGenerator
from .generators import SpamFeatureGenerator, AverageWordLengthFeatureGenerator, EmailEmbeddingsFeatureGenerator, RawEmailFeatureGenerator, NonTextCharacterFeatureGenerator
from app.dataclasses import Email

# Constant list of available generators
GENERATORS = {
    "spam": SpamFeatureGenerator,
    "word_length": AverageWordLengthFeatureGenerator,
    "email_embeddings": EmailEmbeddingsFeatureGenerator,
    "raw_email": RawEmailFeatureGenerator,
    "non_text": NonTextCharacterFeatureGenerator
}

class FeatureGeneratorFactory:
    """Factory for creating and managing feature generators"""
    
    def __init__(self):
        self._generators = GENERATORS
    
    def generate_all_features(self, email: Email, 
                            generator_names: List[str] = None) -> Dict[str, Any]:
        """Generate features using multiple generators"""
        if generator_names is None:
            generator_names = list(self._generators.keys())
        
        all_features = {}
        
        for gen_name in generator_names:
            generator_class = self._generators[gen_name]
            generator = generator_class()
            features = generator.generate_features(email)
            
            # Prefix features with generator name to avoid conflicts
            for feature_name, value in features.items():
                prefixed_name = f"{gen_name}_{feature_name}"
                all_features[prefixed_name] = value
        
        return all_features