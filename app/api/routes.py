import json
import uuid
import numpy as np
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.services.email_topic_inference import EmailTopicInferenceService
from app.features.factory import GENERATORS
from app.dataclasses import Email

router = APIRouter()
TOPICS_FILE = Path("data/topic_keywords.json")
EMAILS_FILE = Path("data/stored_emails.json")

class EmailRequest(BaseModel):
    subject: str
    body: str

class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: str

class EmailClassificationResponse(BaseModel):
    predicted_topic: str
    topic_scores: Dict[str, float]
    features: Dict[str, Any]
    available_topics: List[str]

class EmailAddResponse(BaseModel):
    message: str
    email_id: int
    
class TopicCreateRequest(BaseModel):
    name: str
    description: str
    
class EmailStoreRequest(BaseModel):
    subject: str
    body: str
    ground_truth: Optional[str]=None

@router.post("/emails/classify", response_model=EmailClassificationResponse)
async def classify_email(request: EmailRequest, mode: str = Query("topic")):
    try:
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)
        ## mode = topic 
        if mode == "topic":
            result = inference_service.classify_email(email)
            return EmailClassificationResponse(
                predicted_topic=result["predicted_topic"],
                topic_scores=result["topic_scores"],
                features=result["features"],
                available_topics=result["available_topics"],
            )
        
        ## mode = email stored email classification to nearest
        if mode == "email":
            # computinf embedding for incoming email using existing pipeline
            incoming = inference_service.classify_email(email)
            incoming_vec = np.array(incoming["features"]["email_embeddings_average_embedding"], dtype=float)
            
            # loading stored emails from the JSON file
            stored_all = json.loads(EMAILS_FILE.read_text()) if EMAILS_FILE.exists() else []
            
            # using only stored emails that have real ground_truth label
            stored =[
                s for s in stored_all
                if isinstance(s, dict)
                and isinstance(s.get("ground_truth"), str)
                and s["ground_truth"].strip() != ""
            ]
            
            # ensuring at least 1 labeled email exists
            if len(stored) == 0:
                raise HTTPException(status_code=400, detail="No labeled stored emails with ground_truth available for mode=email")

            best = None
            best_sim = -1.0
            
            # Comparing to each stored labeled email
            for s in stored:
                s_email = Email(subject=s("subject",""), body=s("body",""))
                s_result = inference_service.classify_email(s_email)
                s_vec = np.array(s_result["features"]["email_embeddings_average_embedding"], dtype=float)

                denom = (np.linalg.norm(incoming_vec) * np.linalg.norm(s_vec))
                sim = float(np.dot(incoming_vec, s_vec) / denom) if denom != 0 else -1.0

                if sim > best_sim:
                    best_sim = sim
                    best = s

            predicted = best["ground_truth"].strip()

            # Return in same response model 
            return EmailClassificationResponse(
                predicted_topic=predicted,
                topic_scores={"nearest_email_similarity": best_sim},
                features=incoming["features"],
                available_topics=incoming["available_topics"],
            )
        raise HTTPException(status_code=400, detail="Invalid mode. Use mode=topic or mode=email")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
@router.post("/emails")
async def store_email(request: EmailStoreRequest):
    """ Stores and email with optional ground truth label"""
    try:
        emails = json.loads(EMAILS_FILE.read_text()) if EMAILS_FILE.exists() else []
        
        item = {
            "id": str(uuid.uuid4()),
            "subject": request.subject,
            "body" : request.body,
            "ground_truth": request.ground_truth,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        
        emails.append(item)
        
        tmp = EMAILS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(emails, indent=2))
        tmp.replace(EMAILS_FILE)

        return {"message": "email stored", "email": item}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/topics")
async def topics():
    """Get available email topics"""
    inference_service = EmailTopicInferenceService()
    info = inference_service.get_pipeline_info()
    return {"topics": info["available_topics"]}
    
@router.post("/topics")
async def add_topic(request: TopicCreateRequest):
    """Create a topic and persist it to data/topic_keywords.json"""
    try:
        if TOPICS_FILE.exists():
           topics = json.loads(TOPICS_FILE.read_text())
        else:
           topics = {}
    
        topic_name = request.name.strip().lower()
        # matching to JSON structure
        topics[topic_name] = {"description": request.description.strip()}
        # atomic write to prevent partial/corrupt file
        tmp = TOPICS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(topics, indent=2))
        tmp.replace(TOPICS_FILE)
        
        return {"message": "topic saved", "topics": list(topics.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/features")
async def features():
    """Get available feature generators and their feature names"""
    try:
        available = []
        for name, generator_class in GENERATORS.items():
            gen = generator_class()
            available.append(
                {"name": name, "features": gen.feature_names}
                )
        return {"available_generators": available}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline/info") 
async def pipeline_info():
    inference_service = EmailTopicInferenceService()
    return inference_service.get_pipeline_info()

# TODO: LAB ASSIGNMENT - Part 2 of 2  
# Create a GET endpoint at "/features" that returns information about all feature generators
# available in the system.
#
# Requirements:
# 1. Create a GET endpoint at "/features"
# 2. Import FeatureGeneratorFactory from app.features.factory
# 3. Use FeatureGeneratorFactory.get_available_generators() to get generator info
# 4. Return a JSON response with the available generators and their feature names
# 5. Handle any exceptions with appropriate HTTP error responses
#
# Expected response format:
# {
#   "available_generators": [
#     {
#       "name": "spam",
#       "features": ["has_spam_words"]
#     },
#     ...
#   ]
# }
#
# Hint: Look at the existing endpoints above for patterns on error handling
# Hint: You may need to instantiate generators to get their feature names

