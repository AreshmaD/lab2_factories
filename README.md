# Email Topic Classification Lab - Factory Pattern

This lab demonstrates the Factory Pattern in machine learning feature generation and email topic classification using cosine similarity.

## Overview

The system classifies emails into topics (work, personal, promotion, newsletter, support) using:
- **Factory Pattern** for feature generation
- **Embedding-based similarity** using cosine distance
- **RESTful API** for classification and data management

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
# Run on all interfaces (required for EC2 access)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Access the API
- **Local**: http://localhost:8000
- **EC2**: http://YOUR_EC2_PUBLIC_IP:8000
- **API Documentation (Swagger)**: http://YOUR_EC2_PUBLIC_IP:8000/docs
- **Alternative Docs (ReDoc)**: http://YOUR_EC2_PUBLIC_IP:8000/redoc

## Getting Started - Explore the System

### 1. View Available Topics
```bash
curl http://YOUR_EC2_PUBLIC_IP:8000/api/v1/topics
```

### 2. Classify an Email
```bash
curl -X POST "http://YOUR_EC2_PUBLIC_IP:8000/api/v1/emails/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Meeting tomorrow at 2pm",
    "body": "Let'\''s discuss the quarterly budget and project deadlines"
  }'
```

### 3. Get Pipeline Information
```bash
curl http://YOUR_EC2_PUBLIC_IP:8000/api/v1/pipeline/info
```

### 4. Interactive API Documentation
Visit `http://YOUR_EC2_PUBLIC_IP:8000/docs` in your browser for:
- Interactive API testing
- Request/response examples
- Schema definitions

## Understanding the Architecture

### Factory Pattern Implementation
- **Location**: `app/features/factory.py`
- **Generators**: `app/features/generators.py` 
- **Pattern**: See `GENERATORS` constant for available feature generators

### Feature Generators
1. **SpamFeatureGenerator** - Detects spam keywords
2. **AverageWordLengthFeatureGenerator** - Calculates average word length
3. **EmailEmbeddingsFeatureGenerator** - Creates embeddings from email length
4. **RawEmailFeatureGenerator** - Extracts raw email text

### Classification Model
- **Location**: `app/models/similarity_model.py`
- **Method**: Cosine similarity between email embeddings and topic description embeddings
- **Topics**: Stored in `data/topic_keywords.json`

## Key Files to Examine

1. **`app/features/factory.py`** - Factory pattern implementation
2. **`app/features/generators.py`** - Feature generator classes
3. **`app/models/similarity_model.py`** - Classification logic
4. **`app/api/routes.py`** - REST API endpoints
5. **`data/topic_keywords.json`** - Topic definitions and descriptions

## Lab Activities

During this lab session, you will:

### 1. Explore the Factory Pattern
- Examine `app/features/factory.py` to understand how the Factory Pattern creates different feature generators
- Look at `app/features/generators.py` to see the different feature generator implementations
- Test how adding new generators would extend the system's capabilities

### 2. Understand the Classification Pipeline
- Review `app/models/similarity_model.py` to see how cosine similarity works for email classification
- Examine `data/topic_keywords.json` to understand topic definitions
- Test the classification endpoint with different email examples

### 3. API Exploration and Testing
- Use the interactive Swagger documentation at `/docs` to explore all endpoints
- Test email classification with various email types (work, personal, promotional, etc.)
- Experiment with the pipeline information endpoint to understand system internals

### 4. Code Analysis and Discussion
- Discuss how the Factory Pattern makes the feature generation system extensible
- Analyze the trade-offs between embedding-based similarity and other classification approaches
- Explore how the system handles different types of email content

### 5. System Architecture Review
- Map out the data flow from email input to classification output
- Identify the separation of concerns between different modules


## Learning Objectives

- Understand the **Factory Pattern** for extensible feature generation
- Learn **embedding-based similarity** for classification
- Analyze **REST API design** following proper conventions
- Explore **file-based data persistence** patterns
- Experience **machine learning pipeline** architecture

## Troubleshooting

- **Can't access from browser**: Make sure you're running with `--host 0.0.0.0`
- **Port issues**: Check that port 8000 is open in your EC2 security group
- **JSON errors**: Use the Swagger docs at `/docs` for proper request format
- **File permissions**: Ensure the `data/` directory is writable
