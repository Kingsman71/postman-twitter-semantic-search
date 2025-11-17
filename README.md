# Postman Twitter API - Semantic Search

A tool that enables semantic search across the Twitter API v2 Postman collection using AI-powered embeddings.

## Overview

This project provides a command-line utility to search through Twitter API v2 endpoints semantically. Instead of exact keyword matching, it uses machine learning embeddings to understand the meaning of your queries and find the most relevant API endpoints.

## Features

- **Semantic Search**: Search API endpoints by meaning, not just keywords
- **AI-Powered**: Uses `sentence-transformers` for generating embeddings
- **Fast Retrieval**: Leverages FAISS for efficient similarity search
- **Postman Integration**: Works with the official Twitter API v2 Postman collection

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`:
  - `sentence-transformers`
  - `faiss-cpu`
  - `numpy`

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the semantic search utility:
   ```bash
   python semantic_search.py --query "your search query"
   ```

## How It Works

1. **Loads** the Twitter API v2 Postman collection
2. **Extracts** all API request information (methods, URLs, parameters, descriptions)
3. **Generates** embeddings using sentence transformers
4. **Indexes** them with FAISS for fast similarity search
5. **Returns** the most relevant endpoints based on your query

## Files

- `semantic_search.py` - Main utility script
- `Twitter API v2.postman_collection.json` - Official Twitter API v2 endpoints
- `Twitter API v2.postman_environment.json` - Environment variables for Postman
- `requirements.txt` - Python dependencies
