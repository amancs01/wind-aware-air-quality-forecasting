# Project Architecture

This document describes the overall architecture of the project.

This answers: 
  How is the entire project organized?

Data Collection

↓

Validation

↓

Merge

↓

Preprocessing

↓

Feature Engineering

↓

Model Training

↓

Evaluation

↓

Dashboard

# Current Architecture

Project Layers

User Script
    ↓
API Layer
    ↓
Utilities
    ↓
Configuration
    ↓
Filesystem

Current Components

config.py
Central configuration for API endpoints, paths and constants.

utils.py
Shared filesystem utilities.

api.py
OpenAQ API communication.

download_weather.py
Weather ingestion pipeline.

download_air_quality.py
Air Quality ingestion pipeline.

## Planned Refactor

Upcoming architecture:

downloaders
        ↓
API Modules
        ↓
Reusable HTTP Client
        ↓
requests