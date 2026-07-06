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

## Current Data Pipeline

The project now follows a multi-stage data engineering pipeline:

Raw Data
↓
Validation
↓
Profiling
↓
Merge Weather + Air Quality
↓
Timestamp Alignment
↓
Trim Leading Missing PM2.5 Records
↓
Feature Engineering
↓
Model Training

Each stage writes its output to a separate directory, allowing intermediate datasets to be inspected and reproduced without modifying previous stages.