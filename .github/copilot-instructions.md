# Copilot Instructions for Python Learning Repository

## Repository Overview
This is an **educational repository** from the CodeBasics YouTube channel containing Python tutorials, exercises, and practical examples. It's designed for beginners learning Python, data science, and machine learning - not a production application.

## Project Structure & Organization

### Learning Path Progression
1. **Basics/** - Start here for Python fundamentals (variables, functions, loops, classes, file I/O, JSON)
2. **Advanced/** - Decorators, FastAPI, regex patterns
3. **Modules/** - Standard library tutorials (argparse, pandas, urllib)
4. **DataScience/** - Complete end-to-end projects (BangloreHomePrices, CelebrityFaceRecognition)
5. **ML/** - 19 numbered machine learning topics (1_linear_reg → 19_Bagging)
6. **DeepLearningML/** - Neural networks and deep learning (numbered 1-22)
7. **Library-specific folders**: numpy/, pandas/, matpltlib/, jupyter/

### Numbered Topic Convention
ML and DeepLearningML topics follow a **numbered progression** (e.g., `1_linear_reg/`, `2_linear_reg_multivariate/`). This numbering indicates recommended learning order, not arbitrary organization.

## Code Patterns & Conventions

### Jupyter Notebooks as Primary Format
- Most tutorials use `.ipynb` notebooks with markdown explanations and executable code cells
- Notebooks include visual aids (images like `homepricetable.JPG`, `scatterplot.JPG`)
- Use `%matplotlib inline` for inline plotting
- Educational notebooks contain step-by-step explanations with headings like "Problem Statement"

### Python Script Patterns
- Standalone `.py` files accompany notebooks (e.g., `linearReg.py` alongside `1_linear_regression.ipynb`)
- Scripts follow simple, readable patterns for beginners - **no complex abstractions**
- Standard imports: pandas, numpy, sklearn, matplotlib

### Data Files Convention
Each tutorial folder contains its own **CSV data files** (e.g., `homeprices.csv`, `salaries.csv`, `weather_data.csv`) - data is co-located with code, not centralized.

### Exercise Structure
- `Exercise/` subfolders contain practice problems with solutions
- Exercises often include detailed comments explaining problem requirements
- Example: `Basics/Exercise/9_for/9_for_exercise.py`

## Technology Stack by Topic

**Data Science Projects** (DataScience/):
- Flask servers for model serving
- sklearn for ML models
- Frontend: HTML/CSS/JavaScript
- Deployment: AWS EC2 + nginx (see BangloreHomePrices/README.md)

**ML Algorithms** (ML/):
- Primary: scikit-learn (sklearn)
- Data: pandas, numpy
- Visualization: matplotlib
- Model persistence: pickle/joblib

**Deep Learning** (DeepLearningML/):
- TensorFlow/Keras
- Topics: activation functions, gradient descent, CNNs, transfer learning, word embeddings
- GPU benchmarking examples included

**Testing** (unittesting_pytest/):
- pytest framework
- Patterns: fixtures, parametrize, custom markers
- Naming: `test_*.py` files

## Key Workflows

### Running Notebooks
Open `.ipynb` files in VS Code or Jupyter. Most require:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

### Testing
```bash
pytest unittesting_pytest/
pytest -m windows  # custom markers
```

### Data Science Project Structure (BangloreHomePrices example)
```
model/ - Jupyter notebook + saved model
server/ - Flask server (server.py, requirements.txt)
client/ - HTML/CSS/JS frontend
```

## Important Distinctions

### NOT a Production Codebase
- No complex project architecture or microservices
- No CI/CD pipelines or docker containers
- Focus on **pedagogical clarity over engineering best practices**

### Educational Code Style
- Verbose variable names for clarity
- Extensive comments explaining logic
- Simple function designs without heavy abstraction
- Direct, procedural code over OOP when teaching basics

### Hindi Language Support
Some basic tutorials have Hindi versions (`Basics/Hindi/`) - parallel content in Indian regional language.

## When Creating New Content

1. **Follow numbering convention** if adding to ML/DeepLearningML sequences
2. **Include CSV data files** in the same folder as notebooks
3. **Add markdown cells** explaining concepts, not just code
4. **Use simple, beginner-friendly patterns** - avoid advanced Python features unless teaching them
5. **Include visual assets** (JPG/PNG) for complex concepts
6. **Add Exercise subfolder** if creating a tutorial series

## Common Imports Reference
```python
# Data manipulation
import pandas as pd
import numpy as np

# Visualization  
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

# Deep Learning
import tensorflow as tf
from tensorflow import keras

# Utilities
import json
import math
import time
```

## Repository Context
- Owner: codebasics
- YouTube: https://www.youtube.com/channel/UCh9nVJoWXmFb7sLApWGcLPQ
- Purpose: Accompany video tutorials for Python/ML/DS learning
- Contributions welcome for additional learning examples
