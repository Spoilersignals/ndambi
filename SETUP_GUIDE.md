# 🚀 Quick Setup Guide

## Prerequisites

Before you begin, ensure you have:
- ✅ Python 3.8 or higher installed
- ✅ pip package manager
- ✅ Internet connection (for downloading data)
- ✅ At least 4GB free disk space
- ✅ 8GB RAM recommended

## Step-by-Step Installation

### Step 1: Navigate to Project Directory

```bash
cd c:\Users\pesak\Desktop\NDAMBI
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages (~2-3 minutes).

### Step 4: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('brown')"
```

### Step 5: Configure Environment (Optional)

```bash
copy .env.example .env
```

Edit `.env` if you have API keys for Alpha Vantage or News API (not required for basic functionality).

## Verification

Test if everything is installed correctly:

```bash
python -c "import tensorflow; import pandas; import streamlit; print('✓ All packages installed successfully!')"
```

## First Run

### Option A: Interactive Menu (Recommended)

```bash
python main.py
```

Follow the menu:
1. Choose option 1 to collect data (default: AAPL stock)
2. Choose option 2 to preprocess data
3. Choose option 3 to train ANN model
4. Choose option 5 to evaluate model
5. Choose option 8 to launch dashboard

### Option B: Quick Pipeline

```bash
python main.py
```

Then select option 7 (Run Complete Pipeline) to automate all steps.

### Option C: Dashboard First

```bash
streamlit run app.py
```

Access at: http://localhost:8501

## Expected Timeline

- Data Collection: ~30-60 seconds
- Preprocessing: ~15-30 seconds
- Model Training: ~5-15 minutes (depending on CPU/GPU)
- Evaluation: ~10-20 seconds

## Troubleshooting

### Issue: "No module named 'tensorflow'"

**Solution:**
```bash
pip install tensorflow==2.13.0
```

### Issue: "Could not find data for symbol"

**Solution:**
- Check internet connection
- Try a different symbol (e.g., GOOGL, MSFT)
- Wait a few seconds and retry

### Issue: "Out of memory"

**Solution:**
Edit `config.py`:
```python
BATCH_SIZE = 16  # Reduce from 32
```

### Issue: TensorFlow GPU warnings

**Solution:**
These are typically harmless. To suppress:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

## System Requirements

### Minimum
- CPU: Dual-core processor
- RAM: 4GB
- Disk: 2GB free space
- OS: Windows 10, Ubuntu 18.04, macOS 10.14+

### Recommended
- CPU: Quad-core processor or GPU
- RAM: 8GB or more
- Disk: 5GB free space
- OS: Windows 11, Ubuntu 20.04+, macOS 11+

## Next Steps

After successful setup:

1. **Read the README.md** for detailed documentation
2. **Check AGENTS.md** for developer guidelines
3. **Experiment with different stocks** using the CLI
4. **Explore the dashboard** for visualizations
5. **Modify hyperparameters** in config.py

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Review the Troubleshooting section above
3. Ensure all dependencies are installed
4. Verify Python version (3.8+)

## Uninstallation

To completely remove the environment:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment folder
rmdir /s venv  # Windows
rm -rf venv    # Linux/Mac
```

## Updates

To update dependencies:

```bash
pip install --upgrade -r requirements.txt
```

---

**Setup complete! 🎉 You're ready to start forecasting stock prices!**
