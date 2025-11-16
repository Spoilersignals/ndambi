# 📦 NDAMBI Stock Market Forecasting System - Installation Guide

## System Requirements

- **Operating System:** Windows 10/11, macOS, or Linux
- **Python:** 3.8 or higher
- **RAM:** Minimum 4GB (8GB recommended)
- **Storage:** At least 2GB free space
- **Internet Connection:** Required for data collection

---

## 🚀 Installation Steps for New Computer

### Step 1: Install Python

1. Download Python from [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. During installation, **check "Add Python to PATH"**
3. Verify installation:
   ```bash
   python --version
   ```

### Step 2: Install Git

1. Download Git from [https://git-scm.com/downloads](https://git-scm.com/downloads)
2. Install with default settings
3. Verify installation:
   ```bash
   git --version
   ```

### Step 3: Clone the Project

Open Command Prompt or Terminal and run:

```bash
# Navigate to your desired location
cd Desktop

# Clone the repository
git clone https://github.com/Spoilersignals/ndambi.git

# Enter the project directory
cd ndambi
```

### Step 4: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 5: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Download NLTK data (for sentiment analysis)
python -c "import nltk; nltk.download('punkt'); nltk.download('brown')"
```

### Step 6: Set Up Environment Variables (Optional - for Email Verification)

Create a `.env` file in the project root:

```bash
# Copy the example file
copy .env.example .env
```

Edit `.env` with your email credentials (if you want email verification):

```env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-app-password
```

**Note:** For Gmail, you need to generate an [App Password](https://support.google.com/accounts/answer/185833)

### Step 7: Verify Installation

```bash
# Check if all modules are installed correctly
python -c "import tensorflow, pandas, streamlit, yfinance; print('All packages installed successfully!')"
```

---

## 🎯 Running the System

### Option 1: Interactive CLI Menu (Recommended for Beginners)

```bash
python main.py
```

Then choose from:
1. Collect Stock Data
2. Preprocess Data
3. Train ANN Model
4. Train LSTM Model
5. Evaluate ANN Model
6. Evaluate LSTM Model
7. Run Complete Pipeline
8. Launch Dashboard (Streamlit)

### Option 2: Streamlit Dashboard (Recommended)

```bash
streamlit run app.py
```

The dashboard will open in your browser at: `http://localhost:8501`

### Option 3: Individual Scripts

```bash
# Collect data
python src/data_collection.py

# Preprocess data
python src/preprocessing.py

# Train ANN model
python src/train.py

# Train LSTM model
python src/train.py lstm

# Evaluate ANN model
python src/evaluate.py ann

# Evaluate LSTM model
python src/evaluate.py lstm
```

---

## 📊 Quick Start Guide

### First Time Setup (Complete Pipeline)

```bash
# 1. Activate virtual environment
venv\Scripts\activate

# 2. Run the main menu
python main.py

# 3. Follow these steps in order:
#    - Option 1: Collect Stock Data (enter symbol: AAPL)
#    - Option 2: Preprocess Data
#    - Option 3: Train ANN Model
#    - Option 5: Evaluate ANN Model
#    - Option 8: Launch Dashboard
```

---

## 📁 Project Structure

```
NDAMBI/
├── data/
│   ├── raw/              # Raw stock data (CSV files)
│   └── processed/        # Preprocessed data (NumPy arrays)
├── models/               # Trained models (.h5 files)
├── reports/              # Evaluation reports and charts
├── src/
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── app.py               # Streamlit dashboard
├── auth.py              # Authentication system
├── config.py            # Configuration settings
├── main.py              # CLI menu
├── requirements.txt     # Python dependencies
├── .env.example         # Email setup template
└── users.db            # SQLite database (auto-created)
```

---

## 🔧 Troubleshooting

### Issue: "Module not found" error

**Solution:**
```bash
# Make sure virtual environment is activated
venv\Scripts\activate

# Reinstall packages
pip install -r requirements.txt
```

### Issue: TensorFlow installation fails

**Solution:**
```bash
# For Windows with no GPU
pip install tensorflow-cpu

# For systems with GPU
pip install tensorflow
```

### Issue: Streamlit doesn't open in browser

**Solution:**
```bash
# Manually open the URL shown in terminal
# Usually: http://localhost:8501
```

### Issue: Email verification not working

**Solution:**
- Skip email setup by using the verification code shown in the terminal
- Or properly configure `.env` with Gmail App Password

### Issue: Data collection fails

**Solution:**
- Check internet connection
- Verify stock symbol is correct (use uppercase, e.g., AAPL)
- Try a different stock symbol

---

## 📚 Additional Resources

- **Documentation:** See `README.md` in project root
- **Setup Guide:** Check `SETUP_GUIDE.md`
- **Email Setup:** Read `EMAIL_SETUP.md`
- **Quick Start:** View `QUICK_START.txt`

---

## 🎓 For Academic Submission

### Required Files to Include:
1. All source code (`src/` folder)
2. Configuration files (`config.py`, `requirements.txt`)
3. Sample outputs (screenshots from `reports/`)
4. Documentation (`README.md`, this guide)
5. Project proposal (`file.txt`)

### Data Collection for Demonstration:
```bash
# Collect data from 2019 to 2025
python main.py
# Choose Option 1
# Enter symbol: AAPL
# Data will cover 2019-01-01 to today (2025)
```

---

## 👤 Contact & Support

**Project Author:** Ndabi Joseph Muiruri  
**Institution:** Kabarak University  
**Repository:** [https://github.com/Spoilersignals/ndambi](https://github.com/Spoilersignals/ndambi)

---

## ✅ Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Git installed
- [ ] Repository cloned
- [ ] Virtual environment created and activated
- [ ] Dependencies installed
- [ ] NLTK data downloaded
- [ ] System tested with `python main.py`
- [ ] Dashboard launched with `streamlit run app.py`

**Installation Complete! 🎉**
