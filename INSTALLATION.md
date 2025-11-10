# 📦 Installation Guide - Stock Market Forecasting System

## For New Users Installing This Project

Follow these steps to get the system running on your computer.

---

## ⚙️ System Requirements

- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 4GB (8GB recommended for training models)
- **Storage**: ~2GB free space

---

## 🚀 Step-by-Step Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Spoilersignals/ndambi.git
cd ndambi
```

Or download as ZIP from GitHub and extract it.

---

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal.

---

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- TensorFlow (Deep Learning)
- Streamlit (Dashboard)
- yfinance (Stock Data)
- pandas, numpy, scikit-learn
- And more...

⏱️ **This may take 5-10 minutes depending on your internet speed.**

---

### Step 4: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('brown')"
```

---

### Step 5: Configure Email (Optional)

If you want email verification to work:

1. Copy the example file:
   ```bash
   copy .env.example .env     # Windows
   cp .env.example .env       # macOS/Linux
   ```

2. Edit `.env` and add your Gmail credentials:
   ```
   SENDER_EMAIL=your-email@gmail.com
   SENDER_PASSWORD=your-app-password
   ```

3. Get Gmail App Password:
   - Go to https://myaccount.google.com/apppasswords
   - Generate password for "Mail"
   - Use the 16-digit code

**Note:** If you skip this, verification codes will display on screen instead of email.

---

### Step 6: Run the Application

#### Option A: Streamlit Dashboard (Recommended)

```bash
streamlit run app.py
```

Open your browser to: **http://localhost:8501**

#### Option B: Command-Line Interface

```bash
python main.py
```

---

## 📋 First-Time Usage

### 1. Create Account
- Click "Create Account"
- Fill in your details
- Get verification code (email or on-screen)
- Verify your account

### 2. Login
- Enter email and password
- Access the dashboard

### 3. Collect Stock Data
- Go to "Data Collection" page
- Enter stock symbol (e.g., AAPL, TSLA, GOOGL)
- Click "Fetch Data"

### 4. Preprocess Data
Run in terminal:
```bash
python -c "from src.preprocessing import DataPreprocessor; dp = DataPreprocessor(); dp.preprocess_and_split()"
```

### 5. Train Models
Run in terminal:
```bash
python src/train.py        # Train ANN
python src/train.py lstm   # Train LSTM
```

### 6. View Predictions
- Go to "Predictions" page in dashboard
- Select model type
- Generate predictions

---

## 🐛 Troubleshooting

### Problem: `ModuleNotFoundError`
**Solution:** Make sure virtual environment is activated and dependencies are installed:
```bash
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Problem: TensorFlow import errors
**Solution:** 
```bash
pip uninstall tensorflow
pip install tensorflow==2.13.0
```

### Problem: "Email not sending"
**Solution:** 
- Use Gmail App Password (not regular password)
- Or skip email config - codes will show on screen

### Problem: Streamlit won't start
**Solution:**
```bash
pip install --upgrade streamlit
streamlit run app.py
```

### Problem: Stock data not fetching
**Solution:**
```bash
pip install --upgrade yfinance
```

---

## 📁 Project Structure

```
NDAMBI/
├── venv/                          # Virtual environment (you create this)
├── data/
│   ├── raw/                       # Downloaded stock data
│   └── processed/                 # Preprocessed data for training
├── models/                        # Trained models (.h5 files)
├── src/
│   ├── data_collection.py         # Fetch stock data
│   ├── preprocessing.py           # Data preparation
│   ├── model.py                   # ANN & LSTM models
│   ├── train.py                   # Training script
│   └── evaluate.py                # Model evaluation
├── app.py                         # Streamlit dashboard
├── main.py                        # CLI interface
├── auth.py                        # Authentication system
├── requirements.txt               # Python dependencies
├── .env                           # Email config (you create this)
└── README.md                      # Project overview
```

---

## 🎯 Complete Workflow Example

```bash
# 1. Activate environment
venv\Scripts\activate

# 2. Run dashboard
streamlit run app.py

# 3. In browser:
#    - Create account & login
#    - Collect data for AAPL

# 4. In terminal (new window, venv activated):
python -c "from src.preprocessing import DataPreprocessor; dp = DataPreprocessor(); dp.preprocess_and_split()"

# 5. Train models
python src/train.py
python src/train.py lstm

# 6. View predictions in dashboard!
```

---

## ✅ Verification Checklist

After installation, verify everything works:

- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip list` shows packages)
- [ ] NLTK data downloaded
- [ ] Streamlit dashboard opens at http://localhost:8501
- [ ] Can create account and login
- [ ] Can fetch stock data
- [ ] Can preprocess data
- [ ] Can train models
- [ ] Can view predictions

---

## 📞 Getting Help

If you encounter issues:

1. Check `SETUP_GUIDE.md` for detailed instructions
2. Read `AUTHENTICATION_README.md` for auth issues
3. See `EMAIL_SETUP.md` for email configuration
4. Check GitHub Issues: https://github.com/Spoilersignals/ndambi/issues

---

## 🎓 For Academic Use

This is an academic project demonstrating:
- Artificial Neural Networks (ANN)
- Long Short-Term Memory (LSTM) networks
- Stock market prediction using deep learning
- User authentication systems
- Email verification integration

**Developed by:** Ndabi Joseph Muiruri  
**Institution:** Kabarak University  
**Course:** Bachelor of Business Information Technology

---

## 📝 Quick Commands Summary

```bash
# Installation
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run dashboard
streamlit run app.py

# Run CLI
python main.py

# Collect data
python -c "from src.data_collection import DataCollector; dc = DataCollector('AAPL'); dc.collect_all_data()"

# Preprocess
python -c "from src.preprocessing import DataPreprocessor; dp = DataPreprocessor(); dp.preprocess_and_split()"

# Train models
python src/train.py        # ANN
python src/train.py lstm   # LSTM

# Evaluate
python src/evaluate.py ann
python src/evaluate.py lstm
```

---

**That's it! You're ready to use the Stock Market Forecasting System.** 🎉

For more details, see:
- `README.md` - Project overview
- `AGENTS.md` - Developer commands
- `QUICK_START.txt` - One-page reference
