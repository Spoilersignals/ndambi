# 🔐 User Authentication System - Quick Start

## ✅ What's Been Added

Your Stock Market Forecasting System now has:
- **User Registration** with email verification
- **Secure Login/Logout** system
- **Email Verification Codes** (6-digit, expires in 10 minutes)
- **Protected Dashboard** (login required)

## 🚀 Quick Setup (3 Steps)

### Step 1: Create `.env` File

Copy the example file:
```bash
copy .env.example .env
```

Edit `.env` and add your Gmail credentials:
```
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=xxxx-xxxx-xxxx-xxxx
```

**Get Gmail App Password:**
1. Go to https://myaccount.google.com/apppasswords
2. Generate password for "Mail" → "Windows Computer"
3. Copy the 16-digit code

### Step 2: Test the System

```bash
venv\Scripts\python.exe -c "from auth import AuthManager; print('✅ Auth system ready!')"
```

### Step 3: Run the App

```bash
venv\Scripts\streamlit run app.py
```

## 📱 How to Use

### First Time User

1. **Open the app** → You'll see the login page
2. **Click "Create Account"**
3. **Fill in**:
   - Full Name: `Joseph Muiruri`
   - Email: `your-email@gmail.com`
   - Password: minimum 6 characters
4. **Check your email** for the 6-digit code
5. **Enter the code** → Account verified!
6. **Login** with your email and password

### Returning User

1. **Enter email and password**
2. **Click Login**
3. **Access the dashboard**

## 🎯 System Flow

```
┌─────────────┐
│   Register  │ → Email Sent
└──────┬──────┘
       ↓
┌─────────────┐
│   Verify    │ → Enter 6-digit code
└──────┬──────┘
       ↓
┌─────────────┐
│    Login    │ → Access Dashboard
└──────┬──────┘
       ↓
┌─────────────┐
│  Dashboard  │ → Data, Models, Predictions
└─────────────┘
```

## 📧 Email Not Working?

**Don't worry!** If email fails, the verification code will be shown on screen:

```
✅ Registration successful! Your verification code is: 123456
```

Just copy the code and paste it in the verification page.

## 🗄️ Database

User data is stored in `users.db` (SQLite):
- Passwords are hashed (SHA-256)
- Sessions are tokenized
- Verification codes expire

**Reset everything:**
```bash
del users.db
```

## 🔒 Security Features

- ✅ Password hashing (not stored as plain text)
- ✅ Secure session tokens
- ✅ Timed verification codes (10 min expiry)
- ✅ Email validation
- ✅ Protected routes

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Email not sending | Use Gmail App Password, not regular password |
| "Invalid code" | Code expires in 10 minutes - register again |
| "Email already registered" | Use Login instead of Register |
| Forgot password | Delete `users.db` and re-register (dev mode) |

## 📄 Files Added

```
NDAMBI/
├── auth.py                      # Authentication logic
├── users.db                     # User database (auto-created)
├── .env                         # Email config (YOU CREATE THIS)
├── .env.example                 # Template
├── EMAIL_SETUP.md               # Detailed email guide
└── app.py                       # Updated with auth pages
```

## ⚡ Commands Reference

```bash
# Run the app
venv\Scripts\streamlit run app.py

# Test authentication
venv\Scripts\python.exe -c "from auth import AuthManager; print('Ready')"

# Reset database
del users.db

# Create .env file
copy .env.example .env
notepad .env
```

## 🎓 For Your Project

This authentication system adds:
- **Security layer** for your academic project
- **User management** for multi-user access
- **Professional feature** for demonstration
- **Email integration** showing real-world functionality

You can now demonstrate:
1. User registration with email verification
2. Secure authentication flow
3. Session management
4. Database integration (SQLite)

---

**Need help?** Check [EMAIL_SETUP.md](EMAIL_SETUP.md) for detailed email configuration guide.
