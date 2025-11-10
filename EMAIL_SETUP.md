# Email Verification Setup Guide

## Overview
The Stock Market Forecasting System now includes user authentication with email verification. Users must create an account and verify their email before accessing the system.

## Features
- ✅ User registration with email verification
- ✅ Secure password hashing (SHA-256)
- ✅ 6-digit verification codes (expires in 10 minutes)
- ✅ Session-based authentication
- ✅ Email notifications via SMTP

## Setup Instructions

### 1. Install Required Package
```bash
pip install python-dotenv
```

### 2. Configure Email Settings

#### Option A: Using Gmail (Recommended)

1. **Enable 2-Factor Authentication** on your Google Account:
   - Go to https://myaccount.google.com/security
   - Enable "2-Step Verification"

2. **Generate App Password**:
   - Go to https://myaccount.google.com/apppasswords
   - Select app: "Mail"
   - Select device: "Windows Computer" (or your device)
   - Click "Generate"
   - Copy the 16-digit password

3. **Create `.env` file** in the project root:
```bash
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-16-digit-app-password
```

#### Option B: Using Other Email Providers

**Outlook/Hotmail:**
```bash
SMTP_SERVER=smtp-mail.outlook.com
SMTP_PORT=587
SENDER_EMAIL=your-email@outlook.com
SENDER_PASSWORD=your-password
```

**Yahoo:**
```bash
SMTP_SERVER=smtp.mail.yahoo.com
SMTP_PORT=587
SENDER_EMAIL=your-email@yahoo.com
SENDER_PASSWORD=your-app-password
```

### 3. Test Email Configuration

Run this command to test email sending:
```bash
venv\Scripts\python.exe -c "from auth import AuthManager; am = AuthManager(); print('Email sent:', am.send_verification_email('test@example.com', '123456', 'Test User'))"
```

## Usage Flow

### 1. Registration
1. Run the Streamlit app: `streamlit run app.py`
2. Click **"Create Account"**
3. Enter:
   - Full Name
   - Email address
   - Password (minimum 6 characters)
   - Confirm password
4. Click **"Register"**
5. Check your email for the 6-digit verification code

### 2. Email Verification
1. Enter the 6-digit code from your email
2. Click **"Verify"**
3. Code expires in 10 minutes

### 3. Login
1. Enter your email and password
2. Click **"Login"**
3. Access the full system features

### 4. Logout
- Click **"Logout"** button in the sidebar

## Troubleshooting

### Email Not Sending
**Problem:** "Email error: (535, b'5.7.8 Username and Password not accepted')"

**Solutions:**
1. Make sure 2FA is enabled on Gmail
2. Use App Password, not regular password
3. Check that email and password in `.env` are correct
4. Enable "Less secure app access" (not recommended)

### Verification Code Not Received
1. Check spam/junk folder
2. Verify email address is correct
3. Try resending by registering again
4. Check console output for the code (fallback mode)

### "User not found or already verified"
- Email is already registered
- Try logging in instead
- Or use a different email

### Code Expired
- Verification codes expire after 10 minutes
- Register again to get a new code

## Development Mode (No Email)

If you can't configure email, the system will show the verification code in the success message:

```python
# In auth.py, line 94-96
if email_sent:
    return True, "Registration successful! Check your email for verification code."
else:
    return True, f"Registration successful! Your verification code is: {code}"
```

The code will be displayed on screen instead of being emailed.

## Security Notes

- ✅ Passwords are hashed using SHA-256
- ✅ Session tokens are cryptographically secure
- ✅ Verification codes expire automatically
- ✅ Email credentials are stored in `.env` (not committed to git)
- ⚠️ Never share your `.env` file
- ⚠️ Use App Passwords, not real passwords

## Database

User data is stored in `users.db` (SQLite):
- **users** table: email, password_hash, full_name, verification status
- **sessions** table: active login sessions

To reset the database:
```bash
del users.db
```

## Commands Summary

```bash
# Install dependencies
pip install python-dotenv

# Create .env file
copy .env.example .env
# Edit .env with your email settings

# Run the app
streamlit run app.py

# Test email (optional)
venv\Scripts\python.exe -c "from auth import AuthManager; am = AuthManager(); print('Test:', am.send_verification_email('test@email.com', '123456', 'Test'))"
```

## Support

If you encounter issues:
1. Check the `.env` file configuration
2. Verify Gmail App Password is correct
3. Check firewall/antivirus settings
4. Try using a different email provider
5. Enable development mode (code shown on screen)
