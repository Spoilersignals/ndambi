import sqlite3
import hashlib
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import os

class AuthManager:
    def __init__(self, db_path='users.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for users"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                is_verified INTEGER DEFAULT 0,
                verification_code TEXT,
                code_expires_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def generate_verification_code(self):
        """Generate 6-digit verification code"""
        return ''.join([str(secrets.randbelow(10)) for _ in range(6)])
    
    def send_verification_email(self, email, code, full_name):
        """Send verification code via email"""
        # Email configuration
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        sender_email = os.getenv('SENDER_EMAIL', 'your-email@gmail.com')
        sender_password = os.getenv('SENDER_PASSWORD', 'your-app-password')
        
        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = "Stock Market System - Email Verification"
        message["From"] = sender_email
        message["To"] = email
        
        # HTML email body
        html = f"""
        <html>
          <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
              <h2 style="color: #1f77b4;">Welcome to Stock Market Forecasting System!</h2>
              <p>Hello <strong>{full_name}</strong>,</p>
              <p>Thank you for registering. Please use the verification code below to complete your registration:</p>
              <div style="background-color: #f4f4f4; padding: 15px; text-align: center; font-size: 24px; font-weight: bold; letter-spacing: 5px; margin: 20px 0; border-radius: 5px;">
                {code}
              </div>
              <p style="color: #666; font-size: 14px;">This code will expire in 10 minutes.</p>
              <p>If you didn't request this, please ignore this email.</p>
              <hr style="margin: 30px 0; border: none; border-top: 1px solid #ddd;">
              <p style="color: #999; font-size: 12px;">Stock Market Forecasting System Using ANNs</p>
            </div>
          </body>
        </html>
        """
        
        part = MIMEText(html, "html")
        message.attach(part)
        
        try:
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, email, message.as_string())
            return True
        except Exception as e:
            print(f"Email error: {str(e)}")
            return False
    
    def register_user(self, email, password, full_name):
        """Register new user and send verification email"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if email already exists
            cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
            if cursor.fetchone():
                return False, "Email already registered"
            
            # Generate verification code
            code = self.generate_verification_code()
            expires_at = (datetime.now() + timedelta(minutes=10)).isoformat()
            
            # Hash password
            password_hash = self.hash_password(password)
            
            # Insert user
            cursor.execute('''
                INSERT INTO users (email, password_hash, full_name, verification_code, code_expires_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (email, password_hash, full_name, code, expires_at))
            
            conn.commit()
            
            # Send verification email
            email_sent = self.send_verification_email(email, code, full_name)
            
            if email_sent:
                return True, "Registration successful! Check your email for verification code."
            else:
                return True, f"Registration successful! Your verification code is: {code}"
            
        except Exception as e:
            return False, f"Registration error: {str(e)}"
        finally:
            conn.close()
    
    def verify_email(self, email, code):
        """Verify user email with code"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT verification_code, code_expires_at 
                FROM users 
                WHERE email = ? AND is_verified = 0
            ''', (email,))
            
            result = cursor.fetchone()
            
            if not result:
                return False, "User not found or already verified"
            
            stored_code, expires_at = result
            
            # Check if code expired
            if datetime.now() > datetime.fromisoformat(expires_at):
                return False, "Verification code expired. Please request a new one."
            
            # Check if code matches
            if stored_code != code:
                return False, "Invalid verification code"
            
            # Mark as verified
            cursor.execute('''
                UPDATE users 
                SET is_verified = 1, verification_code = NULL, code_expires_at = NULL
                WHERE email = ?
            ''', (email,))
            
            conn.commit()
            return True, "Email verified successfully! You can now login."
            
        except Exception as e:
            return False, f"Verification error: {str(e)}"
        finally:
            conn.close()
    
    def login(self, email, password):
        """Login user and return session token"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            password_hash = self.hash_password(password)
            
            cursor.execute('''
                SELECT id, full_name, is_verified 
                FROM users 
                WHERE email = ? AND password_hash = ?
            ''', (email, password_hash))
            
            result = cursor.fetchone()
            
            if not result:
                return False, "Invalid email or password", None
            
            user_id, full_name, is_verified = result
            
            if not is_verified:
                return False, "Please verify your email first", None
            
            # Create session token
            session_token = secrets.token_urlsafe(32)
            
            cursor.execute('''
                INSERT INTO sessions (user_id, session_token)
                VALUES (?, ?)
            ''', (user_id, session_token))
            
            conn.commit()
            
            return True, "Login successful!", {
                'user_id': user_id,
                'email': email,
                'full_name': full_name,
                'session_token': session_token
            }
            
        except Exception as e:
            return False, f"Login error: {str(e)}", None
        finally:
            conn.close()
    
    def logout(self, session_token):
        """Logout user by removing session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM sessions WHERE session_token = ?', (session_token,))
            conn.commit()
            return True
        except:
            return False
        finally:
            conn.close()
    
    def validate_session(self, session_token):
        """Check if session is valid"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT u.id, u.email, u.full_name
                FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.session_token = ?
            ''', (session_token,))
            
            result = cursor.fetchone()
            
            if result:
                return True, {
                    'user_id': result[0],
                    'email': result[1],
                    'full_name': result[2]
                }
            return False, None
            
        except:
            return False, None
        finally:
            conn.close()
