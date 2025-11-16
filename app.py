import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
from dotenv import load_dotenv
from src.data_collection import DataCollector
from src.preprocessing import DataPreprocessor
from src.model import StockPredictionANN, LSTMStockPredictor
from config import Config
from auth import AuthManager

load_dotenv()

st.set_page_config(
    page_title="Stock Market Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional styling
st.markdown("""
<style>
    /* Main background - clean gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 600;
        color: #0ea5e9;
    }
    
    /* Headers - compact */
    h1 {
        color: #1e293b !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h3 {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 1.25rem !important;
    }
    
    /* Buttons - professional look */
    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #0284c7 0%, #0891b2 100%);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.4);
    }
    
    /* Input fields - cleaner */
    .stTextInput>div>div>input, .stDateInput>div>div>input, .stNumberInput>div>div>input {
        border-radius: 6px;
        border: 1px solid #cbd5e1;
        padding: 8px 12px;
        font-size: 0.95rem;
    }
    
    .stTextInput>div>div>input:focus, .stDateInput>div>div>input:focus {
        border-color: #0ea5e9;
        box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.1);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #10b981;
        border-radius: 6px;
        padding: 12px;
        color: white;
    }
    
    .stError {
        background-color: #ef4444;
        border-radius: 6px;
        padding: 12px;
        color: white;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
        margin: 1.5rem 0;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 6px;
        overflow: hidden;
        font-size: 0.9rem;
    }
    
    /* Reduce spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'auth_step' not in st.session_state:
    st.session_state.auth_step = 'login'
if 'pending_email' not in st.session_state:
    st.session_state.pending_email = None

auth_manager = AuthManager()

@st.cache_data
def load_raw_data(symbol):
    filepath = os.path.join(Config.RAW_DATA_DIR, f'{symbol}_raw_data.csv')
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

@st.cache_data
def load_training_history(model_type):
    filepath = os.path.join(Config.MODELS_DIR, f'{model_type}_training_history.json')
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def login_page():
    """Login page"""
    st.markdown('<div style="text-align: center; margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.title("🔐 Stock Market Forecasting")
    st.markdown('<p style="color: #475569; font-size: 1rem;">Sign in to your account</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div style="background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
        
        email = st.text_input("📧 Email Address", key="login_email", placeholder="your.email@example.com")
        password = st.text_input("🔒 Password", type="password", key="login_password", placeholder="Enter your password")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🚀 Sign In", type="primary", use_container_width=True):
                if email and password:
                    success, message, user_data = auth_manager.login(email, password)
                    
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user_data = user_data
                        st.success(message)
                        st.experimental_rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter email and password")
        
        with col_btn2:
            if st.button("📝 Create Account", use_container_width=True):
                st.session_state.auth_step = 'register'
                st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def register_page():
    """Registration page"""
    st.markdown('<div style="text-align: center; margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.title("📝 Create Account")
    st.markdown('<p style="color: #475569; font-size: 1rem;">Join to start forecasting</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div style="background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
        
        full_name = st.text_input("👤 Full Name", key="reg_name", placeholder="Joseph Muiruri")
        email = st.text_input("📧 Email Address", key="reg_email", placeholder="your-email@gmail.com")
        password = st.text_input("🔒 Password", type="password", key="reg_password", placeholder="Minimum 6 characters")
        password_confirm = st.text_input("🔒 Confirm Password", type="password", key="reg_password_confirm", placeholder="Re-enter password")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("✅ Register", type="primary", use_container_width=True):
                # Validate fields
                if not full_name or not full_name.strip():
                    st.error("❌ Please enter your full name")
                elif not email or not email.strip():
                    st.error("❌ Please enter your email")
                elif not password:
                    st.error("❌ Please enter a password")
                elif not password_confirm:
                    st.error("❌ Please confirm your password")
                elif password != password_confirm:
                    st.error("❌ Passwords do not match!")
                elif len(password) < 6:
                    st.error("❌ Password must be at least 6 characters")
                else:
                    # All validation passed
                    success, message = auth_manager.register_user(
                        email.strip(), 
                        password, 
                        full_name.strip()
                    )
                    
                    if success:
                        st.success(message)
                        st.session_state.pending_email = email.strip()
                        st.session_state.auth_step = 'verify'
                        st.experimental_rerun()
                    else:
                        st.error(message)
        
        with col_btn2:
            if st.button("🔙 Back to Login", use_container_width=True):
                st.session_state.auth_step = 'login'
                st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def verify_email_page():
    """Email verification page"""
    st.markdown('<div style="text-align: center; margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.title("✉️ Verify Your Email")
    st.markdown('<p style="color: #475569; font-size: 1rem;">Enter the code sent to your inbox</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div style="background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
        
        st.info(f"📧 A verification code has been sent to **{st.session_state.pending_email}**")
        
        st.markdown("<br>", unsafe_allow_html=True)
        code = st.text_input("🔢 Enter 6-Digit Verification Code", max_chars=6, key="verify_code", placeholder="000000")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("✅ Verify", type="primary", use_container_width=True):
                if code and len(code) == 6:
                    success, message = auth_manager.verify_email(st.session_state.pending_email, code)
                    
                    if success:
                        st.success(message)
                        st.balloons()
                        st.session_state.auth_step = 'login'
                        st.session_state.pending_email = None
                        st.experimental_rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter a 6-digit code")
        
        with col_btn2:
            if st.button("🔙 Back to Login", use_container_width=True):
                st.session_state.auth_step = 'login'
                st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Check authentication
    if not st.session_state.authenticated:
        if st.session_state.auth_step == 'login':
            login_page()
        elif st.session_state.auth_step == 'register':
            register_page()
        elif st.session_state.auth_step == 'verify':
            verify_email_page()
        return
    
    # Authenticated user interface
    st.markdown('<div style="margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.title("📈 Stock Market Forecasting Dashboard")
    st.markdown(f'<p style="color: #64748b; font-size: 0.95rem;">Welcome, <strong>{st.session_state.user_data["full_name"]}</strong></p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.sidebar.header("⚙️ Configuration")
    
    # Logout button
    if st.sidebar.button("🚪 Logout"):
        auth_manager.logout(st.session_state.user_data.get('session_token'))
        st.session_state.authenticated = False
        st.session_state.user_data = None
        st.experimental_rerun()
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Select Page",
        ["📊 Data Collection", "🔧 Model Training", "📈 Predictions", "📋 Reports"]
    )
    
    if page == "📊 Data Collection":
        data_collection_page()
    elif page == "🔧 Model Training":
        model_training_page()
    elif page == "📈 Predictions":
        predictions_page()
    elif page == "📋 Reports":
        reports_page()

def data_collection_page():
    st.markdown('<div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.header("📊 Data Collection")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08);">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL")
        start_date = st.date_input("Start Date", value=pd.to_datetime("2019-01-01"))
    
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("today"), max_value=pd.to_datetime("2025-12-31"))
    
    if st.button("🚀 Fetch Data", type="primary"):
        with st.spinner(f"Fetching data for {symbol}..."):
            try:
                collector = DataCollector(symbol)
                data = collector.collect_all_data(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                st.success(f"✅ Successfully collected {len(data)} records!")
                
                st.subheader("📋 Data Preview")
                st.dataframe(data.head(20), use_container_width=True)
                
                st.subheader("📊 Data Statistics")
                st.dataframe(data.describe(), use_container_width=True)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.update_layout(
                    title=f"{symbol} Stock Price History",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode='x unified',
                    template='plotly_white',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12, color='#1e3a8a')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    
    data = load_raw_data(Config.DEFAULT_STOCK_SYMBOL)
    if data is not None:
        st.subheader(f"📁 Existing Data: {Config.DEFAULT_STOCK_SYMBOL}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", len(data))
        col2.metric("Latest Price", f"${data['Close'].iloc[-1]:.2f}")
        col3.metric("Avg Price", f"${data['Close'].mean():.2f}")
        col4.metric("Volatility", f"{data['Close'].std():.2f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def model_training_page():
    st.markdown('<div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.header("🔧 Model Training")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08);">', unsafe_allow_html=True)
    model_type = st.selectbox("Select Model Type", ["ANN (Feedforward)", "LSTM (Recurrent)"])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.number_input("Epochs", min_value=10, max_value=500, value=100)
    with col2:
        batch_size = st.number_input("Batch Size", min_value=8, max_value=128, value=32)
    with col3:
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
    
    st.markdown("---")
    
    if st.button("🚀 Start Training", type="primary"):
        st.warning("⚠️ Training requires preprocessed data. Make sure you've collected and preprocessed data first.")
        st.info("💡 Use the command line to train models: `python src/train.py` or `python src/train.py lstm`")
    
    st.markdown("---")
    
    st.subheader("📊 Training History")
    
    model_key = 'ann' if 'ANN' in model_type else 'lstm'
    history = load_training_history(model_key)
    
    if history:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history['loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='#ff7f0e', width=2)
            ))
            fig.add_trace(go.Scatter(
                y=history['val_loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='#2ca02c', width=2)
            ))
            
            fig.update_layout(
                title="Model Loss",
                xaxis_title="Epoch",
                yaxis_title="Loss (MSE)",
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history['mae'],
                mode='lines',
                name='Training MAE',
                line=dict(color='#d62728', width=2)
            ))
            fig.add_trace(go.Scatter(
                y=history['val_mae'],
                mode='lines',
                name='Validation MAE',
                line=dict(color='#9467bd', width=2)
            ))
            
            fig.update_layout(
                title="Model MAE",
                xaxis_title="Epoch",
                yaxis_title="MAE",
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        final_metrics = {
            "Final Training Loss": history['loss'][-1],
            "Final Validation Loss": history['val_loss'][-1],
            "Final Training MAE": history['mae'][-1],
            "Final Validation MAE": history['val_mae'][-1]
        }
        
        st.subheader("📋 Final Metrics")
        cols = st.columns(4)
        for i, (metric, value) in enumerate(final_metrics.items()):
            cols[i].metric(metric, f"{value:.6f}")
    else:
        st.info("ℹ️ No training history found. Train a model first.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def predictions_page():
    st.markdown('<div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.header("📈 Predictions")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08);">', unsafe_allow_html=True)
    model_type = st.selectbox("Select Model", ["ANN", "LSTM"])
    
    model_file = os.path.join(
        Config.MODELS_DIR, 
        f'{"stock_prediction_model.h5" if model_type == "ANN" else "lstm_stock_model.h5"}'
    )
    
    if os.path.exists(model_file):
        st.success(f"✅ {model_type} model found!")
        
        if st.button("🔮 Generate Predictions"):
            with st.spinner("Making predictions..."):
                try:
                    X_test = np.load(os.path.join(Config.PROCESSED_DATA_DIR, 'X_test.npy'))
                    y_test = np.load(os.path.join(Config.PROCESSED_DATA_DIR, 'y_test.npy'))
                    
                    input_shape = (X_test.shape[1], X_test.shape[2])
                    
                    if model_type == "ANN":
                        model = StockPredictionANN(input_shape=input_shape)
                    else:
                        model = LSTMStockPredictor(input_shape=input_shape)
                    
                    model.load_model()
                    
                    predictions = model.predict(X_test).flatten()
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        y=y_test,
                        mode='lines',
                        name='Actual Prices',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        y=predictions,
                        mode='lines',
                        name='Predicted Prices',
                        line=dict(color='#ff7f0e', width=2, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f"Stock Price Prediction - {model_type} Model",
                        xaxis_title="Time Steps",
                        yaxis_title="Normalized Price",
                        hovermode='x unified',
                        template='plotly_white',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    mse = mean_squared_error(y_test, predictions)
                    mae = mean_absolute_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)
                    rmse = np.sqrt(mse)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("MSE", f"{mse:.6f}")
                    col2.metric("RMSE", f"{rmse:.6f}")
                    col3.metric("MAE", f"{mae:.6f}")
                    col4.metric("R² Score", f"{r2:.4f}")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    else:
        st.warning(f"⚠️ {model_type} model not found. Train the model first.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def reports_page():
    st.markdown('<div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.header("📋 Reports")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08);">', unsafe_allow_html=True)
    reports_dir = Config.REPORTS_DIR
    
    if os.path.exists(reports_dir):
        report_files = [f for f in os.listdir(reports_dir) if f.endswith('.txt')]
        image_files = [f for f in os.listdir(reports_dir) if f.endswith('.png')]
        
        if report_files:
            selected_report = st.selectbox("Select Report", report_files)
            
            if selected_report:
                with open(os.path.join(reports_dir, selected_report), 'r') as f:
                    report_content = f.read()
                
                st.text(report_content)
        
        st.markdown("---")
        
        if image_files:
            st.subheader("📊 Visualizations")
            
            selected_images = st.multiselect(
                "Select Visualizations to Display",
                image_files,
                default=image_files[:2] if len(image_files) >= 2 else image_files
            )
            
            for img_file in selected_images:
                st.image(
                    os.path.join(reports_dir, img_file),
                    caption=img_file,
                    use_container_width=True
                )
        else:
            st.info("ℹ️ No visualization images found. Run model evaluation first.")
    else:
        st.warning("⚠️ Reports directory not found.")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
