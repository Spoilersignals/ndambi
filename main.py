import os
import sys
from src.data_collection import DataCollector
from src.preprocessing import DataPreprocessor
from src.train import train_ann_model, train_lstm_model
from src.evaluate import ModelEvaluator
from config import Config

def print_banner():
    banner = """
    ================================================================
                                                               
            STOCK MARKET FORECASTING SYSTEM USING ANNs             
                                                               
    ================================================================
    """
    print(banner)

def print_menu():
    menu = """
    ================================================================
      MAIN MENU                                                    
    ================================================================
      1. Collect Stock Data                                        
      2. Preprocess Data                                           
      3. Train ANN Model                                           
      4. Train LSTM Model                                          
      5. Evaluate ANN Model                                        
      6. Evaluate LSTM Model                                       
      7. Run Complete Pipeline (Collect -> Preprocess -> Train)      
      8. Launch Dashboard (Streamlit)                              
      0. Exit                                                      
    ================================================================
    """
    print(menu)

def collect_data():
    print("\n" + "="*60)
    print("DATA COLLECTION")
    print("="*60)
    
    symbol = input("Enter stock symbol (default: AAPL): ").strip() or "AAPL"
    
    collector = DataCollector(symbol)
    
    try:
        data = collector.collect_all_data()
        print(f"\n✓ Data collection completed successfully!")
        print(f"  Records collected: {len(data)}")
        return True
    except Exception as e:
        print(f"\n✗ Error during data collection: {e}")
        return False

def preprocess_data():
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    symbol = input("Enter stock symbol (default: AAPL): ").strip() or "AAPL"
    input_file = os.path.join(Config.RAW_DATA_DIR, f'{symbol}_raw_data.csv')
    
    if not os.path.exists(input_file):
        print(f"\n✗ Error: Data file not found at {input_file}")
        print("  Please run data collection first (Option 1)")
        return False
    
    preprocessor = DataPreprocessor()
    
    try:
        result = preprocessor.preprocess_pipeline(input_file)
        print(f"\n✓ Data preprocessing completed successfully!")
        print(f"  Training samples: {len(result['X_train'])}")
        print(f"  Validation samples: {len(result['X_val'])}")
        print(f"  Test samples: {len(result['X_test'])}")
        return True
    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}")
        return False

def train_ann():
    print("\n" + "="*60)
    print("TRAINING ANN MODEL")
    print("="*60)
    
    try:
        train_ann_model()
        print("\n✓ ANN model training completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        return False

def train_lstm():
    print("\n" + "="*60)
    print("TRAINING LSTM MODEL")
    print("="*60)
    
    try:
        train_lstm_model()
        print("\n✓ LSTM model training completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        return False

def evaluate_ann():
    print("\n" + "="*60)
    print("EVALUATING ANN MODEL")
    print("="*60)
    
    try:
        evaluator = ModelEvaluator(model_type='ann')
        evaluator.evaluate()
        print("\n✓ ANN model evaluation completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        return False

def evaluate_lstm():
    print("\n" + "="*60)
    print("EVALUATING LSTM MODEL")
    print("="*60)
    
    try:
        evaluator = ModelEvaluator(model_type='lstm')
        evaluator.evaluate()
        print("\n✓ LSTM model evaluation completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        return False

def run_complete_pipeline():
    print("\n" + "="*60)
    print("RUNNING COMPLETE PIPELINE")
    print("="*60)
    
    if not collect_data():
        return
    
    if not preprocess_data():
        return
    
    choice = input("\nTrain which model? (1=ANN, 2=LSTM, 3=Both): ").strip()
    
    if choice == "1":
        train_ann()
        evaluate_ann()
    elif choice == "2":
        train_lstm()
        evaluate_lstm()
    elif choice == "3":
        train_ann()
        evaluate_ann()
        train_lstm()
        evaluate_lstm()
    
    print("\n✓ Complete pipeline executed successfully!")

def launch_dashboard():
    print("\n" + "="*60)
    print("LAUNCHING STREAMLIT DASHBOARD")
    print("="*60)
    print("\nStarting Streamlit server...")
    print("Dashboard will open in your browser at http://localhost:8501")
    print("\nPress Ctrl+C to stop the server.\n")
    
    os.system("streamlit run app.py")

def main():
    print_banner()
    
    while True:
        print_menu()
        
        choice = input("Enter your choice: ").strip()
        
        if choice == "1":
            collect_data()
        elif choice == "2":
            preprocess_data()
        elif choice == "3":
            train_ann()
        elif choice == "4":
            train_lstm()
        elif choice == "5":
            evaluate_ann()
        elif choice == "6":
            evaluate_lstm()
        elif choice == "7":
            run_complete_pipeline()
        elif choice == "8":
            launch_dashboard()
        elif choice == "0":
            print("\nThank you for using the Stock Market Forecasting System!")
            print("Goodbye! 👋\n")
            sys.exit(0)
        else:
            print("\n✗ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")
        print("\n" * 2)

if __name__ == "__main__":
    main()
