#!/usr/bin/env python3
"""
Test script for the gpt-3.5-cleaned.csv dataset integration
"""

import pandas as pd
import sys

def test_dataset_loading():
    """Test if the dataset can be loaded correctly"""
    try:
        print("📊 Testing dataset loading...")
        df = pd.read_csv('gpt-3.5-cleaned.csv')
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📈 Total rows: {len(df)}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Filter out empty annotated_texts
        df_clean = df[df['annotated_texts'].notna() & (df['annotated_texts'] != '')]
        print(f"🧹 Clean rows (with gestures): {len(df_clean)}")
        
        # Show unique gestures
        unique_gestures = df_clean['annotated_texts'].unique()
        print(f"🤟 Unique gestures: {len(unique_gestures)}")
        
        print("\n📝 Sample gestures:")
        for i, gesture in enumerate(unique_gestures[:10]):
            print(f"  {i+1}. {gesture}")
        
        if len(unique_gestures) > 10:
            print(f"  ... and {len(unique_gestures) - 10} more")
        
        return True
        
    except FileNotFoundError:
        print("❌ Error: gpt-3.5-cleaned.csv not found!")
        return False
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_gesture_analysis():
    """Test gesture analysis functionality"""
    try:
        print("\n🔍 Testing gesture analysis...")
        
        # Import the SilentEchoOllama class
        from silent_echo_ollama import SilentEchoOllama
        
        # Initialize the app
        app = SilentEchoOllama()
        
        print(f"✅ SilentEchoOllama initialized successfully!")
        print(f"📊 Dataset loaded: {not app.sign_dataset.empty}")
        
        if not app.sign_dataset.empty:
            gestures = app.get_sign_gestures()
            print(f"🤟 Available gestures: {len(gestures)}")
            
            # Test gesture matching
            test_gesture = "Index Finger"
            similar = app.find_similar_gesture(test_gesture)
            print(f"🎯 Test gesture '{test_gesture}' -> Similar: {similar}")
            
            # Test context retrieval
            if similar:
                context = app.get_gesture_context(similar)
                print(f"📊 Context: {context}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in gesture analysis: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Silent Echo Dataset Integration")
    print("=" * 50)
    
    # Test dataset loading
    dataset_ok = test_dataset_loading()
    
    if dataset_ok:
        # Test gesture analysis
        analysis_ok = test_gesture_analysis()
        
        if analysis_ok:
            print("\n🎉 All tests passed! Dataset integration is working correctly.")
        else:
            print("\n⚠️ Dataset loading OK, but gesture analysis failed.")
            sys.exit(1)
    else:
        print("\n❌ Dataset loading failed. Please check the file.")
        sys.exit(1) 