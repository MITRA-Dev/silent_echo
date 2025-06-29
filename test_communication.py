#!/usr/bin/env python3
"""
Test script for the communication response system using actual dataset
"""

from silent_echo_ollama import SilentEchoOllama

def test_communication_responses():
    """Test the communication response system using actual dataset"""
    print("🚀 Testing Communication Response System with Dataset")
    print("=" * 60)
    
    # Initialize the app
    app = SilentEchoOllama()
    
    print("📊 Dataset Status:")
    print(f"Dataset loaded: {not app.sign_dataset.empty}")
    if not app.sign_dataset.empty:
        print(f"Available gestures: {len(app.get_sign_gestures())}")
    
    # Find actual communication gestures in dataset
    available_gestures = app.get_sign_gestures()
    communication_words = ["hello", "please", "help", "thank", "goodbye", "stop", "yes", "no", "more", "okay"]
    
    print("\n📝 Actual Communication Gestures from Dataset:")
    found_gestures = []
    
    for word in communication_words:
        matching_gestures = [g for g in available_gestures if word.lower() in g.lower()]
        if matching_gestures:
            print(f"✅ '{word}' found in: {matching_gestures[0]}")
            found_gestures.append(matching_gestures[0])
        else:
            print(f"❌ '{word}' not found in dataset")
    
    print(f"\n🔍 Found {len(found_gestures)} communication gestures in dataset")
    
    # Test response generation with actual gestures
    if found_gestures:
        print("\n🧪 Testing Response Generation with Real Gestures:")
        for gesture in found_gestures[:3]:  # Test first 3
            response = app.get_communication_response(gesture)
            if response:
                print(f"✅ '{gesture}' -> '{response}'")
            else:
                print(f"❌ '{gesture}' -> No communication word")
    
    print("\n🎉 Dataset-based communication system test completed!")

if __name__ == "__main__":
    test_communication_responses() 