#!/usr/bin/env python3
"""
Silent Echo Demo
A demonstration script showcasing the key features of Silent Echo for hackathon presentation.
"""

import time
import sys
import os
from silent_echo_ollama import SilentEchoOllama

def print_banner():
    """Print the Silent Echo banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🤟 SILENT ECHO 🤟                        ║
    ║              AI Communication Assistant                      ║
    ║                for the Deaf Community                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def demo_speech_to_text():
    """Demonstrate speech-to-text functionality"""
    print("\n🎤 DEMO: Speech-to-Text Recognition")
    print("=" * 50)
    
    echo = SilentEchoOllama()
    
    print("This demo will show how Silent Echo can:")
    print("1. Listen to your speech")
    print("2. Convert it to text")
    print("3. Generate an AI response")
    print("4. Provide sign language translation")
    
    input("\nPress Enter to start the speech recognition demo...")
    
    print("\n🎙️ Listening for speech... (Speak now!)")
    audio = echo.listen_for_speech()
    
    if audio:
        text = echo.speech_to_text(audio)
        if text:
            print(f"✅ Speech recognized: '{text}'")
            
            print("\n🤖 Getting AI response...")
            response = echo.get_ollama_response(text)
            print(f"✅ AI Response: '{response}'")
            
            print("\n🤟 Sign Language Translation:")
            signs = echo.translate_to_sign_language(response)
            if signs:
                for sign in signs:
                    print(f"   • {sign}")
            else:
                print("   No common signs found in response")
        else:
            print("❌ Could not recognize speech. Please try again.")
    else:
        print("❌ No audio detected. Please try again.")

def demo_text_communication():
    """Demonstrate text-based communication"""
    print("\n⌨️ DEMO: Text Communication")
    print("=" * 50)
    
    echo = SilentEchoOllama()
    
    demo_messages = [
        "Hello, how are you today?",
        "Can you help me learn sign language?",
        "What time is it?",
        "Thank you for your help"
    ]
    
    print("This demo will show text-based communication with AI responses.")
    print("Sample messages:")
    for i, msg in enumerate(demo_messages, 1):
        print(f"{i}. {msg}")
    
    choice = input("\nSelect a message (1-4) or type your own: ")
    
    if choice.isdigit() and 1 <= int(choice) <= 4:
        message = demo_messages[int(choice) - 1]
    else:
        message = choice
    
    print(f"\n📝 User message: '{message}'")
    
    print("\n🤖 Getting AI response...")
    response = echo.get_ollama_response(message)
    print(f"✅ AI Response: '{response}'")
    
    print("\n🤟 Sign Language Translation:")
    signs = echo.translate_to_sign_language(response)
    if signs:
        for sign in signs:
            print(f"   • {sign}")
    else:
        print("   No common signs found in response")

def demo_sign_language_learning():
    """Demonstrate sign language learning features"""
    print("\n📚 DEMO: Sign Language Learning")
    print("=" * 50)
    
    common_signs = {
        "Hello": "👋 Wave your hand",
        "Thank You": "🤲 Flat hand from chin forward",
        "Please": "🤲 Flat hand in circular motion",
        "Yes": "👍 Thumbs up",
        "No": "👎 Thumbs down",
        "Help": "🤲 Both hands up, palms facing",
        "Good": "👍 Thumbs up",
        "Bad": "👎 Thumbs down",
        "Sorry": "🤲 Fist over heart",
        "Love": "🤟 Sign language 'I love you'"
    }
    
    print("Learn basic sign language gestures:")
    print()
    
    for sign, instruction in common_signs.items():
        print(f"🤟 {sign}: {instruction}")
        time.sleep(0.5)
    
    print("\n💡 Tips for learning sign language:")
    print("• Practice in front of a mirror")
    print("• Start with simple gestures")
    print("• Be patient with yourself")
    print("• Practice regularly")

def demo_impact_story():
    """Tell the impact story"""
    print("\n🌟 IMPACT STORY")
    print("=" * 50)
    
    story = """
    Meet Sarah, a deaf college student who struggled with communication 
    in her classes. Before Silent Echo, she had to rely on interpreters 
    or written notes, which often missed the context and nuance of 
    classroom discussions.
    
    With Silent Echo, Sarah can now:
    • 🎤 Understand spoken lectures in real-time
    • 🤖 Get contextual AI responses to her questions
    • 🤟 Learn sign language at her own pace
    • 📱 Communicate more effectively with hearing peers
    
    Sarah's story represents millions of deaf individuals worldwide 
    who face communication barriers every day. Silent Echo aims to 
    bridge these gaps and create a more inclusive world.
    """
    
    print(story)

def demo_technical_features():
    """Showcase technical features"""
    print("\n🛠️ TECHNICAL FEATURES")
    print("=" * 50)
    
    features = [
        ("IBM Granite 3.3 8B Model", "Advanced language model for contextual responses"),
        ("Real-time Speech Recognition", "Google Speech API integration"),
        ("Computer Vision", "OpenCV for hand gesture detection"),
        ("Text-to-Speech", "Google TTS for audio output"),
        ("Web Interface", "Streamlit for accessibility"),
        ("Local AI Processing", "Ollama for privacy and speed"),
        ("Multi-modal Input", "Speech, text, and visual input support"),
        ("Accessibility Design", "WCAG compliant interface")
    ]
    
    for feature, description in features:
        print(f"🔧 {feature}")
        print(f"   {description}")
        print()

def main():
    """Main demo function"""
    print_banner()
    
    print("Welcome to the Silent Echo Demo!")
    print("This demonstration showcases our AI-powered communication assistant for the deaf community.")
    
    demos = [
        ("🎤 Speech-to-Text Demo", demo_speech_to_text),
        ("⌨️ Text Communication Demo", demo_text_communication),
        ("📚 Sign Language Learning", demo_sign_language_learning),
        ("🌟 Impact Story", demo_impact_story),
        ("🛠️ Technical Features", demo_technical_features)
    ]
    
    while True:
        print("\n" + "=" * 60)
        print("DEMO MENU")
        print("=" * 60)
        
        for i, (name, _) in enumerate(demos, 1):
            print(f"{i}. {name}")
        print("0. Exit Demo")
        
        choice = input("\nSelect a demo (0-5): ")
        
        if choice == "0":
            print("\n👋 Thank you for experiencing Silent Echo!")
            print("🤟 Breaking barriers, one conversation at a time.")
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(demos):
            try:
                demos[int(choice) - 1][1]()
                input("\nPress Enter to continue...")
            except KeyboardInterrupt:
                print("\n\n👋 Demo interrupted. Thank you!")
                break
            except Exception as e:
                print(f"\n❌ Demo error: {e}")
                input("Press Enter to continue...")
        else:
            print("❌ Invalid choice. Please select 0-5.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo stopped by user. Thank you!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        sys.exit(1) 