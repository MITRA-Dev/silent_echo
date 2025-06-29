#!/usr/bin/env python3
"""
Simple webcam test script to diagnose camera issues
"""

import cv2
import time
import sys

def test_webcam_simple():
    """Simple webcam test with minimal configuration"""
    print("📹 Simple Webcam Test")
    print("=" * 30)
    
    # Try Media Foundation first (most stable on Windows)
    print("🔍 Testing Media Foundation backend...")
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if cap.isOpened():
            print("   ✅ Camera opened with Media Foundation")
            
            # Test frame reading
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print("   ✅ Frame reading successful")
                print(f"   📐 Frame size: {frame.shape}")
                
                # Try to display frame
                cv2.imshow("Test Camera", frame)
                cv2.waitKey(2000)  # Show for 2 seconds
                cv2.destroyAllWindows()
                
                cap.release()
                print("   ✅ Webcam test completed successfully")
                return True
            else:
                print("   ❌ Frame reading failed")
                cap.release()
        else:
            print("   ❌ Could not open camera with Media Foundation")
    except Exception as e:
        print(f"   ❌ Media Foundation error: {e}")
    
    # Try DirectShow as fallback
    print("\n🔍 Testing DirectShow backend...")
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            print("   ✅ Camera opened with DirectShow")
            
            # Test frame reading
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print("   ✅ Frame reading successful")
                print(f"   📐 Frame size: {frame.shape}")
                
                # Try to display frame
                cv2.imshow("Test Camera", frame)
                cv2.waitKey(2000)  # Show for 2 seconds
                cv2.destroyAllWindows()
                
                cap.release()
                print("   ✅ Webcam test completed successfully")
                return True
            else:
                print("   ❌ Frame reading failed")
                cap.release()
        else:
            print("   ❌ Could not open camera with DirectShow")
    except Exception as e:
        print(f"   ❌ DirectShow error: {e}")
    
    # Try default backend
    print("\n🔍 Testing default backend...")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("   ✅ Camera opened with default backend")
            
            # Test frame reading
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print("   ✅ Frame reading successful")
                print(f"   📐 Frame size: {frame.shape}")
                
                # Try to display frame
                cv2.imshow("Test Camera", frame)
                cv2.waitKey(2000)  # Show for 2 seconds
                cv2.destroyAllWindows()
                
                cap.release()
                print("   ✅ Webcam test completed successfully")
                return True
            else:
                print("   ❌ Frame reading failed")
                cap.release()
        else:
            print("   ❌ Could not open camera with default backend")
    except Exception as e:
        print(f"   ❌ Default backend error: {e}")
    
    print("\n❌ All webcam tests failed")
    return False

if __name__ == "__main__":
    try:
        success = test_webcam_simple()
        if success:
            print("\n✅ Webcam is working correctly!")
        else:
            print("\n❌ Webcam has issues. Check:")
            print("   • Camera permissions")
            print("   • Camera is not in use by another application")
            print("   • Camera drivers are installed")
            print("   • Try restarting your computer")
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
    finally:
        cv2.destroyAllWindows() 