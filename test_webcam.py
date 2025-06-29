#!/usr/bin/env python3
"""
Simple webcam test script to diagnose camera issues
"""

import cv2
import time
import sys

def test_webcam():
    """Test webcam with different backends and indices"""
    print("📹 Webcam Test Script")
    print("=" * 30)
    
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Auto-detect")
    ]
    
    working_configs = []
    
    for backend_code, backend_name in backends:
        print(f"\n🔍 Testing {backend_name} backend...")
        for index in range(4):  # Test indices 0-3
            try:
                print(f"   Testing camera index {index}...")
                cap = cv2.VideoCapture(index, backend_code)
                
                if cap.isOpened():
                    # Test frame reading
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Get camera properties
                        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        print(f"   ✅ Camera {index}: {int(width)}x{int(height)} @ {fps:.1f}fps")
                        working_configs.append({
                            'index': index,
                            'backend': backend_name,
                            'backend_code': backend_code,
                            'resolution': f"{int(width)}x{int(height)}",
                            'fps': fps
                        })
                        
                        # Try to display a frame
                        try:
                            cv2.imshow(f"Camera {index} ({backend_name})", frame)
                            cv2.waitKey(1000)  # Show for 1 second
                            cv2.destroyAllWindows()
                        except Exception as e:
                            print(f"   ⚠️ Could not display frame: {e}")
                    else:
                        print(f"   ❌ Camera {index}: Opened but can't read frames")
                else:
                    print(f"   ❌ Camera {index}: Not accessible")
                    
            except Exception as e:
                print(f"   ❌ Camera {index}: Error - {e}")
            finally:
                if 'cap' in locals() and cap:
                    cap.release()
    
    # Summary
    print(f"\n📋 Summary:")
    if working_configs:
        print(f"   ✅ Working configurations: {len(working_configs)}")
        for config in working_configs:
            print(f"      • Index {config['index']} ({config['backend']}): {config['resolution']} @ {config['fps']:.1f}fps")
        
        # Test the first working configuration
        print(f"\n🧪 Testing first working configuration...")
        first_config = working_configs[0]
        try:
            cap = cv2.VideoCapture(first_config['index'], first_config['backend_code'])
            if cap.isOpened():
                print(f"   Testing camera {first_config['index']} for 5 seconds...")
                start_time = time.time()
                frame_count = 0
                
                while time.time() - start_time < 5:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frame_count += 1
                        cv2.imshow("Test Camera", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print("   ⚠️ Failed to read frame during test")
                        break
                
                actual_fps = frame_count / 5
                print(f"   ✅ Test completed: {frame_count} frames in 5 seconds ({actual_fps:.1f} FPS)")
                
                cap.release()
                cv2.destroyAllWindows()
            else:
                print("   ❌ Failed to open camera for test")
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
    else:
        print("   ❌ No working camera configurations found")
        print("\n💡 Troubleshooting tips:")
        print("   • Check camera permissions in Windows Settings")
        print("   • Ensure camera is not being used by another application")
        print("   • Try updating camera drivers")
        print("   • Test camera in Windows Camera app")
        print("   • Restart your computer")
    
    return working_configs

if __name__ == "__main__":
    try:
        working_configs = test_webcam()
        if working_configs:
            print(f"\n✅ Webcam test completed successfully!")
            print(f"Found {len(working_configs)} working configuration(s)")
        else:
            print(f"\n❌ No working webcam configurations found")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n👋 Test completed.")
    input("Press Enter to exit...") 