#!/usr/bin/env python3
"""
Check if camera is being used by other applications
"""

import cv2
import time
import psutil
import os

def check_camera_usage():
    """Check if camera is being used by other processes"""
    print("🔍 Camera Usage Check")
    print("=" * 30)
    
    # Common applications that might use the camera
    camera_apps = [
        "camera.exe", "CameraApp.exe", "WindowsCamera.exe",
        "Teams.exe", "zoom.exe", "skype.exe", "discord.exe",
        "chrome.exe", "firefox.exe", "edge.exe", "safari.exe",
        "obs64.exe", "obs32.exe", "obs-studio.exe",
        "streamlabs.exe", "xsplit.exe",
        "python.exe", "pythonw.exe"
    ]
    
    print("📋 Checking for applications that might use the camera...")
    camera_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'exe']):
        try:
            proc_name = proc.info['name'].lower()
            for app in camera_apps:
                if app.lower() in proc_name:
                    camera_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'exe': proc.info['exe']
                    })
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if camera_processes:
        print(f"   ⚠️ Found {len(camera_processes)} potentially conflicting processes:")
        for proc in camera_processes:
            print(f"      • {proc['name']} (PID: {proc['pid']})")
            if proc['exe']:
                print(f"        Path: {proc['exe']}")
        print("\n💡 Consider closing these applications before running Silent Echo")
    else:
        print("   ✅ No conflicting camera applications found")
    
    # Test camera availability
    print("\n📹 Testing camera availability...")
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Auto-detect")
    ]
    
    available_cameras = []
    
    for backend_code, backend_name in backends:
        for index in range(3):
            try:
                cap = cv2.VideoCapture(index, backend_code)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available_cameras.append({
                            'index': index,
                            'backend': backend_name,
                            'backend_code': backend_code
                        })
                        print(f"   ✅ Camera {index} available with {backend_name}")
                    cap.release()
                else:
                    cap.release()
            except Exception as e:
                print(f"   ❌ Camera {index} with {backend_name}: {e}")
    
    if available_cameras:
        print(f"\n✅ Found {len(available_cameras)} available camera(s)")
        return True
    else:
        print(f"\n❌ No cameras available")
        return False

def check_windows_camera_permissions():
    """Check Windows camera permissions"""
    print("\n🔐 Checking Windows camera permissions...")
    
    # This is a basic check - in a real implementation you'd need to use Windows API
    print("   💡 Manual check required:")
    print("   1. Open Windows Settings")
    print("   2. Go to Privacy & Security > Camera")
    print("   3. Ensure 'Camera access' is turned On")
    print("   4. Check that your Python/IDE has camera permissions")
    print("   5. If using a virtual environment, check if the Python executable has permissions")

def main():
    print("🤟 Silent Echo - Camera Usage Checker")
    print("=" * 40)
    
    # Check for conflicting processes
    has_conflicts = check_camera_usage()
    
    # Check Windows permissions
    check_windows_camera_permissions()
    
    # Test camera access
    print("\n🧪 Testing camera access...")
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print("   ✅ Camera access test successful")
                print(f"   📐 Frame size: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
            else:
                print("   ❌ Camera opened but can't read frames")
                cap.release()
        else:
            print("   ❌ Cannot open camera")
    except Exception as e:
        print(f"   ❌ Camera access test failed: {e}")
    
    print("\n📋 Summary:")
    if has_conflicts:
        print("   ⚠️ Potential camera conflicts detected")
        print("   💡 Close other camera applications and try again")
    else:
        print("   ✅ No camera conflicts detected")
    
    print("\n💡 Next steps:")
    print("   1. Run 'python test_webcam.py' for detailed camera testing")
    print("   2. If issues persist, try restarting your computer")
    print("   3. Update camera drivers if needed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Check interrupted by user")
    except Exception as e:
        print(f"\n❌ Check failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Check completed.")
    input("Press Enter to exit...") 