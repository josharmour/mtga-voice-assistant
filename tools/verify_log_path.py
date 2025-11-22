import os
log_path = "/mnt/c/Users/joshu/AppData/LocalLow/Wizards Of The Coast/MTGA/Player.log"
if os.path.exists(log_path):
    print(f"✅ Found log at: {log_path}")
else:
    print(f"❌ Log NOT found at: {log_path}")
