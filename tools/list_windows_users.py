import os
try:
    users = os.listdir("/mnt/c/Users")
    print(f"Users found: {users}")
except Exception as e:
    print(f"Error listing users: {e}")
