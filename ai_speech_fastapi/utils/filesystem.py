import os
from datetime import datetime

def setup_user_directory(base_dir: str, username: str) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    user_dir = os.path.join(base_dir, username, date_str)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def remove_file_if_exists(path: str):
    if os.path.exists(path):
        os.remove(path)
