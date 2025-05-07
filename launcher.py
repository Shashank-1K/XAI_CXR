import subprocess
import os

if __name__ == "__main__":
    # Run the Streamlit app from the current directory
    subprocess.call(["streamlit", "run", "app.py"])