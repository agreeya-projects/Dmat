from dotenv import load_dotenv
import os
import subprocess
def set_open_key():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    try:
        if openai_api_key:
            print("Open API Key Successfully Set")
    except Exception as e:
        print("Print Open API KEY",e)

