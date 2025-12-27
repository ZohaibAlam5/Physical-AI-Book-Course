import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env")
else:
    print("Connecting to Google...")
    client = genai.Client(api_key=api_key)
    
    try:
        print("\n✅ AVAILABLE MODELS:")
        for m in client.models.list():
            # In the new SDK, we check 'supported_actions'
            if hasattr(m, 'supported_actions') and "generateContent" in m.supported_actions:
                # Strip 'models/' prefix for cleaner output
                clean_name = m.name.replace("models/", "")
                print(f"- {clean_name}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")