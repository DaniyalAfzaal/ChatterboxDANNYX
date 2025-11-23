import requests
import time

BASE_URL = "http://127.0.0.1:8000"

print("Waiting for LOCAL server to be ready...")
for i in range(10):
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=2)
        if response.status_code == 200:
            print("Local Server is READY!")
            break
    except:
        pass
    print(f"   Waiting... ({i+1}/10)")
    time.sleep(2)

# Test text
test_text = "Local test of async job queue."

print("\nSubmitting job locally...")
try:
    response = requests.post(f"{BASE_URL}/jobs/submit", json={
        "text": test_text,
        "voice_mode": "predefined",
        "predefined_voice_id": "Neil_de_Grass_30s.mp3",
        "chunk_size": 150,
        "split_text": True
    })
    
    print(f"HTTP Status: {response.status_code}")
    print(f"Response: {response.text}")
    
except Exception as e:
    print(f"\nError: {e}")
