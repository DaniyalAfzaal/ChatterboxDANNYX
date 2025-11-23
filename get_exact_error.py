import requests
import json

BASE_URL = "https://daniyalabbasi-pk--chatterbox-async-v1-entrypoint.modal.run"

# Submit a test
resp = requests.post(f"{BASE_URL}/jobs/submit", json={
    "text": "Test",
    "voice_mode": "predefined",
    "predefined_voice_id": "Olivia.wav"
})

job_id = resp.json()["job_id"]
print(f"Job: {job_id}")

import time
for _ in range(60):
    time.sleep(2)
    resp = requests.get(f"{BASE_URL}/jobs/{job_id}/status")
    if resp.status_code == 200:
        data = resp.json()
        status = data.get("status")
        
        if status == "failed":
            # Print EXACT error
            error = data.get("error", "").encode('utf-8', errors='replace').decode('utf-8')
            print("ERROR MESSAGE:")
            print(error)
            
            # Save to file
            with open("exact_error.txt", "w", encoding='utf-8') as f:
                f.write(error)
            print("\nSaved to exact_error.txt")
            break
        elif status == "completed":
            print("SUCCESS!")
            break
