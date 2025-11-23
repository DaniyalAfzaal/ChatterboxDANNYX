import requests
import sys

BASE_URL = "https://daniyalabbasi-pk--chatterbox-async-v1-entrypoint.modal.run"

# Submit a simple job
print("Submitting test job...")
resp = requests.post(f"{BASE_URL}/jobs/submit", json={
    "text": "Test",
    "voice_mode": "predefined",
    "predefined_voice_id": "default_sample.wav"
})

if resp.status_code != 200:
    print(f"Failed to submit: {resp.status_code}")
    sys.exit(1)

job_id = resp.json()["job_id"]
print(f"Job ID: {job_id}")

import time
time.sleep(5)  # Wait for worker to start and fail

# Fetch the log
print("\nFetching debug log...")
resp = requests.get(f"{BASE_URL}/jobs/{job_id}/log")
if resp.status_code == 200:
    print("\n" + "=" * 80)
    print(resp.text)
    print("=" * 80)
    
    # Save to file
    with open("debug_log.txt", "w") as f:
        f.write(resp.text)
    print("\nSaved to debug_log.txt")
else:
    print(f"Could not fetch log: {resp.status_code}")
