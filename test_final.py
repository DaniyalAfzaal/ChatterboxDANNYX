import requests
import time
import sys

BASE_URL = "https://daniyalabbasi-pk--chatterbox-async-v1-entrypoint.modal.run"

print("Submitting job...")
resp = requests.post(f"{BASE_URL}/jobs/submit", json={
    "text": "Hello world, this is a test of the native job queue.",
    "voice_mode": "predefined",
    "predefined_voice_id": "Olivia.wav"  # Use the existing voice
})

if resp.status_code != 200:
    print(f"❌ Submit failed: {resp.status_code} - {resp.text}")
    sys.exit(1)

job_id = resp.json()["job_id"]
print(f"✅ Job submitted: {job_id}\n")

# Poll for status
for i in range(60):  # Poll for up to 60 seconds
    resp = requests.get(f"{BASE_URL}/jobs/{job_id}/status")
    if resp.status_code == 200:
        data = resp.json()
        status = data.get("status")
        progress = data.get("progress", 0)
        message = data.get("message", "")
        print(f"[{i}s] Status: {status} | Progress: {progress:.1f}% | {message}")
        
        if status == "completed":
            print("\n✅ SUCCESS! Downloading result...")
            resp = requests.get(f"{BASE_URL}/jobs/{job_id}/result")
            if resp.status_code == 200:
                filename = f"output_{job_id}.wav"
                with open(filename, "wb") as f:
                    f.write(resp.content)
                print(f"✅ Saved to {filename}")
            break
        elif status == "failed":
            print(f"\n❌ JOB FAILED. Error: {data.get('error')}")
            # Fetch debug log
            resp = requests.get(f"{BASE_URL}/jobs/{job_id}/log")
            if resp.status_code == 200:
                print("\n" + "=" * 80)
                print("DEBUG LOG:")
                print("=" * 80)
                print(resp.text)
                print("=" * 80)
                with open("latest_debug_log.txt", "w") as f:
                    f.write(resp.text)
                print("\nSaved to latest_debug_log.txt")
            break
    else:
        print(f"[{i}s] Status check failed: {resp.status_code}")
    
    time.sleep(1)
