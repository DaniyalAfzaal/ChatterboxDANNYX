import requests
import json
import time

# API endpoint
BASE_URL = "https://daniyalabbasi-pk--chatterbox-async-v1-entrypoint.modal.run"

# Read audiobook script
with open("audiobook_script.txt", "r", encoding="utf-8") as f:
    audiobook_text = f.read().strip()

print(f"📖 Audiobook text loaded: {len(audiobook_text)} characters, ~{len(audiobook_text.split())} words\n")

# Prepare payload
payload = {
    "text": audiobook_text,
    "voice_mode": "predefined",
    "predefined_voice_id": "Neil_de_Grass_30s.mp3",
    "temperature": 0.7,
    "speed": 1.0,
    "top_p": 0.9,
    "top_k": 50
}

print("🚀 Submitting long audiobook job...")
submit_response = requests.post(f"{BASE_URL}/jobs/submit", json=payload)
print(f"Status: {submit_response.status_code}")

if submit_response.status_code != 200:
    print(f"❌ Error: {submit_response.text}")
    exit(1)

submit_data = submit_response.json()
job_id = submit_data.get("job_id")
print(f"✅ Job submitted! Job ID: {job_id}\n")

# Poll for completion
print("⏳ Polling for job completion...")
max_wait_time = 30 * 60  # 30 minutes
poll_interval = 5  # 5 seconds
elapsed = 0

while elapsed < max_wait_time:
    time.sleep(poll_interval)
    elapsed += poll_interval
    
    status_response = requests.get(f"{BASE_URL}/jobs/{job_id}/status")
    if status_response.status_code != 200:
        print(f"❌ Failed to get status: {status_response.status_code}")
        continue
    
    status_data = status_response.json()
    status = status_data.get("status")
    
    print(f"[{elapsed}s] Status: {status}")
    
    if status == "completed":
        print("\n✅ Job completed! Downloading audio...")
        result_response = requests.get(f"{BASE_URL}/jobs/{job_id}/result")
        
        if result_response.status_code == 200:
            output_file = f"audiobook_output_{job_id[:8]}.wav"
            with open(output_file, "wb") as f:
                f.write(result_response.content)
            
            size_mb = len(result_response.content) / 1024 / 1024
            print(f"✅ SUCCESS! Audio saved to {output_file} ({size_mb:.2f} MB)")
            print(f"⏱️ Total time: {elapsed}s ({elapsed/60:.1f} minutes)")
        else:
            print(f"❌ Failed to download: {result_response.status_code}")
        break
        
    elif status == "failed":
        print(f"\n❌ Job failed!")
        error = status_data.get("error", "Unknown error")
        print(f"Error: {error}")
        
        # Try to get log
        log_response = requests.get(f"{BASE_URL}/jobs/{job_id}/log")
        if log_response.status_code == 200:
            print("\n📄 Debug log:")
            print(log_response.text)
        break
        
    elif status == "processing" or status == "queued":
        continue
    else:
        print(f"⚠️ Unknown status: {status}")

if elapsed >= max_wait_time:
    print(f"\n⏱️ Timeout after {max_wait_time/60} minutes")
    print(f"Check job status manually at: {BASE_URL}/manager")
