import requests
import time
import sys

BASE_URL = "https://daniyalabbasi-pk--chatterbox-async-v1-entrypoint.modal.run"

def test_native_queue():
    print(f"🚀 Testing Native Job Queue at {BASE_URL}")
    
    # 1. Submit Job
    payload = {
        "text": "This is a test of the native job queue architecture. It should run in a separate worker.",
        "voice_mode": "predefined",
        "predefined_voice_id": "default_sample.wav"
    }
    
    print("Submitting job...")
    try:
        resp = requests.post(f"{BASE_URL}/jobs/submit", json=payload)
        if resp.status_code != 200:
            print(f"❌ Submission failed: {resp.status_code} {resp.text}")
            return
            
        data = resp.json()
        job_id = data.get("job_id")
        print(f"✅ Job submitted! ID: {job_id}")
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return

    # 2. Poll Status
    print("Polling status...")
    start_time = time.time()
    while True:
        try:
            resp = requests.get(f"{BASE_URL}/jobs/{job_id}/status")
            if resp.status_code != 200:
                print(f"⚠️ Status check failed: {resp.status_code}")
                time.sleep(2)
                continue
                
            status_data = resp.json()
            status = status_data.get("status")
            progress = status_data.get("progress")
            message = status_data.get("message")
            
            print(f"Status: {status} | Progress: {progress}% | Message: {message}")
            
            if status == "completed":
                print("✅ Job completed successfully!")
                break
            elif status == "failed":
                print(f"❌ Job failed: {status_data.get('error')}")
                break
            
            if time.time() - start_time > 300: # 5 mins timeout
                print("❌ Timeout waiting for job completion.")
                break
                
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ Polling error: {e}")
            time.sleep(2)

    # 3. Download Result or Debug Log
    if status == "completed":
        print("Downloading result...")
        resp = requests.get(f"{BASE_URL}/jobs/{job_id}/result")
        if resp.status_code == 200:
            with open(f"output_{job_id}.wav", "wb") as f:
                f.write(resp.content)
            print(f"✅ Audio saved to output_{job_id}.wav")
        else:
            print(f"❌ Download failed: {resp.status_code}")
    elif status == "failed":
        print("\nFetching debug log...")
        resp = requests.get(f"{BASE_URL}/jobs/{job_id}/log")
        if resp.status_code == 200:
            print("=" * 80)
            print("DEBUG LOG:")
            print("=" * 80)
            print(resp.text)
            print("=" * 80)
        else:
            print(f"❌ Could not fetch log: {resp.status_code}")

if __name__ == "__main__":
    test_native_queue()
