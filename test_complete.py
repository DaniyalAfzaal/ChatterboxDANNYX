import requests
import time

BASE_URL = "https://daniyalabbasi-pk--chatterbox-async-v1-entrypoint.modal.run"

print("=" * 80)
print("TESTING NATIVE MODAL JOB QUEUE WITH VOICE FILES")
print("=" * 80)

# Submit job
print("\n1. Submitting TTS job...")
resp = requests.post(f"{BASE_URL}/jobs/submit", json={
    "text": "Hello, this is a test of the Chatterbox native job queue with voice cloning.",
    "voice_mode": "predefined",
    "predefined_voice_id": "Olivia.wav"
})

if resp.status_code != 200:
    print(f"❌ FAILED: {resp.status_code} - {resp.text}")
    exit(1)

job_id = resp.json()["job_id"]
print(f"✅ Job submitted successfully!")
print(f"   Job ID: {job_id}")

# Poll for completion
print("\n2. Waiting for completion...")
max_wait = 120  # seconds
start_time = time.time()

while (time.time() - start_time) < max_wait:
    resp = requests.get(f"{BASE_URL}/jobs/{job_id}/status")
    if resp.status_code == 200:
        data = resp.json()
        status = data.get("status")
        progress = data.get("progress", 0)
        message = data.get("message", "")
        
        print(f"   [{int(time.time() - start_time)}s] {status.upper()} | {progress:.0f}% | {message}")
        
        if status == "completed":
            print(f"\n✅ SUCCESS! Audio generated in {int(time.time() - start_time)} seconds")
            
            # Download result
            resp = requests.get(f"{BASE_URL}/jobs/{job_id}/result")
            if resp.status_code == 200:
                filename = f"test_output_{job_id[:8]}.wav"
                with open(filename, "wb") as f:
                    f.write(resp.content)
                print(f"✅ Audio saved to: {filename}")
                print(f"   File size: {len(resp.content)} bytes")
            break
        elif status == "failed":
            print(f"\n❌ JOB FAILED")
            print(f"   Error: {data.get('error')}")
            
            # Get debug log
            resp = requests.get(f"{BASE_URL}/jobs/{job_id}/log")
            if resp.status_code == 200:
                with open("failure_log.txt", "w") as f:
                    f.write(resp.text)
                print(f"   Debug log saved to: failure_log.txt")
                print("\n   Last 10 lines:")
                lines = resp.text.strip().split('\n')
                for line in lines[-10:]:
                    print(f"     {line}")
            break
            
    time.sleep(2)
else:
    print(f"\n⏱️ TIMEOUT after {max_wait} seconds")

print("\n" + "=" * 80)
