import requests

# Check the first job from our test
job_id = "21933080-6b22-4553-ad2e-19c057414378"
BASE_URL = "https://daniyalabbasi-pk--chatterbox-async-v1-entrypoint.modal.run"

print(f"Checking job: {job_id}")
print("=" * 80)

resp = requests.get(f"{BASE_URL}/jobs/{job_id}/status")
print(f"Status endpoint: {resp.status_code}")
if resp.status_code == 200:
    import json
    status_data = resp.json()
    print(json.dumps(status_data, indent=2))
    
    if status_data.get("status") == "completed":
        print("\n✅ JOB COMPLETED! Downloading audio...")
        resp = requests.get(f"{BASE_URL}/jobs/{job_id}/result")
        if resp.status_code == 200:
            filename = f"FINAL_SUCCESS_{job_id[:8]}.wav"
            with open(filename, "wb") as f:
                f.write(resp.content)
            print(f"✅ Audio saved to: {filename}")
            print(f"   Size: {len(resp.content):,} bytes")
        else:
            print(f"❌ Result endpoint returned: {resp.status_code}")
    else:
        print(f"\n⚠️ Job status: {status_data.get('status')}")
else:
    print(f"Error: {resp.text}")
