import requests

BASE_URL = "https://daniyalabbasi-pk--chatterbox-async-v1-entrypoint.modal.run"

# Get latest job from failing test - check last 2 jobs
status_resp = requests.get(f"{BASE_URL}/jobs/allure")
if status_resp.status_code != 404:
    print(status_resp.json())
else:
    # Manually check the recent job IDs
    # From the test output, we can extract job IDs
    import os
    import glob
    
    # Check current directory for any recent output
    recent_files = sorted(glob.glob("*.wav"), key=os.path.getmtime, reverse=True)
    print(f"Recent WAV files: {recent_files}")
    
    # Check Modal logs directly
    print("\nTrying to get last job status from Modal...")
    
    # Since we don't have the job ID, let's create one more test job
    print("Submitting a new test job...")
    resp = requests.post(f"{BASE_URL}/jobs/submit", json={
        "text": "Quick test",
        "voice_mode": "predefined",
        "predefined_voice_id": "Olivia.wav"
    })
    
    if resp.status_code == 200:
        job_id = resp.json()["job_id"]
        print(f"Submitted job: {job_id}")
        
        import time
        for i in range(60):
            time.sleep(2)
            resp = requests.get(f"{BASE_URL}/jobs/{job_id}/status")
            if resp.status_code == 200:
                data = resp.json()
                print(f"[{i*2}s] {data.get('status')} - {data.get('message')}")
                
                if data.get("status") == "failed":
                    print(f"\n❌ ERROR: {data.get('error')}")
                    
                    # Get full log
                    resp = requests.get(f"{BASE_URL}/jobs/{job_id}/log")
                    if resp.status_code == 200:
                        print("\n=== FULL DEBUG LOG ===")
                        print(resp.text)
                        print("=" * 80)
                    break
                elif data.get("status") == "completed":
                    print(f"\n✅ SUCCESS!")
                    
                    # Download
                    resp = requests.get(f"{BASE_URL}/jobs/{job_id}/result")
                    if resp.status_code == 200:
                        with open(f"FINAL_SUCCESS.wav", "wb") as f:
                            f.write(resp.content)
                        print(f"✅ Saved: FINAL_SUCCESS.wav ({len(resp.content)} bytes)")
                    break
