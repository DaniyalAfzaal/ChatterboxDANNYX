import requests
import json

BASE_URL = "https://daniyalabbasi-pk--chatterbox-async-v1-entrypoint.modal.run"

# Get the latest job by listing files
resp = requests.get(f"{BASE_URL}/custom-api/files")
if resp.status_code == 200:
    files = resp.json()
    if files:
        latest_file = files[-1]  # Last file
        print(f"Latest job file: {latest_file}")
        
        # Extract job ID from path like "jobs/xxx-yyy-zzz/result"
        job_id = latest_file.split("/")[1]
        print(f"Job ID: {job_id}")
        
        # Get status
        resp = requests.get(f"{BASE_URL}/jobs/{job_id}/status")
        if resp.status_code == 200:
            print(f"\nJob Status:")
            print(json.dumps(resp.json(), indent=2))
            
            # Try to download
            if resp.json().get("status") == "completed":
                resp = requests.get(f"{BASE_URL}/jobs/{job_id}/result")
                if resp.status_code == 200:
                    with open(f"SUCCESS_{job_id[:8]}.wav", "wb") as f:
                        f.write(resp.content)
                    print(f"\n✅ SUCCESS! Downloaded audio: SUCCESS_{job_id[:8]}.wav")
                    print(f"   Size: {len(resp.content)} bytes")
else:
    print("Failed to list files")
