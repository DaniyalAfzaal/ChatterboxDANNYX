import requests
import time

BASE_URL = "https://daniyalabbasi-pk--chatterbox-async-v1-entrypoint.modal.run"

print("Waiting for server to be ready...")
for i in range(30):
    try:
        # Check if /docs is accessible (internal server is up)
        # Note: The proxy returns 502 if internal server is down
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("Server is READY!")
            break
    except:
        pass
    print(f"   Waiting... ({i+1}/30)")
    time.sleep(5)
else:
    print("Server did not start in time.")
    # Try to submit anyway to see error
    
# Test text
test_text = "This is a test of the async job queue system. The server should now be ready."

print("\nSubmitting job...")
try:
    response = requests.post(f"{BASE_URL}/jobs/submit", json={
        "text": test_text,
        "voice_mode": "predefined",
        "predefined_voice_id": "Neil_de_Grass_30s.mp3",
        "chunk_size": 150,
        "split_text": True
    })
    
    print(f"HTTP Status: {response.status_code}")
    
    if response.status_code == 200:
        job_data = response.json()
        job_id = job_data["job_id"]
        print(f"Job ID: {job_id}")
        
        # Poll
        print("\nProcessing...")
        while True:
            time.sleep(5)
            status_response = requests.get(f"{BASE_URL}/jobs/{job_id}/status")
            status = status_response.json()
            
            print(f"  {status['progress_percent']:.1f}% - {status['current_step']}")
            
            if status['status'] == 'completed':
                print(f"\nCOMPLETED!")
                
                # Download
                result = requests.get(f"{BASE_URL}/jobs/{job_id}/result")
                filename = f"async_success_{job_id}.wav"
                with open(filename, "wb") as f:
                    f.write(result.content)
                print(f"Saved to {filename}")
                break
                
            elif status['status'] == 'failed':
                print(f"\nFAILED: {status['error_message']}")
                break
    else:
        print(f"\nFailed: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"\nError: {e}")
