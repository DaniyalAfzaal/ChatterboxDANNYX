import requests
import time

BASE_URL = "https://daniyalabbasi-pk--chatterbox-final-entrypoint.modal.run"

# Test text
test_text = "This is a test of the async job queue system that was just deployed from GitHub. The async endpoints should now work perfectly!"

print("📤 Submitting job to FINAL deployment...")
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
        print(f"✅ Job ID: {job_id}")
        
        # Poll
        print("\n⏳ Processing...")
        while True:
            time.sleep(5)
            status_response = requests.get(f"{BASE_URL}/jobs/{job_id}/status")
            status = status_response.json()
            
            print(f"  {status['progress_percent']:.1f}% - {status['current_step']}")
            
            if status['status'] == 'completed':
                print(f"\n✅ COMPLETED!")
                
                # Download
                result = requests.get(f"{BASE_URL}/jobs/{job_id}/result")
                filename = f"async_success_{job_id}.wav"
                with open(filename, "wb") as f:
                    f.write(result.content)
                print(f"✅ Saved to {filename}")
                print(f"\n🎉🎉🎉 ASYNC JOB QUEUE IS WORKING! NO MORE 10-MINUTE TIMEOUTS! 🎉🎉🎉")
                break
                
            elif status['status'] == 'failed':
                print(f"\n❌ FAILED: {status['error_message']}")
                break
    else:
        print(f"\n❌ Failed: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
