import requests
import time

BASE_URL = "https://daniyalabbasi-pk--chatterbox-final-entrypoint.modal.run"

# Simple test text
test_text = "This is a test of the async job queue system. The Modal timeout issue should now be completely resolved."

# 1. Submit
print("📤 Submitting job to new deployment...")
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
        print(f"✅ Estimated chunks: {job_data.get('estimated_chunks', 'N/A')}")
        
        # 2. Poll
        print("\n⏳ Processing...")
        while True:
            time.sleep(5)
            status_response = requests.get(f"{BASE_URL}/jobs/{job_id}/status")
            
            if status_response.status_code != 200:
                print(f"❌ Status check failed: {status_response.status_code}")
                break
                
            status = status_response.json()
            
            print(f"  {status['progress_percent']:.1f}% - {status['current_step']}")
            
            if status['status'] == 'completed':
                print(f"\n✅ COMPLETED!")
                print(f"   Duration: {status.get('audio_duration_seconds', 'Unknown')}s")
                
                # 3. Download
                print("\n📥 Downloading result...")
                result = requests.get(f"{BASE_URL}/jobs/{job_id}/result")
                if result.status_code == 200:
                    filename = f"async_test_{job_id}.wav"
                    with open(filename, "wb") as f:
                        f.write(result.content)
                    print(f"✅ Saved to {filename}")
                    print(f"\n🎉 ASYNC JOB QUEUE IS WORKING! No more 10-minute timeouts!")
                break
                
            elif status['status'] == 'failed':
                print(f"\n❌ FAILED: {status['error_message']}")
                break
    else:
        print(f"\n❌ Failed to submit job!")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
