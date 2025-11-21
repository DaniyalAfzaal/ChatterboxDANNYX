import requests
import time
import json

BASE_URL = "https://ahmedmeyahi2017--chatterboxed-entrypoint-dev.modal.run"

# Simple test text first
test_text = "This is a simple test of the async job queue system."

# 1. Submit with detailed error handling
print("📤 Submitting job...")
try:
    response = requests.post(f"{BASE_URL}/jobs/submit", json={
        "text": test_text,
        "voice_mode": "predefined",
        "predefined_voice_id": "Neil_de_Grass_30s.mp3",
        "chunk_size": 150,
        "split_text": True,
        "allow_partial_success": True
    })
    
    print(f"HTTP Status: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Raw Response: {response.text[:500]}")  # First 500 chars
    
    if response.status_code == 200:
        job_data = response.json()
        job_id = job_data["job_id"]
        print(f"✅ Job ID: {job_id}")
        
        # 2. Poll
        print("\n⏳ Processing...")
        while True:
            time.sleep(5)
            status_response = requests.get(f"{BASE_URL}/jobs/{job_id}/status")
            
            if status_response.status_code != 200:
                print(f"❌ Status check failed: {status_response.status_code}")
                print(f"Response: {status_response.text}")
                break
                
            status = status_response.json()
            
            print(f"  {status['progress_percent']:.1f}% - {status['current_step']}")
            
            if status['status'] == 'completed':
                print(f"\n✅ COMPLETED! Audio duration: {status.get('audio_duration_seconds', 'Unknown')}s")
                
                # 3. Download
                print("\n📥 Downloading...")
                result = requests.get(f"{BASE_URL}/jobs/{job_id}/result")
                filename = f"test_audio_{job_id}.wav"
                with open(filename, "wb") as f:
                    f.write(result.content)
                print(f"✅ Saved to {filename}")
                break
                
            elif status['status'] == 'failed':
                print(f"\n❌ FAILED: {status['error_message']}")
                break
    else:
        print(f"\n❌ Failed to submit job!")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
