"""
Test long audiobook generation (50 pages / ~15,000 characters)
"""
import requests
import time
import json

ENDPOINT = "https://daniyalabbasi-pk--chatterbox-async-v1-entrypoint.modal.run"

# Generate a 50-page story (approximately 15,000 characters)
long_text = """
Chapter 1: The Beginning

In the soft glow of twilight, Sarah stood at the edge of the ancient forest, her heart pounding with anticipation. The trees whispered secrets carried on the evening breeze, their leaves rustling with stories untold. She had waited her entire life for this moment, when the stars aligned and the path would reveal itself.

""" * 50  # Repeat to make it long

char_count = len(long_text)
print(f"📚 Testing Long Audiobook Generation")
print(f"📝 Text length: {char_count:,} characters\n")

# Submit the job
print("🚀 Submitting job...")
payload = {
    "text": long_text,
    "voice_mode": "predefined",
    "predefined_voice_id": "Olivia.wav",
    "generation_params": {
        "temperature": 0.7,
        "chunk_size": 120
    }
}

try:
    response = requests.post(f"{ENDPOINT}/tts", json=payload, timeout=30)
    
    if response.status_code != 200:
        print(f"❌ Failed to submit: {response.status_code}")
        print(response.text)
        exit(1)
    
    result = response.json()
    
    if not result.get("success"):
        print(f"❌ Job submission failed: {result.get('error')}")
        exit(1)
    
    job_id = result["job_id"]
    print(f"✅ Job submitted: {job_id}\n")
    
    # Poll for status
    print("⏳ Monitoring progress...")
    start_time = time.time()
    last_progress = -1
    
    while True:
        status_response = requests.get(f"{ENDPOINT}/jobs/{job_id}/status", timeout=10)
        status = status_response.json()
        
        current_status = status.get("status", "unknown")
        progress = status.get("progress", 0)
        message = status.get("message", "")
        
        # Only print when progress changes significantly
        if progress != last_progress and (progress == 0 or progress == 100 or progress % 10 == 0):
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Status: {current_status} | Progress: {progress:.1f}% | {message}")
            last_progress = progress
        
        if current_status == "completed":
            elapsed = time.time() - start_time
            print(f"\n✅ COMPLETED in {elapsed:.1f}s!")
            
            # Download the result
            print("\n📥 Downloading audio...")
            audio_response = requests.get(f"{ENDPOINT}/jobs/{job_id}/result", timeout=60)
            
            if audio_response.status_code == 200:
                output_file = f"audiobook_{job_id}.wav"
                with open(output_file, "wb") as f:
                    f.write(audio_response.content)
                
                file_size = len(audio_response.content) / (1024 * 1024)  # MB
                print(f"✅ Saved to {output_file} ({file_size:.2f} MB)")
                print(f"\n🎉 LONG AUDIOBOOK TEST PASSED!")
            else:
                print(f"❌ Failed to download: {audio_response.status_code}")
            break
        
        elif current_status == "failed":
            error = status.get("error", "Unknown error")
            print(f"\n❌ Job failed: {error}")
            break
        
        time.sleep(2)
        
except KeyboardInterrupt:
    print("\n\n⚠️ Test interrupted by user")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
