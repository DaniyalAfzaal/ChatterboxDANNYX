import requests
import time

# Read audiobook text
with open("audiobook_script.txt", "r", encoding="utf-8") as f:
    audiobook_text = f.read()

print(f"📖 Audiobook text loaded: {len(audiobook_text)} characters")

# Submit via /jobs/submit (what the UI uses for long text)
url = "https://daniyalabbasi-pk--chatterbox-async-v1-entrypoint.modal.run/jobs/submit"

payload = {
    "text": audiobook_text,
    "voice_mode": "predefined",
    "predefined_voice_id": "Neil_de_Grass_30s.mp3",
    "generation_params": {
        "temperature": 0.8
    }
}

print(f"\n🚀 Submitting long audiobook job (simulating UI)...")
response = requests.post(url, json=payload)
print(f"Status: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    job_id = result.get("job_id")
    print(f"✅ Job submitted! Job ID: {job_id}")
    print(f"\n⏳ Monitoring job progress...")
    print(f"Job will auto-show in /manager page when status is updated")
    print(f"\n📋 Direct job links:")
    print(f"   Status: https://daniyalabbasi-pk--chatterbox-async-v1-entrypoint.modal.run/jobs/{job_id}/status")
    print(f"   Manager: https://daniyalabbasi-pk--chatterbox-async-v1-entrypoint.modal.run/manager")
    print(f"   Download (when complete): https://daniyalabbasi-pk--chatterbox-async-v1-entrypoint.modal.run/jobs/{job_id}/result")
    
    # Monitor for first 2 minutes to verify it starts
    print(f"\n📊 Monitoring for 2 minutes to verify job starts...")
    for i in range(24):  # 2 minutes (24 * 5 seconds)
        time.sleep(5)
        status_response = requests.get(f"https://daniyalabbasi-pk--chatterbox-async-v1-entrypoint.modal.run/jobs/{job_id}/status")
        if status_response.status_code == 200:
            status_data = status_response.json()
            status = status_data.get("status", "unknown")
            message = status_data.get("message", "")
            progress = status_data.get("progress", 0)
            print(f"[{(i+1)*5}s] Status: {status}, Progress: {progress:.1f}%, Message: {message}")
            
            if status == "completed":
                print(f"\n🎉 Job completed!")
                break
            elif status == "failed":
                print(f"\n❌ Job failed: {status_data.get('error', 'Unknown error')}")
                break
        else:
            print(f"[{(i+1)*5}s] Could not check status (HTTP {status_response.status_code})")
    
    print(f"\n✅ Test complete! Job ID: {job_id}")
    print(f"Job will continue processing. Check /manager page for status.")
    print(f"Expected completion time: ~15-20 minutes")
    
else:
    print(f"❌ Failed to submit job:")
    print(response.text)
