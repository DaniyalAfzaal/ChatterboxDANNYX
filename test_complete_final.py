import requests
import time

BASE_URL = "https://daniyalabbasi-pk--chatterbox-async-v1-entrypoint.modal.run"

print("\n" + "=" * 80)
print("FINAL TEST: Native Job Queue with Voice Files + Tensor Fix")
print("=" * 80 + "\n")

resp = requests.post(f"{BASE_URL}/jobs/submit", json={
    "text": "This is the final test to verify the complete TTS pipeline is working correctly.",
    "voice_mode": "predefined",
    "predefined_voice_id": "Olivia.wav"
})

if resp.status_code != 200:
    print(f"❌ Submit failed: {resp.status_code}")
    print(resp.text)
    exit(1)

job_id = resp.json()["job_id"]
print(f"✅ Job submitted: {job_id}\n")

# Poll
for i in range(120):
    resp = requests.get(f"{BASE_URL}/jobs/{job_id}/status")
    if resp.status_code == 200:
        data = resp.json()
        status = data.get("status")
        progress = data.get("progress", 0)
        message = data.get("message", "")
        
        print(f"\r[{i}s] {status.upper():15} | {progress:5.1f}% | {message}", end="", flush=True)
        
        if status == "completed":
            print(f"\n\n🎉 SUCCESS! Job completed in {i} seconds")
            
            # Download
            resp = requests.get(f"{BASE_URL}/jobs/{job_id}/result")
            if resp.status_code == 200:
                filename = f"FINAL_WORKING_{job_id[:8]}.wav"
                with open(filename, "wb") as f:
                    f.write(resp.content)
                print(f"✅ Audio file saved: {filename}")
                print(f"✅ File size: {len(resp.content):,} bytes")
                print("\n" + "=" * 80)
                print("🎊 COMPLETE SUCCESS - TTS PIPELINE FULLY OPERATIONAL!")
                print("=" * 80)
            else:
                print(f"\n⚠️ Result download failed: {resp.status_code}")
            break
            
        elif status == "failed":
            print(f"\n\n❌ Job failed: {data.get('error')}")
            
            # Get log
            resp = requests.get(f"{BASE_URL}/jobs/{job_id}/log")
            if resp.status_code == 200:
                print("\nLast 15 lines of debug log:")
                print("-" * 80)
                lines = resp.text.strip().split('\n')
                for line in lines[-15:]:
                    print(line)
                print("-" * 80)
            break
    
    time.sleep(1)
else:
    print(f"\n\n⏱️ Test timed out after 120 seconds")
