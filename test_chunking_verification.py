"""
Test Long Audiobook with Chunking Verification
This script will:
1. Submit a ~500 word test audiobook
2. Monitor the job logs to see the verification messages
3. Download the final audio for manual listening
"""
import requests
import time

BASE_URL = "https://ahmedmeyahi2017--chatterbox-async-v1-entrypoint.modal.run"

# Create test text that's long enough to trigger chunking (>225 words)
test_text = """
Chapter One: The Beginning of a Journey. This is a story about artificial intelligence and natural language processing.
How does text-to-speech technology work? It's fascinating to explore the inner workings of neural networks.
The model learns patterns from vast amounts of training data. Each sentence contributes to the overall understanding.

Chapter Two: Technical Details. Machine learning algorithms process audio samples at incredible speeds.
The synthesis engine converts text into phonemes, then into audio waveforms. This process happens in multiple stages.
First, the text is analyzed and broken down into smaller components. Then each component is processed individually.

Chapter Three: Audio Quality Matters. When we concatenate audio chunks, we must ensure seamless transitions.
The sentences should flow naturally without abrupt pauses or clicking sounds. This is critical for audiobook quality.
Users expect professional-grade output that sounds completely natural and engaging throughout the entire narration.

Chapter Four: Verification and Testing. We've added logging to verify no words are lost during chunking.
The system counts words before and after splitting to ensure data integrity. This gives us mathematical proof.
If the counts match, we know for certain that all content has been preserved in the chunking process.

Chapter Five: Real World Application. Audiobooks can be quite long, sometimes hundreds of thousands of words.
The chunking system allows us to process these massive texts without running out of GPU memory.
By splitting at sentence boundaries, we maintain the natural flow and prosody of human speech patterns.

Conclusion: Moving Forward. This test will help us understand if audio quality is acceptable with direct concatenation.
Or if we need to implement crossfading at chunk boundaries for smoother transitions between generated segments.
The verification logs will confirm mathematically that no sentences or words have been skipped or lost.
""".strip()

print(f"[TEST] Test text: {len(test_text)} characters, ~{len(test_text.split())} words")
print(f"Expected: {len(test_text.split())} words across multiple chunks\n")

# Submit the job
payload = {
    "text": test_text,
    "voice_mode": "predefined",
    "predefined_voice_id": "Neil_de_Grass_30s.mp3",
    "generation_params": {
        "temperature": 0.7
    }
}

print("Submitting long text job...")
response = requests.post(f"{BASE_URL}/jobs/submit", json=payload)

if response.status_code != 200:
    print(f"ERROR: {response.text}")
    exit(1)

data = response.json()
job_id = data["job_id"]
print(f"SUCCESS - Job submitted! Job ID: {job_id}\n")

# Monitor for completion
print("Monitoring job (max 10 minutes)...")
for i in range(120):  # 10 minutes max
    time.sleep(5)
    
    # Check status
    status_resp = requests.get(f"{BASE_URL}/jobs/{job_id}/status")
    if status_resp.status_code == 200:
        status_data = status_resp.json()
        status = status_data.get("status")
        message = status_data.get("message", "")
        
        print(f"[{(i+1)*5}s] Status: {status}, Message: {message}")
        
        if status == "completed":
            print(f"\n🎉 Job completed!")
            
            # Download the audio
            result_resp = requests.get(f"{BASE_URL}/jobs/{job_id}/result")
            if result_resp.status_code == 200:
                output_file = f"test_chunking_{job_id[:8]}.wav"
                with open(output_file, "wb") as f:
                    f.write(result_resp.content)
                
                size_mb = len(result_resp.content) / 1024 / 1024
                print(f"✅ Audio saved: {output_file} ({size_mb:.2f} MB)")
            
            # Get and display the log to see verification messages
            log_resp = requests.get(f"{BASE_URL}/jobs/{job_id}/log")
            if log_resp.status_code == 200:
                print(f"\n📄 Debug Log (look for '✅ Verification passed'):")
                print("="*80)
                log_text = log_resp.text
                # Extract just the relevant verification lines
                for line in log_text.split('\n'):
                    if 'Verification' in line or 'chunks' in line.lower() or 'word' in line.lower():
                        print(line)
                print("="*80)
                
                # Save full log
                with open(f"test_chunking_{job_id[:8]}_log.txt", "w") as f:
                    f.write(log_text)
                print(f"\n📋 Full log saved: test_chunking_{job_id[:8]}_log.txt")
            
            print(f"\n🎧 Next step: Listen to {output_file} and check for:")
            print(f"   1. Any missing sentences")
            print(f"   2. Clicking/popping sounds between chunks")
            print(f"   3. Unnatural pauses")
            print(f"   4. Overall audio quality")
            
            break
            
        elif status == "failed":
            print(f"\n❌ Job failed: {status_data.get('error')}")
            break

print(f"\n✅ Test complete!")
print(f"Check the log for verification messages.")
print(f"Manager URL: {BASE_URL}/manager")
