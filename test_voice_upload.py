"""
Test voice upload functionality
"""
import requests
from pathlib import Path

# Modal endpoint
ENDPOINT = "https://daniyalabbasi-pk--chatterbox-async-v1-entrypoint.modal.run"

print("🔊 Testing Voice Upload...\n")

# Find a test voice file
test_voice = Path("voices/Olivia.wav")

if not test_voice.exists():
    print(f"❌ Test file not found: {test_voice}")
    exit(1)

print(f"📤 Uploading {test_voice.name}...")

# Upload the file
with open(test_voice, "rb") as f:
    files = {"file": (test_voice.name, f, "audio/wav")}
    response = requests.post(f"{ENDPOINT}/upload_predefined_voice", files=files, timeout=60)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}\n")

if response.json().get("success"):
    print("✅ Voice uploaded successfully!")
    
    # Verify it appears in the voices list
    print("\n📋 Checking voices list...")
    voices_response = requests.get(f"{ENDPOINT}/get_predefined_voices", timeout=10)
    voices = voices_response.json()
    
    print(f"Found {len(voices)} voices:")
    for v in voices:
        marker = "✅" if v['id'] == test_voice.name else "  "
        print(f"  {marker} {v['name']} ({v['id']})")
    
    if any(v['id'] == test_voice.name for v in voices):
        print("\n✅ UPLOAD TEST PASSED!")
    else:
        print("\n❌ Voice not found in list!")
else:
    print(f"❌ Upload failed: {response.json().get('error')}")
