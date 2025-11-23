"""
Quick test to call the API directly and see what error we get
"""
import requests

url = "https://daniyalabbasi-pk--chatterbox-async-v1-entrypoint.modal.run/tts"

payload = {
    "text": "The cosmos is within us. We are made of star-stuff.",
    "voice_mode": "predefined",
    "predefined_voice_id": "Neil_de_Grass_30s.mp3",
    "temperature": 0.7,
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "speed_factor": 1.0,
    "seed": 0,
    "language": "en",
    "split_text": True,
    "chunk_size": 120,
    "output_format": "mp3"
}

print("🚀 Submitting TTS job...")
response = requests.post(url, json=payload, timeout=120)

print(f"Status: {response.status_code}")
print(f"Headers: {response.headers}")

if response.status_code == 200:
    # Should be audio
    print(f"✅ SUCCESS! Got {len(response.content)} bytes")
    with open("test_output.mp3", "wb") as f:
        f.write(response.content)
    print("Saved to test_output.mp3")
else:
    # Error
    print(f"❌ FAILED")
    print(f"Response: {response.text}")
