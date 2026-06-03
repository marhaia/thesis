"""Smoke test for /api/cognitive-load — verifies default is rule-based."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from io import BytesIO
from PIL import Image
from stage1.app import app

def make_test_image():
    img = Image.new("RGB", (64, 64), color=(200, 200, 200))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

client = app.test_client()

print("=== Test 1: Default (rule-based, no opt-in) ===")
resp = client.post("/api/cognitive-load", data={
    "image": (make_test_image(), "smoke.png"),
    "task_type": "search",
    "target_specificity": "medium",
    "time_pressure": "medium",
    "search_mode": "known_item",
    "profile_preset": "neutral",
    # use_trained_model NOT sent → defaults to False
})
data = resp.get_json()
print(f"  HTTP status:         {resp.status_code}")
print(f"  prediction_source:   {data.get('prediction_source')}")
print(f"  trained_requested:   {data.get('trained_model_requested')}")
print(f"  trained_available:   {data.get('trained_model_available')}")
print(f"  cognitive_load_score:{data.get('adjusted_prediction', {}).get('cognitive_load_score'):.1f}")
print()

print("=== Test 2: Explicit opt-in for trained model ===")
resp2 = client.post("/api/cognitive-load", data={
    "image": (make_test_image(), "smoke2.png"),
    "task_type": "search",
    "target_specificity": "medium",
    "time_pressure": "medium",
    "search_mode": "known_item",
    "profile_preset": "neutral",
    "use_trained_model": "true",
})
data2 = resp2.get_json()
print(f"  HTTP status:         {resp2.status_code}")
print(f"  prediction_source:   {data2.get('prediction_source')}")
print(f"  trained_requested:   {data2.get('trained_model_requested')}")
print(f"  trained_available:   {data2.get('trained_model_available')}")
print(f"  cognitive_load_score:{data2.get('adjusted_prediction', {}).get('cognitive_load_score'):.1f}")
