"""Smoke test for the active Stage-2 v1 route contract."""
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

print("=== Test 1: Default qualitative scenario proxy ===")
resp = client.post("/api/cognitive-load", data={
    "image": (make_test_image(), "smoke.png"),
    "task_type": "search",
    "time_pressure": "medium",
})
data = resp.get_json()
print(f"  HTTP status:         {resp.status_code}")
print(f"  scenario_proxy:      {data.get('stage2_scenario_proxy')}")
print(f"  cross_signal_review: {data.get('cross_signal_review')}")
print(f"  jokinen_diagnostic:  {data.get('jokinen_diagnostic')}")
print()

print("=== Test 2: Legacy ML field is rejected ===")
resp2 = client.post("/api/cognitive-load", data={
    "image": (make_test_image(), "smoke2.png"),
    "task_type": "search",
    "time_pressure": "medium",
    "use_trained_model": "true",
})
data2 = resp2.get_json()
print(f"  HTTP status:         {resp2.status_code}")
print(f"  error:               {data2.get('error')}")
