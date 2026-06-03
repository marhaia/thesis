#!/usr/bin/env python3
"""Generate a simple test automotive GUI screenshot for testing."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_test_hmi():
    """Create a simple automotive HMI mock-up."""
    W, H = 1280, 720
    img = Image.new("RGB", (W, H), (20, 20, 30))  # dark background
    draw = ImageDraw.Draw(img)

    # Instrument cluster background (left)
    draw.ellipse([50, 150, 400, 550], outline=(60, 60, 80), width=3)
    draw.ellipse([80, 180, 370, 520], outline=(40, 120, 200), width=2)

    # Speed display
    draw.text((180, 320), "120", fill=(255, 255, 255))
    draw.text((180, 360), "km/h", fill=(150, 150, 150))

    # RPM gauge (right of speed)
    draw.ellipse([420, 200, 650, 500], outline=(60, 60, 80), width=3)
    draw.text((500, 330), "3.5", fill=(255, 255, 255))
    draw.text((490, 370), "x1000", fill=(150, 150, 150))

    # Central infotainment area
    draw.rectangle([700, 50, 1230, 670], outline=(60, 60, 80), width=2)

    # Navigation map (simplified)
    draw.rectangle([720, 70, 1210, 450], fill=(30, 50, 40))
    # Roads
    draw.line([(720, 260), (1210, 260)], fill=(100, 100, 120), width=3)
    draw.line([(950, 70), (950, 450)], fill=(100, 100, 120), width=3)
    draw.line([(800, 150), (1100, 380)], fill=(80, 80, 100), width=2)
    # Car position
    draw.ellipse([935, 245, 965, 275], fill=(0, 120, 255))

    # Media controls
    draw.rectangle([720, 480, 820, 530], fill=(50, 50, 70), outline=(80, 80, 100))
    draw.text((745, 495), "<<", fill=(200, 200, 200))

    draw.rectangle([840, 480, 940, 530], fill=(0, 100, 200), outline=(0, 140, 255))
    draw.text((875, 495), "▶", fill=(255, 255, 255))

    draw.rectangle([960, 480, 1060, 530], fill=(50, 50, 70), outline=(80, 80, 100))
    draw.text((990, 495), ">>", fill=(200, 200, 200))

    # Temperature display
    draw.rectangle([720, 560, 960, 640], fill=(40, 40, 55), outline=(60, 60, 80))
    draw.text((740, 580), "22.5°C", fill=(200, 200, 200))
    draw.text((740, 610), "A/C ON", fill=(0, 180, 100))

    # Volume slider
    draw.rectangle([980, 560, 1210, 640], fill=(40, 40, 55), outline=(60, 60, 80))
    draw.text((1000, 580), "Vol", fill=(150, 150, 150))
    draw.rectangle([1000, 610, 1100, 625], fill=(60, 60, 80))
    draw.rectangle([1000, 610, 1060, 625], fill=(0, 120, 255))

    # Status bar top
    draw.rectangle([0, 0, W, 35], fill=(15, 15, 25))
    draw.text((20, 8), "14:32", fill=(200, 200, 200))
    draw.text((W - 120, 8), "BMW iDrive", fill=(100, 100, 120))

    # Warning indicators
    draw.ellipse([450, 50, 480, 80], fill=(255, 180, 0))  # warning light
    draw.ellipse([500, 50, 530, 80], outline=(0, 200, 0), width=2)  # ok light

    out_path = os.path.join(os.path.dirname(__file__), "data", "screenshots", "test_hmi.png")
    img.save(out_path)
    print(f"Test HMI screenshot saved to: {out_path}")
    return out_path

if __name__ == "__main__":
    create_test_hmi()
