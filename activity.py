import pyautogui
import random
import time

pyautogui.FAILSAFE = True

screen_width, screen_height = pyautogui.size()
INTERVAL = 60  # 3 minutes

# Avoid edges/top browser tabs area
SAFE_MARGIN = 150

def random_position():
    x = random.randint(SAFE_MARGIN, screen_width - SAFE_MARGIN)
    y = random.randint(SAFE_MARGIN, screen_height - SAFE_MARGIN)
    return x, y

def random_hover():
    x, y = random_position()

    duration = random.uniform(0.5, 2.0)

    # Always move first
    pyautogui.moveTo(x, y, duration=duration)

    # Tiny human-like jitter
    pyautogui.moveRel(
        random.randint(-10, 10),
        random.randint(-10, 10),
        duration=0.2
    )

def random_right_click():
    # Small pause after hover
    time.sleep(random.uniform(1, 3))

    pyautogui.click(button='right')

    # Close context menu safely
    time.sleep(1)
    pyautogui.press('esc')

print("Starting in 3 seconds...")
print("Move mouse to top-left corner to stop.")
time.sleep(3)

try:
    while True:

        # ALWAYS hover first
        random_hover()

        # Right click only sometimes
        if random.random() < 0.3:  # 30% chance
            random_right_click()
            action = "hover + right_click"
        else:
            action = "hover only"

        print(f"Action performed: {action}")
        print(f"Waiting {INTERVAL // 60} minutes...\n")

        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("Stopped manually.")