import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('frontend/index.html', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if 'id="view-' in line or 'live-btn-mic' in line or 'live-call' in line:
            print(f"{i}: {line.strip()[:90]}")
