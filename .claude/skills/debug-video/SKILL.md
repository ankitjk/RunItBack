---
name: debug-video
description: Debug RunItBack detection on a video using verbose logging, saved debug frames, or single-timestamp inspection. Use when detection output is wrong and the cause needs diagnosing - bad game boundaries, wrong characters, or missed stock losses.
---

# /debug-video - Debug detection on a specific video

Debug frame extraction and detection for a SSBU replay video with verbose output.

## Usage

```
/debug-video <youtube_url>                    # Full debug analysis
/debug-video <youtube_url> --frames           # Extract and save debug frames
/debug-video <youtube_url> --timestamp <sec>  # Debug specific timestamp
```

## Instructions

### Full Debug Analysis

1. Run analysis with debug logging:
   ```bash
   python -m runitback analyze "<youtube_url>" --debug 2>&1 | head -200
   ```

2. Look for:
   - GO! screen detections and timestamps
   - GAME! screen detections and timestamps
   - Game boundary pairing logic
   - Character detection results
   - Stock loss events

### Extract Debug Frames

1. Use the frame extractor to save frames:
   ```bash
   python -c "
   from runitback.video.downloader import download_video
   from runitback.video.frame_extractor import FrameExtractor
   import cv2
   import os

   video_path = download_video('<youtube_url>')
   extractor = FrameExtractor(video_path, sample_fps=2.0)

   os.makedirs('debug_frames', exist_ok=True)
   for i, (frame, ts) in enumerate(extractor.extract_frames()):
       if i < 100:  # First 100 frames
           cv2.imwrite(f'debug_frames/frame_{i:04d}_{ts:.1f}s.png', frame)
   print(f'Saved frames to debug_frames/')
   "
   ```

2. Examine saved frames in `debug_frames/` directory.

### Debug Specific Timestamp

1. Extract frame at a specific timestamp and run detection:
   ```bash
   python -c "
   from runitback.video.downloader import download_video
   from runitback.cv.llm_detector import detect_game_screen_with_gemini
   import cv2

   video_path = download_video('<youtube_url>')
   cap = cv2.VideoCapture(video_path)
   cap.set(cv2.CAP_PROP_POS_MSEC, <timestamp_sec> * 1000)
   ret, frame = cap.read()
   cap.release()

   if ret:
       result = detect_game_screen_with_gemini(frame, use_cache=False)
       print(f'Detection result: {result}')
       cv2.imwrite('debug_frame.png', frame)
       print('Saved frame to debug_frame.png')
   "
   ```

## Common Debug Scenarios

### Game boundaries incorrect
- Check GO!/GAME! detection confidence
- Verify pairing logic in logs
- Look for replay menu GO! screens being included

### Character detection wrong
- Extract frame from character select or in-game
- Check Gemini response for character names
- Verify player 1 vs player 2 assignment

### Stock loss missed
- Check damage values around kill timestamp
- Verify stock counter detection
- Look for white flash detection

## Notes

- Debug frames are saved to `debug_frames/` (gitignored)
- Use `--debug` flag for verbose logging
- Set `use_cache=False` to force fresh API calls
- Run from the project root with the project environment active (`pip install -e .` per README)
