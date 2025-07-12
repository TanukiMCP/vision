# TanukiMCP Vision Server with Video Recording

A comprehensive computer vision MCP server that provides both basic screen capture/interaction tools and advanced video recording capabilities for temporal analysis.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Check if everything is installed correctly
python install_check.py
```

### 2. Run the Server

```bash
# Start the server
python server.py

# Or with custom host/port
python server.py --host 0.0.0.0 --port 8080
```

### 3. Connect from Cursor

Add to your MCP settings:
```json
{
  "mcpServers": {
    "tanuki-vision": {
      "command": "python",
      "args": ["path/to/server.py"],
      "transport": "streamable-http",
      "endpoint": "http://localhost:8000/mcp"
    }
  }
}
```

## 🛠️ Dependencies

### Core Dependencies (Required)
- `fastmcp` - MCP server framework
- `opencv-python` - Computer vision
- `pillow` - Image processing
- `numpy` - Numerical operations
- `pyautogui` - UI automation
- `mss` - Screen capture
- `uvicorn` - Web server

### Video Dependencies (Optional)
- `ffmpeg-python` - Video processing backend
- `moviepy` - Video creation and manipulation
- `imageio` - Image sequence handling

**Note**: If video dependencies are missing, the server will still run but video recording features will be disabled with clear error messages.

## 🎯 Available Tools

### Basic Vision Tools (8)
1. `capture_screen()` - Take screenshots
2. `click_coordinate()` - Click at specific coordinates
3. `type_text()` - Type text
4. `press_keys()` - Press keyboard shortcuts
5. `scroll()` - Scroll mouse wheel
6. `move_mouse()` - Move mouse cursor
7. `drag()` - Drag between coordinates
8. `get_screen_info()` - Get monitor information

### Video Recording Tools (7)
1. `start_screen_recording()` - Begin recording session
2. `stop_screen_recording()` - End recording and save
3. `get_frame_at_time()` - Extract frame at timestamp
4. `get_key_frames()` - Extract key frames
5. `analyze_motion_in_range()` - Detect motion between frames
6. `correlate_actions_with_motion()` - Link actions to visual changes
7. `cleanup_recording_session()` - Clean up temporary files

## 📝 Example Usage

### Basic Screenshot and Interaction
```python
# Take a screenshot
result = await capture_screen()
screenshot_b64 = result["screenshot"]

# Click on coordinates identified by LLM analysis
await click_coordinate(x=100, y=200)

# Type text
await type_text("Hello World")
```

### Video Recording Workflow
```python
# Start recording
session = await start_screen_recording(fps=12, max_duration=120)
session_id = session["session_id"]

# Perform actions (automatically logged)
await click_coordinate(100, 200)
await type_text("test data")
await scroll(clicks=3, direction="down")

# Analyze the recording
motion = await analyze_motion_in_range(session_id, 0, 30)
key_frames = await get_key_frames(session_id, max_frames=5)

# Extract specific frames for analysis
frame = await get_frame_at_time(session_id, timestamp=15.5)

# Correlate actions with visual changes
correlations = await correlate_actions_with_motion(session_id, 0, 30)

# Clean up
await cleanup_recording_session(session_id)
```

## 🏗️ Architecture

The server follows a "Smart Agent, Simple Tools" architecture:

- **LLM Agent (Cursor)**: Performs intelligent analysis and decision making
- **MCP Transport**: Carries JSON data and base64 images
- **Vision Server**: Provides primitive capture, interaction, and video tools

This design works within MCP's constraints while enabling powerful video analysis capabilities.

## 🔧 Configuration

Edit the `VisionConfig` class in `server.py`:

```python
@dataclass
class VisionConfig:
    # Basic settings
    screenshot_quality: int = 85
    click_delay: float = 0.1
    
    # Video settings
    video_fps: int = 10
    max_recording_duration: int = 300  # 5 minutes
    motion_threshold: float = 30.0
    enable_action_logging: bool = True
```

## 🐛 Troubleshooting

### Check Dependencies
```bash
python install_check.py
```

### Common Issues

1. **"Video processing dependencies not available"**
   - Install video dependencies: `pip install ffmpeg-python moviepy imageio`
   - The server will work without these, just without video features

2. **"Import mss could not be resolved"**
   - Install: `pip install mss`
   - This is required for screen capture

3. **"FastMCP not found"**
   - Install: `pip install fastmcp`
   - This is the core MCP framework

### Linter Warnings

The code may show linter warnings about missing imports when optional dependencies aren't installed. This is expected behavior - the server gracefully handles missing dependencies at runtime.

## 🎬 Use Cases

- **UI Testing**: Record interactions and verify visual feedback
- **Bug Reproduction**: Capture issues with temporal context
- **Performance Analysis**: Measure UI response times
- **UX Research**: Analyze user behavior patterns
- **Accessibility Testing**: Verify keyboard navigation
- **Cross-Browser Testing**: Compare workflows across browsers

## 📊 Video Analysis Features

- **Motion Detection**: Frame-by-frame difference analysis
- **Action Correlation**: Link user actions to visual changes
- **Key Frame Extraction**: Intelligent frame sampling
- **Temporal Analysis**: Find significant visual changes
- **Memory Efficient**: Rolling buffer with configurable limits

## 🔒 Security Notes

- Screen recording is performed locally
- No data is transmitted outside your system
- Temporary video files are automatically cleaned up
- All processing happens on the MCP server

## 📄 License

This project is part of the TanukiMCP ecosystem. See the main project for licensing information. "# vision" 
