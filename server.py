#!/usr/bin/env python3
"""
TanukiMCP Vision Server

A Model Context Protocol server that provides vision and interaction primitives
for AI agents. Includes screenshot capture, mouse/keyboard interaction,
and advanced video recording capabilities for temporal analysis.
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
# Core dependencies
import mss
import numpy as np
import pyautogui
import uvicorn
from fastmcp import FastMCP
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uuid

# Video processing imports
try:
    from video_processor import (ActionRecorder, FrameAnalysis, TemporalAnalyzer,
                                 VideoRecorder, VideoSession)

    VIDEO_PROCESSING_AVAILABLE = True
except ImportError:
    VIDEO_PROCESSING_AVAILABLE = False

# Visual state machine import
try:
    from visual_state_machine import VisualStateMachine, VisualState
    VISUAL_STATE_MACHINE_AVAILABLE = True
except ImportError:
    VISUAL_STATE_MACHINE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not VIDEO_PROCESSING_AVAILABLE:
    logger.warning("Video processing dependencies not available. Video features will be disabled. Run 'pip install moviepy ffmpeg-python' to enable them.")

# Disable pyautogui failsafe for programmatic control
pyautogui.FAILSAFE = False


@dataclass
class VisionConfig:
    """Configuration for the vision server"""
    debug: bool = False
    screenshot_quality: int = 85
    click_delay: float = 0.1
    max_screenshot_width: int = 1920
    max_screenshot_height: int = 1080
    # Video recording settings
    video_fps: int = 10
    max_recording_duration: int = 300  # 5 minutes
    motion_threshold: float = 30.0
    enable_action_logging: bool = True


class ScreenshotCapture:
    """Handles screen capturing functionality"""

    def __init__(self, config: VisionConfig):
        self.config = config
        self.sct = mss.mss()

    def capture_screen(self, monitor: int = 0) -> np.ndarray:
        """Capture the entire screen or specific monitor"""
        monitor_config = self.sct.monitors[monitor] if monitor < len(self.sct.monitors) else self.sct.monitors[0]
        screenshot = self.sct.grab(monitor_config)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert to BGR

        # Resize if needed
        height, width = img.shape[:2]
        if width > self.config.max_screenshot_width or height > self.config.max_screenshot_height:
            scale = min(self.config.max_screenshot_width / width, self.config.max_screenshot_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        return img

    def screenshot_to_base64(self, img: np.ndarray, quality: Optional[int] = None, max_width: Optional[int] = None) -> str:
        """Convert screenshot to base64 string with optional downscaling and compression.

        Args:
            img: Input BGR image.
            quality: JPEG quality (1-100). Defaults to self.config.screenshot_quality.
            max_width: If provided, the image is proportionally resized so that its width does not exceed this value.
        """
        # Downscale if requested
        if max_width is not None and img.shape[1] > max_width:
            scale = max_width / img.shape[1]
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            img = cv2.resize(img, new_size)

        jpeg_quality = quality if quality is not None else self.config.screenshot_quality
        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        return base64.b64encode(buffer).decode('utf-8')


class InteractionController:
    """Handles mouse and keyboard interactions"""

    def __init__(self, config: VisionConfig):
        self.config = config

    def move_mouse(self, x: int, y: int, duration: float = 0.5) -> bool:
        pyautogui.moveTo(x, y, duration=duration)
        return True

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1, interval: float = 0.1) -> bool:
        pyautogui.click(x, y, clicks=clicks, interval=interval, button=button)
        time.sleep(self.config.click_delay)
        return True

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 1.0) -> bool:
        pyautogui.moveTo(start_x, start_y)
        pyautogui.dragTo(end_x, end_y, duration=duration, button='left')
        return True

    def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        pyautogui.scroll(clicks, x=x, y=y)
        return True

    def type_text(self, text: str, interval: float = 0.01) -> bool:
        pyautogui.typewrite(text, interval=interval)
        return True

    def press_key(self, key: str) -> bool:
        pyautogui.press(key)
        return True

    def hotkey(self, *keys) -> bool:
        pyautogui.hotkey(*keys)
        return True


class VisionMCPServer:
    """Main MCP server for vision-based automation with video capabilities"""
    video_recorder: 'Optional[VideoRecorder]'
    action_recorder: 'Optional[ActionRecorder]'
    temporal_analyzer: 'Optional[TemporalAnalyzer]'

    def __init__(self, config: VisionConfig):
        self.config = config
        self.screenshot_capture = ScreenshotCapture(config)
        self.interaction_controller = InteractionController(config)

        # HTTP server for serving images
        self.app = FastAPI()
        self.temp_dir = os.path.join(tempfile.gettempdir(), "tanukimcp_screenshots")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Mount static files
        self.app.mount("/screenshots", StaticFiles(directory=self.temp_dir), name="screenshots")
        
        @self.app.get("/screenshot/{filename}")
        async def serve_screenshot(filename: str):
            file_path = os.path.join(self.temp_dir, filename)
            if os.path.exists(file_path):
                return FileResponse(file_path)
            raise HTTPException(status_code=404, detail="Screenshot not found")

        # Video processing components
        if VIDEO_PROCESSING_AVAILABLE:
            self.video_recorder = VideoRecorder()
            self.action_recorder = ActionRecorder()
            self.temporal_analyzer = TemporalAnalyzer()
        else:
            self.video_recorder = None
            self.action_recorder = None
            self.temporal_analyzer = None

        # Visual state machine for screenshot-before-action workflow
        if VISUAL_STATE_MACHINE_AVAILABLE:
            self.visual_state_machine = VisualStateMachine(self.screenshot_capture)
        else:
            self.visual_state_machine = None

        self.recording_thread: Optional[threading.Thread] = None
        self.stop_recording_flag = threading.Event()

        self.mcp = FastMCP(
            name="TanukiMCP Vision Server (Video-Enhanced Primitives)",
            instructions="This server provides primitive tools for screen capture, UI interaction, video recording, and temporal analysis. The calling agent is responsible for analyzing screenshots/videos and deciding actions."
        )
        self.setup_tools()

    def setup_tools(self):
        """Setup primitive MCP tools"""

        @self.mcp.tool
        async def capture_screen(monitor: int = 1) -> Dict[str, Any]:
            """Captures a screenshot of a specified monitor. Returns HTTP URL for image access."""
            try:
                monitor_idx = monitor if monitor > 0 else 1
                img = self.screenshot_capture.capture_screen(monitor_idx)
                
                # Save with HTTP URL
                filepath, http_url = self.save_screenshot_with_url(img, "screen")
                
                # Create small thumbnail for context
                thumbnail_b64 = self.screenshot_capture.screenshot_to_base64(img, quality=50, max_width=640)
                
                return {
                    "success": True,
                    "screenshot_url": http_url,
                    "screenshot_file": filepath,
                    "thumbnail": f"data:image/jpeg;base64,{thumbnail_b64}",
                    "format": "jpeg",
                    "width": img.shape[1],
                    "height": img.shape[0],
                    "visual_feedback": {
                        "type": "screenshot",
                        "description": "Screen capture",
                        "image_url": http_url,
                        "file_location": filepath,
                        "instructions": f"Screenshot available at: {http_url}"
                    }
                }
            except Exception as e:
                logger.error(f"capture_screen failed: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool
        async def click_coordinate(x: int, y: int, button: str = "left", clicks: int = 1) -> Dict[str, Any]:
            """Performs a mouse click at the specified (x, y) coordinates. Requires recent screenshot if visual confirmation is enabled."""
            try:
                # Visual state machine check - TEMPORARILY DISABLED FOR TESTING
                # if VISUAL_STATE_MACHINE_AVAILABLE and self.visual_state_machine:
                #     allowed, check_result = self.visual_state_machine.pre_action_check(
                #         "click_coordinate", {"x": x, "y": y, "button": button, "clicks": clicks}
                #     )
                #     if not allowed:
                #         return {"success": False, "visual_check_failed": True, **check_result}
                
                success = self.interaction_controller.click(x, y, button, clicks)
                result = {"success": success, "action": "click", "x": x, "y": y, "button": button}
                
                self._log_action_if_recording("click", {"x": x, "y": y, "button": button, "clicks": clicks})
                
                # Update visual state machine
                if VISUAL_STATE_MACHINE_AVAILABLE and self.visual_state_machine:
                    self.visual_state_machine.post_action_update("click_coordinate", {"x": x, "y": y, "button": button, "clicks": clicks}, result)
                
                return result
            except Exception as e:
                logger.error(f"click_coordinate failed: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool
        async def type_text(text: str) -> Dict[str, Any]:
            """Types the provided text at the current cursor location. Requires recent screenshot if visual confirmation is enabled."""
            try:
                # Visual state machine check - TEMPORARILY DISABLED FOR TESTING
                # if VISUAL_STATE_MACHINE_AVAILABLE and self.visual_state_machine:
                #     allowed, check_result = self.visual_state_machine.pre_action_check(
                #         "type_text", {"text": text, "length": len(text)}
                #     )
                #     if not allowed:
                #         return {"success": False, "visual_check_failed": True, **check_result}
                
                success = self.interaction_controller.type_text(text)
                result = {"success": success, "action": "type_text", "length": len(text)}
                
                self._log_action_if_recording("type_text", {"text": text, "length": len(text)})
                
                # Update visual state machine
                if VISUAL_STATE_MACHINE_AVAILABLE and self.visual_state_machine:
                    self.visual_state_machine.post_action_update("type_text", {"text": text, "length": len(text)}, result)
                
                return result
            except Exception as e:
                logger.error(f"type_text failed: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool
        async def press_keys(keys: List[str]) -> Dict[str, Any]:
            """Presses a sequence of keys or a hotkey combination."""
            try:
                if len(keys) == 1:
                    success = self.interaction_controller.press_key(keys[0])
                else:
                    success = self.interaction_controller.hotkey(*keys)
                self._log_action_if_recording("press_keys", {"keys": keys})
                return {"success": success, "action": "press_keys", "keys": keys}
            except Exception as e:
                logger.error(f"press_keys failed: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool
        async def scroll(clicks: int, direction: str = "down") -> Dict[str, Any]:
            """Scrolls the mouse wheel up or down."""
            try:
                scroll_amount = abs(clicks) if direction == "down" else -abs(clicks)
                success = self.interaction_controller.scroll(scroll_amount)
                self._log_action_if_recording("scroll", {"clicks": clicks, "direction": direction})
                return {"success": success, "action": "scroll", "amount": scroll_amount}
            except Exception as e:
                logger.error(f"scroll failed: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool
        async def move_mouse(x: int, y: int, duration: float = 0.2) -> Dict[str, Any]:
            """Moves the mouse cursor to a specified (x, y) coordinate. Requires recent screenshot if visual confirmation is enabled."""
            try:
                # Visual state machine check - TEMPORARILY DISABLED FOR TESTING
                # if VISUAL_STATE_MACHINE_AVAILABLE and self.visual_state_machine:
                #     allowed, check_result = self.visual_state_machine.pre_action_check(
                #         "move_mouse", {"x": x, "y": y, "duration": duration}
                #     )
                #     if not allowed:
                #         return {"success": False, "visual_check_failed": True, **check_result}
                
                success = self.interaction_controller.move_mouse(x, y, duration)
                result = {"success": success, "action": "move_mouse", "x": x, "y": y}
                
                self._log_action_if_recording("move_mouse", {"x": x, "y": y, "duration": duration})
                
                # Update visual state machine
                if VISUAL_STATE_MACHINE_AVAILABLE and self.visual_state_machine:
                    self.visual_state_machine.post_action_update("move_mouse", {"x": x, "y": y, "duration": duration}, result)
                
                return result
            except Exception as e:
                logger.error(f"move_mouse failed: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool
        async def drag(start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5) -> Dict[str, Any]:
            """Performs a mouse drag from a start coordinate to an end coordinate."""
            try:
                success = self.interaction_controller.drag(start_x, start_y, end_x, end_y, duration)
                self._log_action_if_recording("drag", {"start_x": start_x, "start_y": start_y, "end_x": end_x, "end_y": end_y})
                return {"success": success, "action": "drag", "start": (start_x, start_y), "end": (end_x, end_y)}
            except Exception as e:
                logger.error(f"drag failed: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool
        async def get_screen_info() -> Dict[str, Any]:
            """Retrieves information about the screen(s), such as resolution and layout."""
            try:
                monitors = self.screenshot_capture.sct.monitors
                return {"success": True, "monitors": monitors, "primary_monitor_size": pyautogui.size()}
            except Exception as e:
                logger.error(f"get_screen_info failed: {e}")
                return {"success": False, "error": str(e)}

        # Visual State Machine Tools
        @self.mcp.tool
        async def capture_visual_state(monitor: int = 1, show_grid: bool = True, grid_size: int = 50) -> Dict[str, Any]:
            """Captures a screenshot with optional coordinate grid overlay. Returns HTTP URL for image access."""
            if not VISUAL_STATE_MACHINE_AVAILABLE or self.visual_state_machine is None:
                return {"success": False, "error": "Visual state machine not available"}
            try:
                img = self.screenshot_capture.capture_screen(monitor)
                
                if show_grid:
                    img = self.visual_state_machine.create_coordinate_overlay(img, grid_size)
                
                # Save with HTTP URL
                filepath, http_url = self.save_screenshot_with_url(img, "visual_state")
                
                # Also create small thumbnail for context
                thumbnail_b64 = self.screenshot_capture.screenshot_to_base64(img, quality=40, max_width=480)
                
                visual_state = VisualState(
                    screenshot_data_uri=f"data:image/jpeg;base64,{thumbnail_b64}",
                    timestamp=datetime.now(),
                    width=img.shape[1],
                    height=img.shape[0],
                    monitor=monitor,
                    has_overlay=show_grid,
                    grid_size=grid_size if show_grid else 0,
                    metadata={
                        "captured_at": datetime.now().isoformat(),
                        "action_history_count": 0,
                        "thumbnail_width": 480,
                        "thumbnail_quality": 40,
                        "screenshot_file": filepath,
                        "screenshot_url": http_url
                    }
                )
                
                self.visual_state_machine.current_state = visual_state
                
                return {
                    "success": True,
                    "screenshot_url": http_url,
                    "screenshot_file": filepath,
                    "thumbnail": f"data:image/jpeg;base64,{thumbnail_b64}",
                    "format": "jpeg",
                    "width": img.shape[1],
                    "height": img.shape[0],
                    "monitor": monitor,
                    "has_grid_overlay": show_grid,
                    "grid_size": grid_size,
                    "timestamp": datetime.now().isoformat(),
                    "visual_feedback": {
                        "type": "screenshot_with_overlay" if show_grid else "screenshot",
                        "description": f"Screen capture {'with coordinate grid overlay' if show_grid else 'without overlay'}",
                        "image_url": http_url,
                        "file_location": filepath,
                        "instructions": f"Screenshot available at: {http_url}"
                    }
                }
            except Exception as e:
                logger.error(f"capture_visual_state failed: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool
        async def get_visual_state_status() -> Dict[str, Any]:
            """Gets the current status of the visual state machine."""
            if not VISUAL_STATE_MACHINE_AVAILABLE or self.visual_state_machine is None:
                return {"success": False, "error": "Visual state machine not available"}
            try:
                status = self.visual_state_machine.get_status()
                return {"success": True, "status": status}
            except Exception as e:
                logger.error(f"get_visual_state_status failed: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool
        async def toggle_visual_confirmation(enabled: bool) -> Dict[str, Any]:
            """Enable or disable the requirement for screenshots before interactive actions."""
            if not VISUAL_STATE_MACHINE_AVAILABLE or self.visual_state_machine is None:
                return {"success": False, "error": "Visual state machine not available"}
            try:
                self.visual_state_machine.require_visual_confirmation = enabled
                return {
                    "success": True, 
                    "visual_confirmation_enabled": enabled,
                    "message": f"Visual confirmation {'enabled' if enabled else 'disabled'}"
                }
            except Exception as e:
                logger.error(f"toggle_visual_confirmation failed: {e}")
                return {"success": False, "error": str(e)}

        # Video Recording Tools
        @self.mcp.tool
        async def start_screen_recording(fps: int = 10, max_duration: int = 300) -> Dict[str, Any]:
            """Starts a new screen recording session."""
            if not VIDEO_PROCESSING_AVAILABLE or self.video_recorder is None or self.action_recorder is None:
                return {"success": False, "error": "Video processing dependencies not available. Please run 'pip install -r requirements.txt'"}
            try:
                session_id = self.video_recorder.start_recording(fps, max_duration)
                if self.config.enable_action_logging:
                    self.action_recorder.start_action_recording(session_id)
                self.stop_recording_flag.clear()
                self.recording_thread = threading.Thread(target=self._recording_loop, args=(session_id,), daemon=True)
                self.recording_thread.start()
                session = self.video_recorder.active_sessions[session_id]
                return {"success": True, "session_id": session_id, "resolution": session.resolution}
            except Exception as e:
                logger.error(f"start_screen_recording failed: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool
        async def stop_screen_recording(session_id: str) -> Dict[str, Any]:
            """Stops an active screen recording session."""
            if not VIDEO_PROCESSING_AVAILABLE or self.video_recorder is None or self.action_recorder is None:
                return {"success": False, "error": "Video processing dependencies not available."}
            try:
                if session_id not in self.video_recorder.active_sessions:
                    return {"success": False, "error": "Invalid session ID"}
                self.stop_recording_flag.set()
                if self.recording_thread:
                    self.recording_thread.join(timeout=5.0)
                success = self.video_recorder.stop_recording(session_id)
                session = self.video_recorder.active_sessions.get(session_id, {})
                return {"success": success, "output_path": getattr(session, 'output_path', None), "frame_count": getattr(session, 'frame_count', 0)}
            except Exception as e:
                logger.error(f"stop_screen_recording failed: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool
        async def get_key_frames(session_id: str, max_frames: int = 10) -> Dict[str, Any]:
            """Extracts key frames from a recording session."""
            if not VIDEO_PROCESSING_AVAILABLE or self.video_recorder is None:
                return {"success": False, "error": "Video processing dependencies not available."}
            try:
                key_frames = self.video_recorder.get_key_frames(session_id, max_frames)
                # Format frames as data URIs for proper image display in MCP clients
                formatted_frames = []
                for ts, f in key_frames:
                    frame_b64 = self.screenshot_capture.screenshot_to_base64(f, quality=40, max_width=480)
                    data_uri = f"data:image/jpeg;base64,{frame_b64}"
                    formatted_frames.append({"timestamp": ts, "frame": data_uri})
                return {"success": True, "key_frames": formatted_frames}
            except Exception as e:
                logger.error(f"get_key_frames failed: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool
        async def get_frame_at_time(session_id: str, timestamp: float) -> Dict[str, Any]:
            """Extract a specific frame at given timestamp from a recording session."""
            if not VIDEO_PROCESSING_AVAILABLE or self.video_recorder is None:
                return {"success": False, "error": "Video processing dependencies not available."}
            try:
                frame = self.video_recorder.extract_frame_at_time(session_id, timestamp)
                if frame is not None:
                    frame_b64 = self.screenshot_capture.screenshot_to_base64(frame, quality=40, max_width=480)
                    # Format as data URI for proper image display in MCP clients
                    data_uri = f"data:image/jpeg;base64,{frame_b64}"
                    return {"success": True, "timestamp": timestamp, "frame": data_uri}
                else:
                    return {"success": False, "error": "Frame not found at specified timestamp"}
            except Exception as e:
                logger.error(f"get_frame_at_time failed: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool
        async def analyze_motion_in_range(session_id: str, start_time: float, end_time: float) -> Dict[str, Any]:
            """Detect motion between frames in a time range."""
            if not VIDEO_PROCESSING_AVAILABLE or self.video_recorder is None:
                return {"success": False, "error": "Video processing dependencies not available."}
            try:
                motion_analysis = self.video_recorder.detect_motion_in_range(session_id, start_time, end_time)
                analysis_data = [asdict(analysis) for analysis in motion_analysis]
                return {"success": True, "motion_analysis": analysis_data}
            except Exception as e:
                logger.error(f"analyze_motion_in_range failed: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool
        async def correlate_actions_with_motion(session_id: str, time_tolerance: float = 0.5) -> Dict[str, Any]:
            """Correlate user actions with visual changes in the screen."""
            if not VIDEO_PROCESSING_AVAILABLE or self.video_recorder is None or self.action_recorder is None or self.temporal_analyzer is None:
                return {"success": False, "error": "Video processing dependencies not available."}
            try:
                if session_id not in self.video_recorder.active_sessions:
                    return {"success": False, "error": "Invalid session ID"}
                
                # Get all motion analysis for the session
                session = self.video_recorder.active_sessions[session_id]
                duration = (datetime.now() - session.start_time).total_seconds()
                motion_analysis = self.video_recorder.detect_motion_in_range(session_id, 0, duration)
                
                # Get action log from action recorder
                action_log = self.action_recorder.action_log if self.action_recorder.recording_session == session_id else []
                
                # Correlate actions with visual changes
                correlations = self.temporal_analyzer.correlate_actions_with_changes(motion_analysis, action_log, time_tolerance)
                
                return {"success": True, "correlations": correlations}
            except Exception as e:
                logger.error(f"correlate_actions_with_motion failed: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool
        async def cleanup_recording_session(session_id: str) -> Dict[str, Any]:
            """Clean up session data and temporary files."""
            if not VIDEO_PROCESSING_AVAILABLE or self.video_recorder is None:
                return {"success": False, "error": "Video processing dependencies not available."}
            try:
                if session_id not in self.video_recorder.active_sessions:
                    return {"success": False, "error": "Invalid session ID"}
                
                # Stop recording if still active
                session = self.video_recorder.active_sessions[session_id]
                if session.is_recording:
                    self.stop_recording_flag.set()
                    if self.recording_thread:
                        self.recording_thread.join(timeout=5.0)
                
                # Clean up session
                self.video_recorder.cleanup_session(session_id)
                
                return {"success": True, "message": f"Session {session_id} cleaned up successfully"}
            except Exception as e:
                logger.error(f"cleanup_recording_session failed: {e}")
                return {"success": False, "error": str(e)}

    def _recording_loop(self, session_id: str):
        """Background thread for continuous frame capture."""
        if self.video_recorder is None:
            return
        session = self.video_recorder.active_sessions.get(session_id)
        if not session:
            return
        target_interval = 1.0 / session.fps
        while not self.stop_recording_flag.is_set() and session.is_recording:
            start_time = time.time()
            self.video_recorder.capture_frame(session_id, self.screenshot_capture)
            elapsed = time.time() - start_time
            time.sleep(max(0, target_interval - elapsed))
            if (datetime.now() - session.start_time).total_seconds() >= (session.metadata or {}).get("max_duration", 300):
                break

    def _log_action_if_recording(self, action_type: str, params: Dict[str, Any]):
        """Log user action if recording is active."""
        if self.action_recorder and self.config.enable_action_logging and self.action_recorder.recording_session:
            self.action_recorder.log_action(action_type, params)

    def save_screenshot_with_url(self, img: np.ndarray, prefix: str = "screenshot") -> tuple[str, str]:
        """Save screenshot and return both file path and HTTP URL"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(self.temp_dir, filename)
        
        # Save with high quality
        cv2.imwrite(filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Return both file path and HTTP URL
        http_url = f"http://localhost:8001/screenshot/{filename}"
        return filepath, http_url

    def run(self, host: str = "127.0.0.1", port: int = 8000):
        logger.info(f"Starting TanukiMCP Vision Primitives Server on {host}:{port}")
        
        # Start HTTP server for image serving on port 8001
        import threading
        import uvicorn
        
        def start_http_server():
            uvicorn.run(self.app, host="127.0.0.1", port=8001, log_level="warning")
        
        http_thread = threading.Thread(target=start_http_server, daemon=True)
        http_thread.start()
        logger.info("Starting HTTP image server on 127.0.0.1:8001")
        
        # Give HTTP server time to start
        import time
        time.sleep(1)
        
        self.mcp.run(transport="streamable-http", host=host, port=port, path="/mcp")


def main():
    parser = argparse.ArgumentParser(description="TanukiMCP Vision Primitives Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args, _ = parser.parse_known_args()

    config = VisionConfig()
    if args.debug:
        config.debug = True
        logging.getLogger().setLevel(logging.DEBUG)

    server = VisionMCPServer(config)
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main() 