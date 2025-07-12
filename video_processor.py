#!/usr/bin/env python3
"""
Video Processing Module for TanukiMCP Vision Server
Handles screen recording, frame extraction, and temporal analysis.
Provides primitive video capabilities that work within MCP transport limitations.
"""

import asyncio
import base64
import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import tempfile
import uuid

# Video processing dependencies
import cv2
import numpy as np
try:
    import ffmpeg
except ImportError:
    # moviepy uses ffmpeg as a backend, but we don't call it directly.
    # The error will be caught when moviepy fails to import or run.
    pass

MOVIEPY_AVAILABLE = True
try:
    from moviepy.editor import VideoFileClip, ImageSequenceClip
except ImportError:
    MOVIEPY_AVAILABLE = False


@dataclass 
class VideoSession:
    """Represents an active video recording session"""
    session_id: str
    start_time: datetime
    fps: int
    resolution: Tuple[int, int]
    output_path: str
    frame_count: int = 0
    is_recording: bool = False
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class FrameAnalysis:
    """Analysis results for a video frame"""
    frame_number: int
    timestamp: float
    motion_detected: bool
    motion_intensity: float
    diff_from_previous: float
    keyframe: bool = False

class VideoRecorder:
    """Handles screen recording with frame-based analysis"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "tanuki_video"
        self.temp_dir.mkdir(exist_ok=True)
        self.active_sessions: Dict[str, VideoSession] = {}
        self.frame_cache: Dict[str, List[np.ndarray]] = {}
        self.motion_threshold = 30.0
        
    def start_recording(self, fps: int = 10, max_duration: int = 300) -> str:
        """Start a new screen recording session"""
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(self.temp_dir / f"recording_{session_id}_{timestamp}")
        
        # Get screen dimensions (assuming single monitor for now)
        try:
            import mss
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                resolution = (monitor['width'], monitor['height'])
        except (ImportError, IndexError):
            # Fallback resolution if mss is not available or no monitors found
            resolution = (1920, 1080)
        
        session = VideoSession(
            session_id=session_id,
            start_time=datetime.now(),
            fps=fps,
            resolution=resolution,
            output_path=output_path,
            metadata={
                "max_duration": max_duration,
                "created_at": datetime.now().isoformat()
            }
        )
        
        self.active_sessions[session_id] = session
        self.frame_cache[session_id] = []
        session.is_recording = True  # Mark session as actively recording
        
        return session_id
    
    def capture_frame(self, session_id: str, screenshot_capture) -> bool:
        """Capture a single frame for the recording session"""
        if session_id not in self.active_sessions:
            return False
            
        session = self.active_sessions[session_id]
        if not session.is_recording:
            return False
            
        try:
            # Capture frame using existing screenshot capability
            frame = screenshot_capture.capture_screen(1)
            self.frame_cache[session_id].append(frame.copy())
            session.frame_count += 1
            
            # Limit cache size to prevent memory issues
            if len(self.frame_cache[session_id]) > 300:  # ~30 seconds at 10fps
                self.frame_cache[session_id].pop(0)
                
            return True
        except Exception as e:
            print(f"Frame capture error: {e}")
            return False
    
    def stop_recording(self, session_id: str) -> bool:
        """Stop recording and save video"""
        if session_id not in self.active_sessions:
            return False
            
        session = self.active_sessions[session_id]
        session.is_recording = False
        
        try:
            # Save frames as video using moviepy
            if MOVIEPY_AVAILABLE and len(self.frame_cache[session_id]) > 0:
                frames = self.frame_cache[session_id]
                clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=session.fps)
                clip.write_videofile(f"{session.output_path}.mp4", verbose=False, logger=None)
                
            elif not MOVIEPY_AVAILABLE:
                print("WARNING: MoviePy is not installed. Cannot save video file. Please run 'pip install moviepy'.")
                return False

            return True
        except Exception as e:
            print(f"Video save error: {e}")
            return False
    
    def extract_frame_at_time(self, session_id: str, timestamp: float) -> Optional[np.ndarray]:
        """Extract a specific frame at given timestamp"""
        if session_id not in self.active_sessions:
            return None
            
        session = self.active_sessions[session_id]
        frame_index = int(timestamp * session.fps)
        
        if 0 <= frame_index < len(self.frame_cache[session_id]):
            return self.frame_cache[session_id][frame_index]
        return None
    
    def detect_motion_in_range(self, session_id: str, start_time: float, end_time: float) -> List[FrameAnalysis]:
        """Detect motion between frames in a time range"""
        if session_id not in self.active_sessions:
            return []
            
        session = self.active_sessions[session_id]
        frames = self.frame_cache[session_id]
        
        start_frame = int(start_time * session.fps)
        end_frame = int(end_time * session.fps)
        
        results = []
        prev_gray = None
        
        for i in range(max(0, start_frame), min(len(frames), end_frame + 1)):
            frame = frames[i]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            motion_detected = False
            motion_intensity = 0.0
            diff_from_previous = 0.0
            
            if prev_gray is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_gray, gray)
                diff_from_previous = float(np.mean(diff.astype(np.float32)))
                
                # Detect motion using threshold
                motion_intensity = diff_from_previous
                motion_detected = motion_intensity > self.motion_threshold
            
            analysis = FrameAnalysis(
                frame_number=i,
                timestamp=i / session.fps,
                motion_detected=motion_detected,
                motion_intensity=motion_intensity,
                diff_from_previous=diff_from_previous,
                keyframe=(i % (session.fps * 5) == 0)  # Keyframe every 5 seconds
            )
            
            results.append(analysis)
            prev_gray = gray
        
        return results
    
    def get_key_frames(self, session_id: str, max_frames: int = 10) -> List[Tuple[float, np.ndarray]]:
        """Extract key frames from the recording session"""
        if session_id not in self.active_sessions:
            return []
            
        session = self.active_sessions[session_id]
        frames = self.frame_cache[session_id]
        
        if len(frames) == 0:
            return []
        
        # Simple keyframe extraction: evenly distributed frames
        indices = np.linspace(0, len(frames) - 1, min(max_frames, len(frames)), dtype=int)
        
        key_frames = []
        for idx in indices:
            timestamp = idx / session.fps
            key_frames.append((timestamp, frames[idx]))
        
        return key_frames
    
    def cleanup_session(self, session_id: str):
        """Clean up session data and temporary files"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Remove video file if exists
            video_path = Path(f"{session.output_path}.mp4")
            if video_path.exists():
                video_path.unlink()
            
            # Clear from memory
            del self.active_sessions[session_id]
            if session_id in self.frame_cache:
                del self.frame_cache[session_id]

class ActionRecorder:
    """Records user actions with timestamps for behavioral analysis"""
    
    def __init__(self):
        self.action_log: List[Dict[str, Any]] = []
        self.recording_session: Optional[str] = None
        self.start_time: Optional[datetime] = None
    
    def start_action_recording(self, session_id: str) -> bool:
        """Start recording user actions"""
        self.recording_session = session_id
        self.start_time = datetime.now()
        self.action_log = []
        return True
    
    def log_action(self, action_type: str, params: Dict[str, Any]) -> bool:
        """Log a user action with timestamp"""
        if not self.recording_session or not self.start_time:
            return False
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        action_entry = {
            "timestamp": elapsed,
            "action_type": action_type,
            "params": params,
            "absolute_time": datetime.now().isoformat()
        }
        
        self.action_log.append(action_entry)
        return True
    
    def stop_action_recording(self) -> List[Dict[str, Any]]:
        """Stop recording and return action log"""
        log = self.action_log.copy()
        self.recording_session = None
        self.start_time = None
        self.action_log = []
        return log
    
    def get_actions_in_timerange(self, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """Get actions that occurred within a specific time range"""
        return [
            action for action in self.action_log 
            if start_time <= action["timestamp"] <= end_time
        ]

class TemporalAnalyzer:
    """Analyzes video data over time to find patterns and changes"""
    
    def __init__(self):
        pass
    
    def find_significant_changes(self, motion_analysis: List[FrameAnalysis], threshold: float = 50.0) -> List[Dict[str, Any]]:
        """Find frames with significant visual changes"""
        changes = []
        
        for analysis in motion_analysis:
            if analysis.motion_intensity > threshold:
                changes.append({
                    "timestamp": analysis.timestamp,
                    "frame_number": analysis.frame_number,
                    "intensity": analysis.motion_intensity,
                    "type": "significant_change"
                })
        
        return changes
    
    def correlate_actions_with_changes(self, motion_analysis: List[FrameAnalysis], 
                                     action_log: List[Dict[str, Any]], 
                                     time_tolerance: float = 0.5) -> List[Dict[str, Any]]:
        """Correlate user actions with visual changes in the screen"""
        correlations = []
        
        for action in action_log:
            action_time = action["timestamp"]
            
            # Find motion events near this action
            nearby_motion = [
                m for m in motion_analysis 
                if abs(m.timestamp - action_time) <= time_tolerance and m.motion_detected
            ]
            
            if nearby_motion:
                correlation = {
                    "action": action,
                    "visual_changes": [
                        {
                            "timestamp": m.timestamp,
                            "intensity": m.motion_intensity,
                            "delay": m.timestamp - action_time
                        }
                        for m in nearby_motion
                    ]
                }
                correlations.append(correlation)
        
        return correlations 