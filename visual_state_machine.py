#!/usr/bin/env python3
"""
Visual State Machine for TanukiMCP Vision Server
Implements intelligent state management requiring visual feedback before actions.
Similar to Puppeteer's screenshot-before-action approach.
"""

import base64
import time
import tempfile
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np


@dataclass
class VisualState:
    """Represents the current visual state of the screen"""
    screenshot_data_uri: str  # Data URI format for MCP client image display
    timestamp: datetime
    width: int
    height: int
    monitor: int
    has_overlay: bool = False
    grid_size: int = 50
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PendingAction:
    """Represents an action waiting for visual confirmation"""
    action_type: str
    params: Dict[str, Any]
    requires_screenshot: bool = True
    timestamp: datetime = field(default_factory=datetime.now)


class VisualStateMachine:
    """State machine that enforces visual feedback before interactions"""
    
    def __init__(self, screenshot_capture, max_screenshot_age: float = 5.0):
        self.screenshot_capture = screenshot_capture
        self.max_screenshot_age = max_screenshot_age
        self.current_state: Optional[VisualState] = None
        self.pending_actions: List[PendingAction] = []
        self.action_history: List[Dict[str, Any]] = []
        self.require_visual_confirmation = True
        
        # Interactive actions that require screenshots
        self.interactive_actions = {
            'click_coordinate', 'type_text', 'press_keys', 'scroll', 
            'move_mouse', 'drag'
        }
    
    def create_coordinate_overlay(self, img: np.ndarray, grid_size: int = 50) -> np.ndarray:
        """Add coordinate grid overlay to screenshot for visual debugging"""
        overlay_img = img.copy()
        height, width = img.shape[:2]
        
        # Grid color (semi-transparent green)
        grid_color = (0, 255, 0)
        
        # Draw vertical lines
        for x in range(0, width, grid_size):
            cv2.line(overlay_img, (x, 0), (x, height), grid_color, 1)
            # Add coordinate labels every 200 pixels
            if x % 200 == 0 and x > 0:
                cv2.putText(overlay_img, str(x), (x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, grid_color, 1)
        
        # Draw horizontal lines
        for y in range(0, height, grid_size):
            cv2.line(overlay_img, (0, y), (width, y), grid_color, 1)
            # Add coordinate labels every 200 pixels
            if y % 200 == 0 and y > 0:
                cv2.putText(overlay_img, str(y), (5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, grid_color, 1)
        
        # Add crosshair at mouse position (if available)
        # This would require getting current mouse position
        
        return overlay_img
    
    def save_screenshot_to_file(self, img: np.ndarray, filename_prefix: str = "screenshot") -> str:
        """Save screenshot to temporary file and return file path"""
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(tempfile.gettempdir(), "tanukimcp_screenshots")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.jpg"
        filepath = os.path.join(temp_dir, filename)
        
        # Save image
        cv2.imwrite(filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        return filepath

    def capture_visual_state(self, monitor: int = 1, show_grid: bool = True, grid_size: int = 50) -> VisualState:
        """Capture current screen state with optional coordinate overlay"""
        img = self.screenshot_capture.capture_screen(monitor)

        if show_grid:
            img = self.create_coordinate_overlay(img, grid_size)

        # Save screenshot to file for reliable access
        screenshot_file = self.save_screenshot_to_file(img, "visual_state")
        
        # Also create a small thumbnail data URI for preview
        THUMB_MAX_WIDTH = 480  # pixels
        THUMB_QUALITY = 40  # JPEG quality for thumbnail
        thumb_b64 = self.screenshot_capture.screenshot_to_base64(img, quality=THUMB_QUALITY, max_width=THUMB_MAX_WIDTH)

        visual_state = VisualState(
            screenshot_data_uri=f"data:image/jpeg;base64,{thumb_b64}",
            timestamp=datetime.now(),
            width=img.shape[1],
            height=img.shape[0],
            monitor=monitor,
            has_overlay=show_grid,
            grid_size=grid_size if show_grid else 0,
            metadata={
                "captured_at": datetime.now().isoformat(),
                "action_history_count": len(self.action_history),
                "thumbnail_width": THUMB_MAX_WIDTH,
                "thumbnail_quality": THUMB_QUALITY,
                "screenshot_file": screenshot_file  # Add file path to metadata
            }
        )
        
        self.current_state = visual_state
        return visual_state
    
    def is_visual_state_fresh(self) -> bool:
        """Check if current visual state is recent enough for actions"""
        if not self.current_state:
            return False
        
        age = (datetime.now() - self.current_state.timestamp).total_seconds()
        return age <= self.max_screenshot_age
    
    def require_fresh_screenshot(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce screenshot requirement before interactive actions"""
        if not self.require_visual_confirmation:
            return {"allowed": True, "reason": "Visual confirmation disabled"}
        
        if action_type not in self.interactive_actions:
            return {"allowed": True, "reason": "Non-interactive action"}
        
        if not self.is_visual_state_fresh():
            return {
                "allowed": False, 
                "reason": "Fresh screenshot required before action",
                "required_action": "capture_visual_state",
                "max_age_seconds": self.max_screenshot_age,
                "current_age_seconds": (
                    (datetime.now() - self.current_state.timestamp).total_seconds() 
                    if self.current_state else None
                )
            }
        
        return {"allowed": True, "reason": "Visual state is fresh"}
    
    def pre_action_check(self, action_type: str, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Check if action can proceed based on visual state machine rules"""
        check_result = self.require_fresh_screenshot(action_type, params)
        
        if not check_result["allowed"]:
            # Log the blocked action
            self.pending_actions.append(PendingAction(
                action_type=action_type,
                params=params,
                requires_screenshot=True
            ))
            
            return False, {
                "action_blocked": True,
                "block_reason": check_result["reason"],
                "required_action": check_result.get("required_action"),
                "suggestion": f"Call capture_visual_state(show_grid=True) before {action_type}",
                "pending_action": {"type": action_type, "params": params}
            }
        
        return True, {"action_allowed": True}
    
    def post_action_update(self, action_type: str, params: Dict[str, Any], result: Dict[str, Any]):
        """Update state machine after successful action"""
        action_record = {
            "action_type": action_type,
            "params": params,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "visual_state_age": (
                (datetime.now() - self.current_state.timestamp).total_seconds()
                if self.current_state else None
            )
        }
        
        self.action_history.append(action_record)
        
        # Remove any matching pending actions
        self.pending_actions = [
            pa for pa in self.pending_actions 
            if not (pa.action_type == action_type and pa.params == params)
        ]
        
        # For certain actions, we might want to invalidate the current visual state
        # to force a new screenshot for the next action
        if action_type in ['click_coordinate', 'type_text', 'press_keys']:
            # These actions likely change the screen, so visual state becomes stale
            if self.current_state:
                self.current_state.metadata["invalidated_by"] = action_type
    
    def get_status(self) -> Dict[str, Any]:
        """Get current state machine status"""
        return {
            "visual_confirmation_required": self.require_visual_confirmation,
            "current_visual_state": {
                "exists": self.current_state is not None,
                "fresh": self.is_visual_state_fresh(),
                "age_seconds": (
                    (datetime.now() - self.current_state.timestamp).total_seconds()
                    if self.current_state else None
                ),
                "has_overlay": self.current_state.has_overlay if self.current_state else False,
                "resolution": f"{self.current_state.width}x{self.current_state.height}" if self.current_state else None
            },
            "pending_actions": len(self.pending_actions),
            "action_history": len(self.action_history),
            "max_screenshot_age": self.max_screenshot_age,
            "interactive_actions": list(self.interactive_actions)
        } 