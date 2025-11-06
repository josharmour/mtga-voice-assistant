#!/usr/bin/env python3
"""
Enhanced screenshot utility for bug reports.
- Detects available screenshot tools (cross-desktop compatible)
- Captures both MTGA and Advisor windows
- Intelligently resizes images for Claude's vision API
- Handles diverse Linux desktop environments
"""

import subprocess
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

# Screenshot tools to try, in order of preference
SCREENSHOT_TOOLS = [
    ("gnome-screenshot", ["-f", "{output}"]),  # GNOME
    ("maim", ["-u", "{output}"]),  # Lightweight (Openbox, i3, etc.)
    ("import", ["-window", "root", "{output}"]),  # ImageMagick (universal)
    ("scrot", ["{output}"]),  # Alternative lightweight
    ("spectacle", ["-b", "-o", "{output}"]),  # KDE
    ("flameshot", ["gui", "-r", "-p", "{output}"]),  # Modern screenshot tool
]

# Using ImgBB for hosting (supports up to 32MB)
# No comment body size limit for ImgBB URLs - can use higher resolution
# Target: high-quality debugging screenshots, reasonable file size
MAX_IMAGE_DIMENSION = 2560  # 2560p for excellent debugging detail
TARGET_SIZE_KB = 200  # Target ~200KB JPEG for ImgBB upload


class ScreenshotCapture:
    """Capture and process screenshots for bug reports."""

    def __init__(self):
        self.available_tool = self._find_screenshot_tool()
        if self.available_tool:
            logger.info(f"Screenshot tool detected: {self.available_tool[0]}")
        else:
            logger.warning("No screenshot tool found (gnome-screenshot, scrot, import, etc.)")

    def _find_screenshot_tool(self) -> Optional[Tuple[str, List[str]]]:
        """Find first available screenshot tool on system."""
        for tool_name, args_template in SCREENSHOT_TOOLS:
            try:
                result = subprocess.run(
                    ["which", tool_name],
                    capture_output=True,
                    timeout=2,
                    text=True
                )
                if result.returncode == 0:
                    logger.debug(f"Found screenshot tool: {tool_name}")
                    return (tool_name, args_template)
            except Exception as e:
                logger.debug(f"Checking for {tool_name}: {e}")
                continue
        return None

    def _find_window_id(self, window_name: str) -> Optional[str]:
        """
        Find window ID by name using xdotool or wmctrl.

        Args:
            window_name: Partial window name to search for

        Returns:
            Window ID or None if not found
        """
        try:
            # Try xdotool first (more reliable)
            result = subprocess.run(
                ["xdotool", "search", "--name", window_name],
                capture_output=True,
                timeout=2,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                window_id = result.stdout.strip().split('\n')[0]  # Get first match
                logger.debug(f"Found window '{window_name}' with ID: {window_id}")
                return window_id
        except Exception as e:
            logger.debug(f"xdotool failed: {e}")

        try:
            # Fallback to wmctrl
            result = subprocess.run(
                ["wmctrl", "-l"],
                capture_output=True,
                timeout=2,
                text=True
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if window_name.lower() in line.lower():
                        parts = line.split()
                        if parts:
                            logger.debug(f"Found window '{window_name}' with ID: {parts[0]}")
                            return parts[0]
        except Exception as e:
            logger.debug(f"wmctrl failed: {e}")

        logger.warning(f"Could not find window '{window_name}'")
        return None

    def capture_window_by_name(self, window_name: str) -> Optional[Path]:
        """
        Capture a specific window by name.

        Args:
            window_name: Partial window name to match (e.g., "MTGA", "Advisor")

        Returns:
            Path to screenshot or None if failed
        """
        if not self.available_tool:
            logger.error("No screenshot tool available")
            return None

        try:
            window_id = self._find_window_id(window_name)
            if not window_id:
                logger.warning(f"Window '{window_name}' not found, capturing full screen")
                return self.capture_screen()

            tool_name, args_template = self.available_tool

            # Try to capture specific window
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                output_path = tmp.name

            try:
                # Use xdotool to capture window by ID
                result = subprocess.run(
                    ["import", "-window", window_id, output_path],
                    capture_output=True,
                    timeout=10,
                    text=True
                )

                if result.returncode == 0 and Path(output_path).exists():
                    logger.info(f"Captured window '{window_name}' to {output_path}")
                    return Path(output_path)
                else:
                    logger.debug(f"Window capture failed, falling back to full screen")
                    Path(output_path).unlink(missing_ok=True)
                    return self.capture_screen()
            except Exception as e:
                logger.debug(f"Window-specific capture failed: {e}, using full screen")
                Path(output_path).unlink(missing_ok=True)
                return self.capture_screen()

        except Exception as e:
            logger.error(f"Failed to capture window '{window_name}': {e}")
            return self.capture_screen()

    def capture_multiple_windows(self, window_names: List[str]) -> Optional[Path]:
        """
        Capture multiple windows and composite them into a single image.
        Windows are arranged side-by-side, scaled to fit within max dimensions.

        Args:
            window_names: List of partial window names to capture (e.g., ["MTGA", "Advisor"])

        Returns:
            Path to composite screenshot or None if failed
        """
        if not self.available_tool:
            logger.error("No screenshot tool available")
            return None

        windows_to_capture = []
        window_images = []

        # Try to find and capture each window
        for window_name in window_names:
            try:
                window_id = self._find_window_id(window_name)
                if window_id:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        output_path = tmp.name

                    result = subprocess.run(
                        ["import", "-window", window_id, output_path],
                        capture_output=True,
                        timeout=10,
                        text=True
                    )

                    if result.returncode == 0 and Path(output_path).exists():
                        try:
                            img = Image.open(output_path)
                            window_images.append((window_name, img, output_path))
                            logger.info(f"Captured window '{window_name}': {img.size}")
                        except Exception as e:
                            logger.warning(f"Could not open captured image for {window_name}: {e}")
                            Path(output_path).unlink(missing_ok=True)
                    else:
                        logger.debug(f"Failed to capture window '{window_name}'")
                        Path(output_path).unlink(missing_ok=True)
                else:
                    logger.debug(f"Window '{window_name}' not found")
            except Exception as e:
                logger.debug(f"Error capturing window '{window_name}': {e}")

        # If we got some windows, composite them
        if window_images:
            try:
                # Calculate composite dimensions (side-by-side layout)
                total_width = sum(img.size[0] for _, img, _ in window_images)
                max_height = max(img.size[1] for _, img, _ in window_images)

                # Check if we need to scale down
                if total_width > MAX_IMAGE_DIMENSION or max_height > MAX_IMAGE_DIMENSION:
                    scale = min(MAX_IMAGE_DIMENSION / total_width, MAX_IMAGE_DIMENSION / max_height)
                    scaled_images = []
                    for name, img, _ in window_images:
                        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                        scaled_images.append((name, img.resize(new_size, Image.Resampling.LANCZOS)))
                    window_images = [(name, img, "") for name, img in scaled_images]
                    total_width = sum(img.size[0] for _, img, _ in window_images)
                    max_height = max(img.size[1] for _, img, _ in window_images)

                # Create composite image
                composite = Image.new('RGB', (total_width, max_height), (255, 255, 255))
                x_offset = 0
                for window_name, img, _ in window_images:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        if img.mode == 'RGBA':
                            bg = Image.new('RGB', img.size, (255, 255, 255))
                            bg.paste(img, mask=img.split()[3])
                            img = bg
                        else:
                            img = img.convert('RGB')
                    composite.paste(img, (x_offset, 0))
                    x_offset += img.size[0]

                # Save composite
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    composite_path = tmp.name

                composite.save(composite_path, "PNG", optimize=True)
                logger.info(f"Created composite screenshot: {composite.size}")

                # Clean up temp files
                for _, _, path in window_images:
                    if path:
                        Path(path).unlink(missing_ok=True)

                return Path(composite_path)

            except Exception as e:
                logger.error(f"Failed to composite windows: {e}")
                # Clean up temp files
                for _, _, path in window_images:
                    if path:
                        Path(path).unlink(missing_ok=True)

        # Fall back to full screen if composite fails
        logger.warning(f"Could not capture all windows, falling back to full screen")
        return self.capture_screen()

    def capture_screen(self) -> Optional[Path]:
        """Capture full screen."""
        if not self.available_tool:
            logger.error("No screenshot tool available")
            return None

        tool_name, args_template = self.available_tool

        try:
            # Create temporary file for screenshot
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                output_path = tmp.name

            # Replace {output} placeholder with actual path
            args = [arg.format(output=output_path) for arg in args_template]

            logger.debug(f"Running: {tool_name} {' '.join(args)}")
            result = subprocess.run(
                [tool_name] + args,
                capture_output=True,
                timeout=10,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"Screenshot tool failed: {result.stderr}")
                return None

            if not Path(output_path).exists():
                logger.error(f"Screenshot file not created at {output_path}")
                return None

            logger.info(f"Screenshot captured: {output_path}")
            return Path(output_path)

        except subprocess.TimeoutExpired:
            logger.error("Screenshot tool timed out")
            return None
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            return None

    def resize_for_claude(self, image_path: Path, quality: int = 80) -> Path:
        """
        Resize and compress image for bug reports.

        Uploaded to ImgBB for external hosting (no GitHub body size limit).
        Target: max 2560 pixels (2.5K), optimized for ~200KB JPEG

        Args:
            image_path: Path to original image
            quality: JPEG quality (1-95, default 80 for excellent quality)

        Returns:
            Path to optimized image
        """
        try:
            img = Image.open(image_path)
            original_size = img.size
            logger.info(f"Original image: {original_size[0]}x{original_size[1]} mode={img.mode}")

            # Convert RGBA to RGB FIRST before any resizing
            if img.mode in ('RGBA', 'PA', 'P', 'L'):
                try:
                    # Create white background
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    # Handle different modes
                    if img.mode == 'P':
                        # Paleted image
                        if 'transparency' in img.info:
                            img.putalpha(img.info['transparency'])
                            rgb_img.paste(img, mask=img.split()[3])
                        else:
                            rgb_img.paste(img)
                    elif img.mode == 'L':
                        # Grayscale
                        rgb_img = img.convert('RGB')
                    else:
                        # RGBA or PA - has alpha channel
                        rgb_img.paste(img, mask=img.split()[-1])
                    img = rgb_img
                    logger.info("Converted image to RGB for JPEG compatibility")
                except Exception as e:
                    logger.warning(f"Could not convert image mode {img.mode}: {e}, converting with convert()")
                    img = img.convert('RGB')

            # Calculate scaling needed
            max_dim = max(img.size)
            if max_dim > MAX_IMAGE_DIMENSION:
                scale = MAX_IMAGE_DIMENSION / max_dim
                new_size = (
                    int(img.size[0] * scale),
                    int(img.size[1] * scale)
                )
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized to: {new_size[0]}x{new_size[1]}")

            # Save optimized version as PNG (lossless, better for debugging)
            output_path = image_path.parent / f"{image_path.stem}_optimized.png"
            if img.mode == 'RGBA':
                # PNG supports transparency, keep it
                img.save(output_path, "PNG", optimize=True)
            else:
                # Convert to RGB if needed, then save as PNG
                if img.mode != 'RGB':
                    logger.warning(f"Image is {img.mode}, converting to RGB before save")
                    img = img.convert('RGB')
                img.save(output_path, "PNG", optimize=True)

            file_size_kb = output_path.stat().st_size / 1024
            logger.info(f"Optimized image: {output_path.name} ({file_size_kb:.1f}KB)")

            return output_path

        except Exception as e:
            logger.error(f"Failed to resize image: {e}")
            return image_path

    def capture_and_optimize(self) -> Optional[Path]:
        """Capture screenshot and optimize for Claude."""
        screenshot = self.capture_screen()
        if not screenshot:
            return None

        try:
            optimized = self.resize_for_claude(screenshot)
            # Clean up original if we created an optimized version
            if optimized != screenshot:
                screenshot.unlink()
            return optimized
        except Exception as e:
            logger.error(f"Failed to optimize screenshot: {e}")
            return screenshot


def take_bug_report_screenshots() -> List[Path]:
    """
    Capture screenshots for bug report.
    Returns list of paths to captured images.
    """
    capturer = ScreenshotCapture()

    if not capturer.available_tool:
        logger.warning("Cannot take screenshots - no screenshot tool available")
        return []

    screenshots = []

    # Capture screen (which includes both MTGA and Advisor windows if visible)
    screenshot = capturer.capture_and_optimize()
    if screenshot:
        screenshots.append(screenshot)
        logger.info(f"Captured screenshot for bug report: {screenshot}")

    return screenshots


if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.DEBUG)

    print("Testing screenshot utility...")
    capturer = ScreenshotCapture()

    if capturer.available_tool:
        tool, args = capturer.available_tool
        print(f"\n✓ Screenshot tool found: {tool}")
        print(f"  Command: {tool} {' '.join(args)}")
    else:
        print("\n✗ No screenshot tool found")
        print("Install one of: gnome-screenshot, scrot, import (ImageMagick), maim, spectacle")
