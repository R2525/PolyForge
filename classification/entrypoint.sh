#!/bin/bash
# Start Xvfb (Virtual Framebuffer)
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
# Wait a moment for Xvfb to start
sleep 1
# Execute the Python script passed as arguments
python hybrid_pipeline.py "$@"
