# StreamCam App

**StreamCam App** is a Java-based real-time video stream recognition project.  
The application reads a live video stream, displays it in a desktop window, and detects bicycles using a YOLO model exported to ONNX format.

The project is built with:

- Java 17
- Maven
- JavaCV / FFmpeg
- ONNX Runtime
- YOLO11n
- YouTube / HLS stream support

---

## Features

- Reads video stream from YouTube using `yt-dlp`
- Plays the stream inside a Java desktop window
- Runs YOLO object detection on live frames
- Detects bicycles from the video stream
- Draws bounding boxes around detected bicycles
- Uses separate threads for:
    - video capturing
    - object detection
    - UI rendering
- Optimized to avoid frame queue delay
- Keeps only the latest frame for recognition
- Reduces stream quality for better real-time performance

---

## Project Purpose

The main goal of this project is to test how object recognition can be implemented in Java using a real video stream.

The application is focused on bicycle detection.  
It can be used as a base for:

- traffic monitoring
- smart camera systems
- object detection experiments
- real-time computer vision projects
- Java-based AI stream processing

---

## How It Works

The application works in the following flow:

```text
YouTube URL
    ↓
yt-dlp extracts direct stream URL
    ↓
FFmpegFrameGrabber reads video frames
    ↓
Java displays the stream
    ↓
ONNX Runtime runs YOLO model
    ↓
Bicycles are detected
    ↓
Bounding boxes are drawn on the video