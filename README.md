# 👤 Face Recognition Attendance System

A smart, automated attendance tracking application built with Python and computer vision. 

**Face Recognition Attendance** eliminates the need for manual roll calls or ID card scanning. By leveraging your computer's webcam, this system detects faces in real-time, compares them against a directory of known profiles, and automatically logs the recognized individuals into a daily attendance sheet with accurate timestamps.

## ✨ Features
* **Real-Time Detection:** Captures and processes live video feeds using OpenCV.
* **High-Accuracy Recognition:** Utilizes deep learning-based facial encodings to map and identify facial features.
* **Automated Logging:** Instantly records the names, dates, and times of recognized individuals into a structured CSV file.
* **Dynamic Database:** Easily add new users by simply dropping a clear, named photo of them into the `images/` directory.
* **Visual Feedback:** Displays a live video window with bounding boxes and names overlaid on recognized faces.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** * `opencv-python` (cv2) for image processing and camera control.
  * `face_recognition` (built on dlib) for robust facial mapping.
  * `numpy` for matrix calculations.
  * `pandas` for handling and exporting the attendance data.
  * `datetime` for timestamping.

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed. You will also need a working webcam. *(Note: Installing the `face_recognition` library may require you to have CMake and C++ build tools installed on your system for `dlib` to compile properly).*

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/tausf3472/face_recognition_attendance.git](https://github.com/tausf3472/face_recognition_attendance.git)
