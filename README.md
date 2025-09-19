# Real-time Traffic Analysis and Parking Safety with YOLOv8

## 📌 Project Overview  
This project presents a comprehensive system for **real-time traffic analysis** and **parking safety assessment**, powered by state-of-the-art computer vision algorithms. By leveraging the **YOLOv8 architecture**, the system detects, classifies, tracks, and analyzes vehicles from live video feeds, enabling authorities to monitor road conditions effectively.  

The system also integrates a **parking safety module** that determines whether vehicles are correctly parked in designated areas, empowering traffic enforcement bodies to improve road safety and reduce risks.  

---

## 🎯 Objectives  
- Develop a user-friendly interface for real-time traffic analysis and parking monitoring.  
- Seamlessly integrate **live highway surveillance camera feeds** into the system.  
- Utilize **YOLOv8** for accurate vehicle detection, classification, tracking, and speed estimation.  
- Generate **heatmaps** to visualize lane-wise traffic density over time.  
- Implement a **parking safety module** to detect improperly parked vehicles.  
- Compare performance using tracking algorithms such as **SORT** and **ByteTrack**.  

---

## 🛠️ Methodology  
The project pipeline integrates multiple modules:  
- **Video Capture & Preprocessing** – Incoming video feeds are prepared for analysis.  
- **Object Detection & Tracking** – YOLOv8 detects vehicles, while SORT and ByteTrack ensure robust multi-object tracking.  
- **Vehicle Counting** – Counts vehicles via line-based and polygon-based methods.  
- **Velocity Estimation** – Calculates vehicle speed from tracked motion.  
- **Heatmap Generation** – Visualizes traffic flow and congestion patterns.  
- **Parking Safety Assessment** – Detects correct/incorrect parking using YOLOv8 segmentation and OpenCV-based boundary detection.  

**Dataset Used**:  
- **UA-DETRAC Dataset** (vehicles: car, truck, bus, ambulance).  
- **Custom dataset** for parking detection.  

---

## 📊 Results  
The YOLOv8 model achieved strong performance:  
- **Precision**: 92.9%  
- **Recall**: 93.2%  
- **mAP50**: 95.7%  
- **mAP50-95**: 76.6%  

### Key Features Demonstrated  
- **Line Counter** – Tracks vehicles crossing a predefined line.  
- **Polygon Counter** – Counts vehicles in irregular regions of interest.  
- **Traffic Heatmaps** – Visualize lane congestion and usage trends.  
- **Parking Safety** – Real-time detection of correctly vs. incorrectly parked vehicles.  

---

## ✅ Conclusion  
The system delivers a practical and efficient solution for:  
- Monitoring real-time traffic dynamics.  
- Generating visual insights via heatmaps.  
- Enhancing road safety with parking compliance detection.  

It serves as a powerful tool for traffic authorities and supports the broader vision of **smart cities**.  

