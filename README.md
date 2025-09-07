# Intelligent Video Environment & Item Detection  

This project uses **YOLOv8** for analyzing a video, detecting objects across frames, and classifying the type of environment (Home, Shop, Office, or Unknown).  

Detected objects are tracked uniquely, counted, and saved into a structured **JSON report**.  

---

## 🚀 Features  
- Object detection using **Ultralytics YOLOv8**  
- Tracks objects across frames (avoids double counting)  
- Classifies environment type based on detected items  
- Exports a detailed `report.json` with:  
  - Environment type  
  - Unique item counts  

---

## 📂 Project Structure  
