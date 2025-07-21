# 🪨 Mineral-Detection

_Image Color Analysis for Mineral Identification_

---

## 🌟 Overview

**Mineral-Detection** is a comprehensive, interactive web application designed to assist geologists, students, and enthusiasts in analyzing ore/rock images for mineral identification. Leveraging state-of-the-art image color analysis and edge detection algorithms, this project delivers intuitive visualizations and professional PDF reports in a user-friendly interface.

---

## 🧑‍🔬 What Does This Project Do?

This app helps you:
- **Upload photos of rocks/minerals** and analyze their color makeup
- **Detect edges and structures** within the images to highlight mineral boundaries
- **Visualize the results** using interactive pie charts, histograms, and 3D scatter plots
- **Download a detailed PDF report** summarizing the findings with embedded charts and analysis
- **Switch between professional and student modes** for tailored experiences

---

## 🚀 Features

### 🖼️ Image Upload
- Drag and drop or select rock/ore images directly from your device.
- Supports common image  formats (JPG, PNG, etc.).

### 🎨 Color Detection
- Uses advanced clustering techniques (such as **K-Means**) to identify the most prominent colors in your sample.
- Helps in determining the type and concentration of minerals present based on color distribution.

### 🪞 Edge Detection
- Utilizes robust edge detection (OpenCV, Canopy) to outline the physical structure, cracks, and boundaries in rock images.
- Essential for highlighting mineralogical zones and textural features.

### 📊 Visualization Tools
- **Pie Charts:** Display the proportion of detected colors/minerals.
- **Histograms:** Show the color distribution and frequency.
- **3D Scatter Plots:** Visualize color clusters in RGB space for deeper insight.

### 📄 PDF Report Generation
- Download a **professional PDF report** summarizing the analysis.
- Includes all key visuals and detailed explanations.
- Great for field reports, lab work, or educational purposes.

### 🧑‍🏫 Dual Modes: Professional & Student
- **Professional mode:** Clean interface for research and industrial use.
- **Student mode:** Educational interface with tooltips and step-by-step guidance.

---

## 🧰 Technology Stack

| Layer        | Technology                     | Purpose                                 |
|--------------|-------------------------------|-----------------------------------------|
| Frontend     | Streamlit, Flask              | Web interface & alternate student site  |
| Backend      | Python, OpenCV, Canopy        | Image processing & edge detection       |
| Visualization| Matplotlib                    | Charts and graphs                       |
| Reporting    | ReportLab                     | PDF generation                          |

---

## 🏁 Getting Started

### 1. Clone This Repository
```bash
git clone https://github.com/Ankit-03G/Mineral-Detection.git
cd Mineral-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```
- For the student site (if available):  
  ```bash
  python flask_app.py
  ```

### 4. Upload an Image & Start Exploring!
- Click on "Browse files" or drag and drop your rock image into the app.
- Click "Analyze" to begin color and edge detection.
- Explore interactive charts and download your PDF report.

---

## 🧭 Example Workflow

1. **Select or drag a sample rock photo into the app.**
2. **Review the detected color clusters**—see which minerals dominate your sample.
3. **Check the edge visualization**—observe cracks, veins, or other geological features.
4. **Download a PDF  report**—includes all charts, explanations, and your original image.

---

## 🌱 Planned Enhancements

- 🤖 Integrate AI/ML models for automatic rock/mineral classification.
- 🖌️ Add advanced customization for visualizations.
- 📁 Support analyzing multiple images in batch mode.
- 🛡️ Implement enhanced error handling and feedback.
- 🌐 Add export options (CSV, JSON) for further analysis.

---

## 💡 Tips for Best Results

- Use high-resolution, well-lit images for more accurate analysis.
- For batch analysis or research use, prefer the professional mode.
- Check the app interface for sample images and usage guides.

---

## 🤝 Contributing

We welcome contributions!  
- Open issues for bugs and suggestions.
- Fork the repo and submit a pull request for improvements.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

Made with ❤️ by [Ankit-03G](https://github.com/Ankit-03G)
