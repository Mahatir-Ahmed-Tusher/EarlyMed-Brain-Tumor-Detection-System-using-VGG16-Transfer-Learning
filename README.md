![logo](https://github.com/user-attachments/assets/c76f0457-f570-4551-81e7-feb6a56c8a4d)

# **EarlyMed: Brain Tumor Detection System using VGG16 Transfer Learning**  

### **An AI-powered Brain Tumor Detection System using VGG16 & Transfer Learning**  

---  

## **🚀 Overview**  
Early detection of brain tumors is crucial for effective treatment and improved patient outcomes. **EarlyMed: Brain Tumor Diagnosis** is a deep learning-based classification system that automates **brain MRI scan analysis** to detect and classify brain tumors into multiple categories. 

This project is a **side project of the EarlyMed Initiative**, developed by our team at **VIT-AP University**. The **EarlyMed Initiative** aims to empower individuals with **AI-driven early health awareness**, assisting in preliminary diagnosis before seeking medical consultation.  

🔗 **Live Demo on Hugging Face Spaces**: [EarlyMed-Brain-Tumor-Diagnosis](https://huggingface.co/spaces/schneeubermensch/EarlyMed-Brain-Tumor-Diagnosis)  

---  

## **📌 Features**  
✅ **Deep Learning-Based Classification**: Uses **VGG16 with transfer learning** to classify MRI scans into **Glioma, Meningioma, Pituitary Tumor, or No Tumor**.  
✅ **High Accuracy (~97%)**: Fine-tuned model with optimized performance on test data.  
✅ **Dual Deployment**: 
- **Web-based (Hugging Face)**: Accessible through Gradio UI.  
- **Local Deployment**: Run using Flask for offline use.  
✅ **Interactive UI**: Upload MRI scans and get predictions with confidence scores instantly.  
✅ **Easy-to-Deploy**: All dependencies and requirements are provided.  
✅ **Saved Model Download**: Due to GitHub’s 100MB file limit, the **model.h5 (122 MB)** file has been uploaded to Google Drive. The download link is provided in a text file named **"Saved Model.txt"**.  

---  

## **🖥️ Tech Stack**  
- **Deep Learning**: TensorFlow, Keras, VGG16 (Pretrained CNN)  
- **Backend**: Flask (for local deployment)  
- **Frontend**: HTML, CSS, JavaScript (for Flask version)  
- **Cloud Deployment**: Gradio, Hugging Face Spaces  

---  

## **🧠 Classification & Prediction**  
Our model classifies brain MRI scans into the following **four classes**:  
- **Glioma**  
- **Meningioma**  
- **Pituitary Tumor**  
- **No Tumor**  

### **How It Works:**  
Users can **upload an MRI scan**, and the system will analyze the image using deep learning and predict the tumor type **along with a confidence score (probability %)**.  

![image](https://github.com/user-attachments/assets/e218fe2e-56ca-4fb4-8bc8-dfbacc584c30)


---  

## **📂 Project Structure**  
```
├── For Gradio/              # Files for Gradio Deployment
│   ├── app.py               # Gradio App Script
│   ├── Saved Model.txt      # Google Drive link for model.h5
│   ├── requirements.txt     # Dependencies for Hugging Face Spaces
│
├── templates/               # Frontend Files for Flask Deployment
│   ├── index.html           # Main Web UI
│
├── .ipynb file              # Jupyter Notebook
│
├── app.py                   # Flask Backend Code
├── Saved Model.txt          # Google Drive link for model.h5
├── requirements.txt         # Dependencies for Local Deployment
├── README.md                # Project Documentation
```  

---  

## **🚀 Deployment Guide**  

### **🔹 1. Run on Hugging Face (No Setup Required)**  
Visit: [EarlyMed-Brain-Tumor-Diagnosis](https://huggingface.co/spaces/schneeubermensch/EarlyMed-Brain-Tumor-Diagnosis) and upload your MRI scan to get instant results.  

### **🔹 2. Local Deployment (Flask Backend + HTML Frontend)**  
If you want to run this project locally using Flask, follow these steps:  

#### **Step 1: Clone the Repository**  
```bash
$ git clone https://github.com/Mahatir-Ahmed-Tusher/EarlyMed-Brain-Tumor-Detection-System-using-VGG16-Transfer-Learning.gitt
$ cd EarlyMed-Brain-Tumor-Detection-System-using-VGG16-Transfer-Learning
```

#### **Step 2: Create Virtual Environment (Recommended)**  
```bash
$ python -m venv venv
$ source venv/bin/activate  # For Linux/macOS
$ venv\Scripts\activate     # For Windows
```

#### **Step 3: Install Dependencies**  
```bash
$ pip install -r requirements.txt
```

#### **Step 4: Run Flask Server**  
```bash
$ python app.py
```

#### **Step 5: Open in Browser**  
Go to `http://127.0.0.1:5000/` and upload an MRI image to get predictions.  

---  

## **🩺 Dataset Information**  
### **Methods**  
The application of deep learning in medical diagnostics is revolutionizing healthcare. According to the **World Health Organization (WHO)**, brain tumor diagnosis requires **detection, classification, and tumor location identification**. This project utilizes **CNN-based multi-task classification** to detect and classify brain tumors, rather than using separate models for different tasks. **Tumor segmentation** is also incorporated using a CNN-based model.  

### **Dataset Composition**  
This dataset is a combination of three sources:  
- **Figshare**  
- **SARTAJ dataset**  
- **Br35H**  

It contains **7023 MRI images** categorized into four classes (**Glioma, Meningioma, Pituitary Tumor, and No Tumor**).  
Link to the dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

🔹 *Note:* Images vary in size, so pre-processing includes **resizing and removing extra margins** for better accuracy.  

---  

# **🛠️ Detailed System Architecture**

The **EarlyMed: Brain Tumor Diagnosis** system follows a structured deep learning pipeline, integrating **pretrained models, Flask-based API, and Gradio UI** for seamless deployment and accessibility. The architecture can be divided into the following key stages:

## **1️⃣ Data Preprocessing & Augmentation**
- Raw MRI images are sourced from multiple datasets (Figshare, SARTAJ, Br35H) and undergo preprocessing:
  - **Resizing**: Standardizing dimensions for consistent input.
  - **Normalization**: Scaling pixel values between 0 and 1.
  - **Data Augmentation**: Random transformations (rotation, flipping, zooming) to enhance model generalization.

## **2️⃣ Deep Learning Model (VGG16 with Transfer Learning)**
- **VGG16 Backbone**: Utilized as a feature extractor.
- **Custom Fully Connected Layers**:
  - Flatten → Dense Layers → Softmax for classification.
- **Optimization & Training**:
  - Loss Function: Categorical Crossentropy.
  - Optimizer: Adam.
  - Validation & Testing on separate data.

## **3️⃣ Model Inference & Prediction**
- The trained model is exported as **model.h5**.
- Input: **User uploads an MRI image**.
- Output: **Prediction of tumor type (Glioma, Meningioma, Pituitary, or No Tumor) with confidence score**.

## **4️⃣ Deployment Strategy**
### **🔹 Web-Based Deployment (Gradio on Hugging Face)**
- Interactive UI using **Gradio** for quick testing.
- Hosted on **Hugging Face Spaces** for public accessibility.

### **🔹 Local Deployment (Flask + HTML Frontend)**
- Flask API serves the model locally.
- **Frontend (index.html)**: Simple UI for image upload & result display.
- Users can run it on their machines by cloning the repository.

## **5️⃣ System Workflow**
```
User Uploads MRI → Image Preprocessing → Model Prediction → Output with Confidence Score
```

## **🔹 Summary of Architecture**
| **Stage**              | **Components**                                |
|----------------------|--------------------------------------|
| **Data Handling**   | Figshare, SARTAJ, Br35H Datasets     |
| **Preprocessing**   | Resizing, Normalization, Augmentation |
| **Model**          | VGG16 + Custom Layers (Transfer Learning) |
| **Backend**        | Flask API (Local), Gradio (Cloud)       |
| **Frontend**       | HTML + CSS (Flask) / Gradio UI          |
| **Deployment**     | Hugging Face Spaces, Local Flask Setup |

This system ensures high accuracy, scalability, and ease of deployment for both researchers and general users seeking early tumor diagnosis.

---

## **📊 Model Performance**  
The model achieves high accuracy across different tumor types:  

| **Class Label**   | **Accuracy (%)** | **Precision** | **Recall** | **F1-Score** |
|------------------|----------------|--------------|------------|-------------|
| **Glioma**       | 96.5%          | 0.95         | 0.97       | 0.96        |
| **Meningioma**   | 94.2%          | 0.93         | 0.94       | 0.93        |
| **Pituitary**    | 97.8%          | 0.98         | 0.97       | 0.97        |
| **No Tumor**     | 99.1%          | 0.99         | 0.99       | 0.99        |

---  
## **📞 Contact Information**
**Mahatir Ahmed Tusher**  
B.Tech in Computer Science (AI/ML)  
VIT-AP University  

🔗 **GitHub**: [Mahatir-Ahmed-Tusher](https://github.com/Mahatir-Ahmed-Tusher)  
🔗 **Google Scholar**: [Mahatir Ahmed Tusher](https://scholar.google.com/citations?user=k8hhhx4AAAAJ&hl=en)  
🔗 **LinkedIn**: [Mahatir Ahmed Tusher](https://in.linkedin.com/in/mahatir-ahmed-tusher-5a5524257)  
📧 **Email**: mahatirtusher@gmail.com

## **🙏 Acknowledgement**  

I would like to express my sincere gratitude to **Saket Choudary Kongara** and **Sivamani Vangapalli** for their inspiration and encouragement throughout this project. While I independently handled every aspect of the work—ranging from dataset preprocessing to model development, deployment, and documentation—their support and motivating words kept me driven and focused. 
Thank you for being a source of motivation! 🚀  


🚀 **Empowering Early Diagnosis with AI - EarlyMed Initiative**
