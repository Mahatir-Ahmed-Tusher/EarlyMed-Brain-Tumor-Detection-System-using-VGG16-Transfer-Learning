import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the trained VGG16 model
try:
    model = tf.keras.models.load_model("model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)

# Class labels (ensure these match your training labels)
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Tumor descriptions and recommendations
tumor_info = {
    'pituitary': """
        **Pituitary Tumor**
        - **What it is**: A growth in the pituitary gland at the base of the brain
        - **Effects**: Can affect hormone production and regulation
        - **Common symptoms**: Headaches, vision problems, irregular periods, fatigue
        - **Next steps**: 
          1. Consult an endocrinologist
          2. Get hormone level tests
          3. Schedule regular MRI monitoring
    """,
    'glioma': """
        **Glioma**
        - **What it is**: A tumor that starts in the glial cells of the brain
        - **Effects**: Can affect brain function depending on location and size
        - **Common symptoms**: Headaches, seizures, memory problems, changes in behavior
        - **Next steps**:
          1. Consult a neuro-oncologist immediately
          2. Get a detailed MRI with contrast
          3. Discuss treatment options (surgery, radiation, chemotherapy)
    """,
    'meningioma': """
        **Meningioma**
        - **What it is**: A tumor that forms in the meninges (brain's protective layers)
        - **Effects**: Can press on the brain or spinal cord
        - **Common symptoms**: Headaches, vision problems, hearing loss, memory issues
        - **Next steps**:
          1. Consult a neurosurgeon
          2. Get regular MRI monitoring
          3. Discuss treatment timing and options
    """,
    'notumor': """
        **No Tumor Detected**
        - **What it means**: No visible signs of tumor in the MRI scan
        - **Next steps**:
          1. Continue regular check-ups
          2. Monitor any symptoms
          3. Maintain healthy lifestyle
        
        Remember: If you have persistent symptoms, consult a healthcare provider even with a negative result.
    """
}

def predict_tumor(image_path):
    print("Received image:", image_path)
    try:
        # Open and preprocess the image
        image = Image.open(image_path)
        image = image.resize((128, 128))
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions) * 100
        
        # Get the predicted class and its information
        predicted_class = class_labels[predicted_class_index]
        
        # Format the result with markdown
        result_text = f"""
        ## 📋 Diagnosis Results
        
        **Predicted Condition:** {predicted_class.title()}
        **Confidence Score:** {confidence_score:.2f}%
        
        {tumor_info[predicted_class]}
        """
        
        return result_text
    except Exception as e:
        error_message = f"Error in predict_tumor: {e}"
        print(error_message)
        return error_message

# Create the Gradio Interface with enhanced UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Header with brain emoji
    with gr.Row():
        gr.Markdown(
            """
            # 🧠 EarlyMed Brain Tumor Detection
            ### AI-Powered Early Detection System
            """
        )
    
    # Main description
    gr.Markdown(
        """
        ## 🎯 Quick Analysis, Reliable Results
        Upload your brain MRI scan below for an instant preliminary assessment. Our AI system will analyze 
        the image and provide detailed insights within seconds.
        """
    )
    
    # Image upload and results section
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="filepath",
                label="Upload MRI Scan",
                elem_id="image_upload"
            )
            submit_btn = gr.Button(
                "🔍 Analyze MRI",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=1):
            diagnosis_output = gr.Markdown(
                label="Diagnosis Results",
                value="Results will appear here after analysis..."
            )
    
    submit_btn.click(fn=predict_tumor, inputs=image_input, outputs=diagnosis_output)
    
    # How it works section
    gr.Markdown(
        """
        ## 🔬 How Our AI Works
        
        At EarlyMed, we believe in the power of AI to assist in early health detection. Our **AI-powered Brain Tumor Diagnosis** tool is designed to analyze MRI scans and provide a **preliminary assessment** of potential tumor types—all in just a few seconds. Here's how it works and why you can trust it:
        **🔍 Step-by-Step Process**
        **1️⃣ Upload Your MRI Scan**
        Simply upload a clear MRI scan of the brain. Our system supports standard medical imaging formats for accurate analysis.
        **2️⃣ AI-Powered Image Processing**
        Our deep learning model, trained on thousands of MRI images, processes your scan. It examines patterns and features that indicate the presence of a tumor.
        **3️⃣ Classification & Prediction**
        The AI model classifies the image into one of four categories:
        * ✔️ **Glioma**
        * ✔️ **Meningioma**
        * ✔️ **Pituitary Tumor**
        * ✔️ **No Tumor**
        **4️⃣ Instant Results & Next Steps**
        Your result is displayed within seconds, including detailed information about the condition and recommended next steps.
        ## 🛡️ Why Is It Reliable?
        * ✅ **Trained on Real Medical Data** – Our AI model has learned from thousands of MRI scans
        * ✅ **Deep Learning for Precision** – We use a **Convolutional Neural Network (CNN)** for analysis
        * ✅ **Fast & Accessible** – Get quick initial assessments from anywhere
        * ✅ **Continuous Improvement** – Regular updates based on new medical data
        """
    )
    
    # Disclaimer
    gr.Markdown(
        """
        ---
        ### ⚠️ Important Disclaimer
        We strongly urge users to consult a healthcare professional for appropriate medical 
        guidance after getting the diagnosis. This initiative is developed by our team at VIT-AP University 
        with the goal of empowering individuals to be more aware of their health before visiting a doctor.
        Our mission is to leverage AI for early detection and better healthcare awareness.
        
        *Developed by the team at VIT-AP University*
        """
    )

if __name__ == "__main__":
    demo.launch(share=True)