
# Direct Preference Optimization (DPO) for GPT-2 Fine-Tuning

[View Model on Hugging Face](https://huggingface.co/SHN157/dpo-model)

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Training & Evaluation Results](#training--evaluation-results)
- [Model Deployment](#model-deployment)
- [Web Application](#web-application)
- [Sample Web App Testing](#sample-web-app-testing)
- [Conclusion](#conclusion)

---

## Student Information
- **Name**: Soe Htet Naing  
- **ID**: st125166  

---

## Project Overview
This project fine-tunes GPT-2 using **Direct Preference Optimization (DPO)** to align the model's responses with user preferences. The training utilizes **UltraFeedback**, a dataset containing human feedback on model-generated responses.

---

## Dataset
The original dataset contained only a **train split** with **60,917** samples and required preprocessing. The dataset included additional metadata fields such as **ratings and models**, which were not necessary for training. To refine it, only **three key fields** were extracted: `prompt`, `chosen`, and `rejected`. After cleaning, the dataset was split into **train (80%) and test (20%)**, resulting in **48,733** training samples and **12,184** test samples.

---

## Model Details
- **Pretrained Model**: `GPT-2`
- **Training Configuration**:
  - **Device**: CUDA (GPU)
  - **Optimizers Tested**: AdamW, RMSprop
  - **Batch Sizes**: 4, 8
  - **Learning Rates**: `5e-4`, `1e-3`, `2e-3`
  - **Beta Values**: `0.1`, `0.2`, `0.3`
  - **Training Steps**: 500
  - **Evaluation Strategy**: Every 500 steps

---

## Training & Evaluation Results
Performance results of 3 different training configurations are shown below.

![Result of config 1](testing/PerformanceConfig1.png)
*Figure 1: Training results for configuration 1.*

![Result of config 2](testing/PerformanceConfig2.png)
*Figure 2: Training results for configuration 2.*

![Result of config 3](testing/PerformanceConfig3.png)
*Figure 3: Training results for configuration 3.*

- **Best Model Configuration**:
  - **Learning Rate**: `0.002`
  - **Batch Size**: `8`
  - **Optimizer**: AdamW
  - **Beta**: `0.3`
  - **Training Steps**: `500`

---

## Model Deployment
The fine-tuned model was uploaded to **Hugging Face Hub** for easy accessibility. The model was stored locally before being uploaded. If the repository did not exist, it was created. The final deployment involved pushing only the trained model files to the repository to ensure minimal storage usage while maintaining accessibility. The uploaded model is available at:

[**Hugging Face Model Repository**](https://huggingface.co/SHN157/dpo-model)

---

## Web Application
The Streamlit web application allows real-time interaction with the fine-tuned model.

### How to Run
```bash
pip install streamlit transformers torch
streamlit run app.py
```
Open [http://localhost:8501/](http://localhost:8501/) in your browser.

---

## Sample Web App Testing
Below is a screenshot from the testing phase of the web application.

![Testing Web App](testing/Webtesting.png)
*Figure 4: Sample test of the web application.*

---

## Conclusion
This project successfully fine-tuned **GPT-2** using Direct Preference Optimization (DPO). The fine-tuned model aligns better with user preferences, and the **Streamlit web app** enables real-time interaction with the chatbot.
