# 🧠 Brain Tumor Detection using SVM (Scikit-Learn)

This project is a **Machine Learning model** using **Support Vector Machine (SVM)** to detect **brain tumors** from MRI scans. The model is trained on a dataset of labeled MRI images and predicts whether an MRI scan contains a tumor or not.

---

## 📌 Features
✅ **Machine Learning Classification:** Uses **SVM** for tumor detection.  
✅ **MRI Image Preprocessing:** Converts images to grayscale and resizes them.  
✅ **Model Training & Evaluation:** Splits dataset into **training & testing** sets.  
✅ **Command-Line Image Prediction:** Allows users to test the model on new MRI images.  
✅ **Result Output Storage:** Predictions are saved in the `Output_result/` folder.  

---

## 📂 Dataset Structure
The dataset contains **MRI scans of brains with and without tumors**. The images are categorized into two classes:
- **`no`** → MRI scans of healthy brains (without tumors)
- **`yes`** → MRI scans of brains with tumors

The dataset is stored in the `Dataset/brain_tumor_dataset/` directory.



---

### **2️⃣ How to Train the Model**
📌 **Purpose:** Explain how users can train the model.

**Example:**
```markdown
## 🎯 Train the Model
Run the script to train and evaluate the model:
```bash
python brain_tumor_detection.py


### **3️How to Test a New Image**
📌 **Purpose:** Show how users can classify a single MRI image.

**Example:**
```markdown
## 🏥 Test on a New MRI Image
To classify a single MRI image, use:
```bash
python brain_tumor_detection.py --test_image "Dataset/brain_tumor_dataset/yes/sample.jpg"

---

### **4️ Model Performance & Accuracy**
📌 **Purpose:** Show how well your model performs.

**Example:**
```markdown
## 📊 Model Performance
| **Metric**        | **Value**  |
|-------------------|------------|
| **Accuracy**      | 84.31%     |
| **Precision**     | 85%        |
| **Recall**        | 90%        |
| **F1-Score**      | 88%        |

### 🔍 Classification Report Example
               precision    recall  f1-score   support
          no       0.83      0.75      0.79        20
         yes       0.85      0.90      0.88        31
    accuracy                           0.84        51





