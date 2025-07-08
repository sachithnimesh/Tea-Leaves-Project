
# 🍃 Tea Leaf Disease Classification System

This repository contains a complete deep learning solution for classifying tea leaf diseases using TensorFlow and deploying the model to a mobile app. The system can identify four classes of tea leaves:

- **Healthy**
- **Tea leaf blight**
- **Tea red leaf spot**
- **Tea red scab**

---

## 📁 Project Structure

```

├── DataForTest/                  # Sample test images for evaluation
│   ├── 360\_F\_713030279\_...jpg
│   ├── Tea-leaf-scab\_1.jpg
│   └── algal-leaf-spot-...jpg

├── Mobile App/                   # Mobile app resources
│   ├── Code For Apk/Describe-It-App/
│   └── SS of App UI/             # Screenshots of the app UI

├── labels.txt                    # Class labels in order used for predictions
├── TF2TFLITE.py                  # Script to convert .h5 model to TensorFlow Lite (.tflite)
├── final\_test\_model.py          # Model evaluation script on test images
├── .gitignore                   # Files/folders to ignore in GitHub repo

````

---

## 🧠 Model Description

The deep learning model is trained using **InceptionV3** with transfer learning and data augmentation techniques. It is optimized using **Keras Tuner** and fine-tuned for better generalization.

---

## ✅ Requirements

Install the following Python packages:

```bash
pip install tensorflow keras matplotlib numpy scikit-learn opencv-python
````

---

## 🚀 How to Use

### 🔍 1. Evaluate on Test Images

Run `final_test_model.py` to test the trained model on images in `DataForTest`.

```bash
python final_test_model.py
```

### 🔄 2. Convert Model to TFLite

Use `TF2TFLITE.py` to convert the `.h5` model to `.tflite` for mobile use.

```bash
python TF2TFLITE.py
```

This will generate a `.tflite` model file usable in Android/iOS apps.

---

## 📱 Mobile App Integration

* The converted model (`model.tflite`) and `labels.txt` are used in the **Describe-It App** inside the `Mobile App/Code For Apk` directory.
* The app allows users to **capture/upload an image of a tea leaf**, detect the disease, and display treatment suggestions.

---

## 🖼️ UI Preview

Screenshots of the mobile app interface are available in:

```
Mobile App/SS of App UI/
```

---

## 📂 labels.txt Format

The order in this file should exactly match the order of output neurons in the model:

```
Healthy
Tea leaf blight
Tea red leaf spot
Tea red scab
```

---

## ✨ Future Improvements

* Real-time camera integration with edge detection
* Model optimization with quantization
* Auto-suggestion for treatments based on disease class

---

## 📄 License

This project is for academic and research use. Contact the repository owner for commercial licensing.

---

## 🙋‍♂️ Author

**Sachith Nimesh**
Computer Science | AI & Data Science Researcher

```
