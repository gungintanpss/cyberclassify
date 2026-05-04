# CyberClassify

Intelligent tool fot detecting and classifying cyberbullying content, built with Streamlit.

---

---

## ⚙️ Setup & Installation

### Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
deactivate                      # keluar venv
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the app
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 🗂️ Pages

| Page | Description |
|------|-------------|
| **Home** | Landing page with a "Get Started" button |
| **About** | Explains the platform and the 5 classification labels |
| **Cyberbullying Classification** | Analyze single text or classify a CSV file |

---

## 🏷️ Classification Labels

| Label | Description |
|-------|-------------|
| `not_cyberbullying` | Content that does not contain cyberbullying |
| `age` | Cyberbullying based on age discrimination |
| `gender` | Cyberbullying based on gender discrimination |
| `ethnicity` | Cyberbullying based on ethnic discrimination |
| `religion` | Cyberbullying based on religious discrimination |

---
