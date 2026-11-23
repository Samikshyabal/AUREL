# AUREL — Knowledge Distillation

**A Deep Learning Project on Teacher–Student Knowledge Transfer using CIFAR-10**

## 1. Project Overview

AUREL is a Knowledge Distillation project that explores how knowledge learned by a larger neural network (Teacher Model) can be transferred to a smaller neural network (Student Model).

The project compares three models:

- **Teacher Model:** Modified ResNet-18
- **Baseline Student:** Custom CNN trained using standard supervised learning
- **Distilled Student:** The same custom CNN trained using Knowledge Distillation

The objective is to study the trade-off between model size, predictive performance and computational efficiency.

## 2. Dataset

The project uses the CIFAR-10 dataset.

- 60,000 colour images
- Image dimensions: 32 × 32 pixels
- 10 object categories
- 50,000 training images
- 10,000 test images

The dataset contains categories such as airplanes, automobiles, birds, cats, dogs, ships and trucks.

## 3. Model Architecture

| Model             | Architecture       | Parameters |
| ----------------- | ------------------ | ---------: |
| Teacher           | Modified ResNet-18 | 11,173,962 |
| Baseline Student  | Custom CNN         |    188,970 |
| Distilled Student | Custom CNN         |    188,970 |

The student models have approximately 98.31% fewer parameters than the teacher.

## 4. Knowledge Distillation Methodology

Knowledge Distillation transfers knowledge from a trained teacher model to a student model.

The student learns through two loss components:

1. **Cross-Entropy Loss:** Learns from the actual ground-truth labels.
2. **Knowledge Distillation Loss:** Learns from the teacher's softened output probabilities.

The training objective is:

`Total Loss = (1 − α) × CE Loss + α × T² × KL Divergence`

Where:

- Temperature (T) = 4
- Alpha (α) = 0.7
- Cross-Entropy weight = 0.3
- Distillation weight = 0.7

## 5. Experimental Results

The following results are from the project's comparative evaluation.

| Metric                   | Teacher | Baseline Student | Distilled Student |
| ------------------------ | ------: | ---------------: | ----------------: |
| Test Accuracy            |  90.38% |           81.52% |            80.88% |
| Test Loss                |  0.2920 |           0.5372 |            0.5900 |
| Best Validation Accuracy |  90.28% |           80.84% |            81.66% |
| Parameters               |  11.17M |          188.97K |           188.97K |

**Key observations:**

- The teacher model achieved the highest test accuracy.
- The distilled student achieved a 0.82 percentage-point improvement in best validation accuracy over the baseline student.
- The baseline student achieved slightly higher test accuracy than the distilled student.
- Both student models are substantially smaller than the teacher model.
- The distilled student did not demonstrate a test-accuracy improvement in this experiment.

_Note: The original teacher training notebook reports 90.30% test accuracy, while the comparative evaluation export reports 90.38%. These are retained as separate source results rather than silently reconciled._

## 6. Application Features

The AUREL Streamlit application provides six research sections:

1. Research Overview
2. Model Intelligence
3. Live Inference Studio
4. Performance Lab
5. Knowledge Transfer Explorer
6. Research Findings

Users can explore model comparisons, upload images for CIFAR-10 classification and examine prediction probabilities from all three models.

## 7. Technologies Used

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Plotly
- Pillow
- CIFAR-10

## 8. Project Structure

```text
AUREL/
├── 01_Data_Teacher/
├── 02_Student_Distillation/
├── 03_Results_Charts/
├── 04_Demo_Report/
├── checkpoints/
├── docs/
├── ui/
├── app.py
├── models.py
├── inference.py
├── preprocessing.py
├── requirements.txt
├── test_inference.py
└── README.md
```

## 9. Installation and Execution

### Prerequisites

- Python 3.10 or later
- Git
- pip

### Clone the repository

```bash
git clone https://github.com/Samikshyabal/AUREL.git
cd AUREL
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 10. Live Application

**AUREL Demo:** https://aurel-knowledge-distillation.streamlit.app/

## 11. Limitations

- The experiment is specific to CIFAR-10 and its ten classes.
- The student models do not outperform the teacher in predictive accuracy.
- The distilled student did not achieve higher test accuracy than the baseline in this experiment.
- Individual prediction confidence should not be interpreted as model accuracy.
- Performance may vary depending on the execution environment and available hardware.

## 12. Project Purpose

This project was developed as an academic exploration of Knowledge Distillation, focusing on model compression, teacher–student learning and comparative performance evaluation.

**AUREL — Knowledge Distillation Project**
