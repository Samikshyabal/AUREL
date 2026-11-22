# AUREL — AI Model Intelligence & Research Environment

**Knowledge Distillation — Teacher and Student Model Evaluation**

AUREL is a local Streamlit application for comparing three trained CIFAR-10 classifiers. It asks a controlled question: **with the same student architecture, does learning from a teacher help more than learning from labels alone?** In the recorded run, distillation improved best validation accuracy, but not test accuracy.

## Run on Windows

Run commands in PowerShell from the project root so Streamlit reads the local configuration. The supplied environment was tested with 64-bit Python 3.13.5. No activation, dataset download, notebook execution, or training is needed.

```powershell
Set-Location 'C:\KD\KD DSWPROJECT'
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 18521 --server.headless true
```

Open **http://127.0.0.1:18521**. Stop your server with `Ctrl+C`. Port 18521 is explicit because 8501 was in use during handover. If you choose another available port, change both the command and the URL. The project configuration binds to loopback, not a public interface.

### Install on a fresh machine

Copy the project **including** `checkpoints`, `ui`, `.streamlit`, and `03_Results_Charts`. Recreate the environment rather than copying `.venv` between machines. In the copied project root, with 64-bit Python 3.13 installed:

```powershell
py -3.13 -m venv .venv
```

For **CPU**, install the matched wheels below. For a compatible **NVIDIA GPU**, use the second command **instead of** the first. CUDA 13.0 wheels were tested on an RTX 3050 Laptop GPU; GPU availability depends on the driver and machine.

```powershell
# CPU option
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
# NVIDIA option (alternative)
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
```

Then install the remaining [runtime requirements](./requirements.txt), check them, and use the launch command above:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

## Use the application

The six sidebar chapters cover the research question, model architectures, live inference, measured performance, the transfer objective, and findings. In **03 · Live Inference Studio**:

1. Upload one PNG, JPEG, or WebP image (up to 10 MB and 16 megapixels). Predictions appear only after successful inference.
2. The preview shows the source image. All models receive the same RGB input; non-native dimensions are resized to 32 × 32 using bilinear interpolation, then normalized as during evaluation.
3. Read each model's predicted class and confidence. The chart and expandable table compare all ten class probabilities. The Studio uses CUDA when available, otherwise CPU. A missing or incompatible checkpoint is an error, not a reason to substitute random weights.

**Confidence is not accuracy.** Confidence is the largest softmax probability for **one image**. A confident prediction can be wrong, especially for an image outside CIFAR-10. **Validation accuracy** is the fraction correct on the validation split, used here to select weights. **Test accuracy** is the fraction correct on a separate labeled test set after selection. Uploading an image does not measure either dataset accuracy or update the weights.

## Dataset and experiment

CIFAR-10 has ten labels: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Its images are 32 × 32 RGB. The notebooks partition the 50,000 training images into **45,000 training / 5,000 validation** (split seed 42) and use the separate **10,000-image test set**. Training uses random crops with padding 4 and random horizontal flips. Validation and test use no random augmentation. After conversion to a tensor, RGB mean is `(0.4914, 0.4822, 0.4465)` and standard deviation is `(0.2470, 0.2435, 0.2616)`.

| Model | Architecture and training | Saved checkpoint |
|---|---|---|
| Teacher | ResNet-18 from scratch; 3 × 3 stride-1 stem, no stem max-pool, four residual stages, global average pool and 512 → 10 classifier. Cross-entropy; SGD, learning rate 0.1, momentum 0.9, weight decay 0.0005; learning rate drops tenfold at milestones 10 and 15. | `checkpoints/teacher_model.pth` |
| Baseline | StudentCNN: three bias-free 3 × 3 convolutions (32, 64, 64 channels), each with batch normalization and ReLU. First two blocks use max-pooling; the third uses adaptive average pooling to 4 × 4. Classifier: 1,024 → 128 → ReLU → dropout 0.2 → 10. Hard-label cross-entropy only. | `checkpoints/student_baseline.pt` |
| Distilled | Identical StudentCNN, with the same optimizer and schedule as the baseline. It receives both the ground-truth label and the frozen teacher's distribution. | `checkpoints/student_kd.pt` |

All three trained for 20 epochs with batch size 128 and checkpoint selection by best validation accuracy. Both students used seed 0, Adam, learning rate 0.001, weight decay 0.0001, and cosine annealing.

**What distillation transfers.** Hard labels name one correct class. Soft targets also convey which other classes the teacher considers plausible. With `p = softmax(logits / T)`, the implemented loss is `(1 − α) × cross_entropy(student_logits, label) + α × T² × KL(p_teacher || p_student)`, with **T = 4, α = 0.7**. This follows the training code, not the reversed KL direction in its Markdown annotation. The teacher is not required to make a distilled-student prediction after training.

## Results and limits

| Recorded metric | Teacher | Baseline student | Distilled student |
|---|---:|---:|---:|
| Best validation accuracy | 90.28% | 80.84% | 81.66% |
| Comparative test accuracy | 90.38% | 81.52% | 80.88% |
| Test cross-entropy loss | 0.2920 | 0.5372 | 0.5900 |
| Parameters | 11,173,962 | 188,970 | 188,970 |
| Historical batch-one forward pass (ms/image) | 3.239 | 0.587 | 0.596 |

The distilled student was **0.82 percentage points higher on best validation accuracy** but **0.64 points lower on test accuracy** than the baseline. Both students use about 1/59.1 as many parameters as the teacher. That reduction comes from the architecture, not distillation.

- **Teacher discrepancy:** [Teacher training](./01_Data_Teacher/Teacher_training.ipynb) twice reports **90.30% test accuracy / 0.2917 loss** after reloading the selected weights. The [Role 3 notebook](<./02_Student_Distillation/Role3_Student_and_Distillation edited.ipynb>) and [accuracy-test export](<./03_Results_Charts/accuracy_test (1).csv>) report **90.38% / 0.2920**. The cause is not established. The table uses the comparative record; the two records are not averaged.
- **Latency:** The displayed export used 20 warm-ups and 100 timed forward passes per model; the export does not identify its device. Other saved runs give different values. These are not measurements of upload processing or current hardware.
- **Evidence:** One student training seed; no supported multi-seed uncertainty or significance claim. The teacher printed all 20 epochs; the students printed epochs 5, 10, 15 and 20. Missing losses are not zeros. The [epoch table](./03_Results_Charts/recorded_epochs.csv) preserves the gaps.
- **Provenance:** Use the exact parameter counts above, not the zero teacher count in the older benchmark CSV. Do not use the placeholder sweep in `role4_outputs.zip` or the older analysis notebook's student scores from a mismatched pooling layer. The full dataset and raw per-image test predictions are not bundled; the app presents recorded results rather than recomputing them.

## Verification and maintenance

From the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_inference
```

For the optional browser check, start AUREL on port 18521 as above and run the following in a **second PowerShell terminal**. The checker uses Microsoft Edge and Playwright; neither is required for model inference. Use a new output directory to avoid overwriting the reference screenshots.

```powershell
Set-Location 'C:\KD\KD DSWPROJECT'
.\.venv\Scripts\python.exe -m pip install playwright==1.65.2
$out = Join-Path $env:TEMP ('aurel-check-' + [guid]::NewGuid())
.\.venv\Scripts\python.exe check_frontend.py --url http://127.0.0.1:18521 --output "$out"
```

The entry point is [app.py](./app.py); [ui](./ui) contains chapter rendering, figures, styles and the distillation diagram. [models.py](./models.py), [preprocessing.py](./preprocessing.py), and [inference.py](./inference.py) define the tested inference path. See the [project log](./docs/PROJECT_LOG.md) for the final audit and [screenshots](./docs/screenshots) for reference views. The Colab notebooks are experiment records, **not startup scripts**; running all cells can train models.
