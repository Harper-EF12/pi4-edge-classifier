# Edge image classifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harper-EF12/pi4-edge-classifier/blob/main/train_classifier.ipynb)

Train an image classifier in Google Colab, then run it on a Raspberry Pi 4
(64-bit Raspberry Pi OS Trixie) with a USB webcam.

| File | Where it runs | What it does |
|---|---|---|
| `train_classifier.ipynb` | Google Colab | Trains the model and exports `.tflite` files |
| `classify.py` | Raspberry Pi | Classifies webcam frames live |
| `capture_images.py` | Raspberry Pi | Takes your own training images, one folder per class |
| `requirements.txt` | Raspberry Pi | Python packages for the venv |

**Tip:** do everything on the Pi, including Colab in the Pi's web browser.
Then files never need copying between computers: uploads come from the Pi's
folders and downloads land in the Pi's `~/Downloads`.

---

## Part 1: Set up the Pi (once)

Install the system packages (OpenCV with GUI support comes from apt):

```bash
sudo apt update
sudo apt install -y python3-opencv python3-venv
```

Download this repository into a folder called `classifier` in your home folder:

```bash
git clone https://github.com/Harper-EF12/pi4-edge-classifier.git ~/classifier
cd ~/classifier
```

To get any later updates to the scripts, run `git pull` in that folder.

Create a virtual environment and install the requirements. `--system-site-packages` lets the
venv see apt's OpenCV and numpy, while pip installs everything else inside the venv only:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Plug in the USB webcam and check it is detected:

```bash
ls /dev/video*
```

**Every time you open a new terminal**, go back into the folder and activate the venv:

```bash
cd ~/classifier && source .venv/bin/activate
```

Your prompt starts with `(.venv)` when it is active.

---

## Part 2: Train the example model (cats and dogs)

1. Click the **Open in Colab** badge at the top of this page, and sign in with a Google account.
2. **File → Save a copy in Drive**, so you have your own copy to run and change.
3. **Runtime → Change runtime type → T4 GPU** (optional, but faster).
4. Run the cells in order with **Shift + Enter**. In section 2, run **Option A** and skip Option B.
5. The last cell downloads `pi_model.zip`. On the Pi it goes to `~/Downloads`. Unzip it into the project folder:

```bash
cd ~/classifier && unzip -o ~/Downloads/pi_model.zip
```

(If you used another computer, copy `pi_model.zip` into `~/classifier` instead, then `unzip -o pi_model.zip`.)

---

## Part 3: Run the classifier

From a terminal on the Pi's desktop, with the venv active:

```bash
python classify.py
```

A window shows the camera feed with the predicted label, confidence and inference time.
Below 70% confidence the label shows as **unsure**, in orange. Press **q** to quit.

Other options:

```bash
python classify.py --model model_float.tflite   # compare with the float model
python classify.py --threshold 0.9              # stricter "unsure" cut-off
python classify.py --image photo.jpg            # classify one image file
python classify.py --camera 1                   # use /dev/video1 instead
python classify.py --headless                   # no window (e.g. over SSH)
```

**Try this:** point the cat/dog model at your keyboard. It still says "cat" or "dog",
because those are the only answers it knows. The fix is a **background** class, which you
can add when you train with your own images.

---

## Part 4: Train with your own images

### 4.1 Capture the images

Choose your classes, and **always include a `background` class**: the empty scene,
plus anything that isn't one of your other classes. Then run:

```bash
python capture_images.py keyboard mouse background
```

- Press **1**, **2**, **3** to save an image to that class. Hold the key down to save a burst.
- Only the area inside the **green square** is saved. Keep the object inside it.
- Press **q** when you've finished. The script creates **`my_images.zip`** in `~/classifier`.

For good results:

- Aim for **at least 100 images per class**, roughly the same number for each.
- Move the object around and change its angle, distance and background between shots.
- Vary the lighting, and take pictures where the classifier will actually be used.
- Running the script again **adds** to the existing images. To start a class again, delete its
  folder, e.g. `rm -r my_images/mouse`.

### 4.2 Upload the images to Colab

1. Open your copy of the notebook in Colab (Google Drive → **Colab Notebooks**) and run section 1.
2. In section 2, **skip Option A**. Run the **Option B** cell.
3. A **Choose Files** button appears under the cell. Click it.
4. Go to your home folder → **classifier**, select **`my_images.zip`** and click **Open**.
5. Wait for the upload to reach 100%. The cell then lists each class and how many images it has.

Check the list is what you expect before carrying on. Each class name should appear,
with no extra folders.

### 4.3 Train, download and run

Run the rest of the notebook exactly as before. Download and unzip `pi_model.zip` as in
Part 2. The new `labels.txt` replaces the old one, so `classify.py` uses your classes automatically:

```bash
cd ~/classifier && unzip -o ~/Downloads/pi_model.zip
python classify.py
```

---

## Troubleshooting

| Problem | Likely cause |
|---|---|
| `ModuleNotFoundError: cv2` | venv created without `--system-site-packages`, or `python3-opencv` not installed |
| `ModuleNotFoundError: ai_edge_litert` | venv not activated, or `pip install -r requirements.txt` not run |
| `Could not open camera 0` | Webcam not plugged in, or another script is still using it, or it is `/dev/video1` |
| Window doesn't appear | Script started over SSH. Run it from the Pi's own desktop, or use `--headless` |
| `Model has N outputs but labels.txt has M labels` | Model and `labels.txt` came from different downloads. Unzip `pi_model.zip` again with `-o` |
| Option B says it couldn't find class folders | The zip doesn't contain one folder per class. Use the zip made by `capture_images.py` |
| A class you didn't expect appears in Option B | An old folder inside `my_images`. Delete it on the Pi and press q in the capture script again to remake the zip |
| Always predicts the same class | Very different lighting or background from the training images, or too few images |
| Confident but wrong when nothing is there | No `background` class. Capture one and retrain |

To check which numpy is in use (it should be apt's, under `/usr/lib/python3/dist-packages`):

```bash
python -c "import numpy; print(numpy.__version__, numpy.__file__)"
```
