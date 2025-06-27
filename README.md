# 🧬 3D Cell Annotation GUI

This GUI is designed to support **manual annotation of cells in 3D grayscale microscopy images**. It allows users to draw, edit, and manage regions of interest (ROIs) corresponding to individual cells across image volumes. Whether you're working with raw microscopy data or analyzing heatmap outputs from a neural network, this tool is optimized for **dense, efficient 3D cell annotation**.

---

## Installation
```git clone https://github.com/postnubilaphoebus/cell_gui.git``` <br>
```cd [insert_repo_location_on_your_machine] ``` <br>
```[optionally create and activate a new environment]``` <br>
```pip install -r requirements.txt```

---

## Run

``` python gui.py ```

---

## ✨ Features

- **Multi-tab support**  
  Load multiple image volumes side-by-side (e.g., raw image + neural net output or raw image + HCR channel). Each tab manages its own image and cell mask independently.

- **Synchronized 3D views**  
  Navigate through **axial, sagittal, and coronal** planes. Zoom and view positions are synchronized across all views and tabs.

- **Flexible file I/O**  
  Save and reload your work in multiple file formats such as `.tif`. Annotations are stored per tab.

- **User-friendly interface**  
  Tool shortcuts are displayed on buttons for easy reference.

---

## 🛠️ Main Functions

### 📂 File Loading
- Drag and drop an image or mask anywhere in the window to load it instantly,  <br>
  or use the File button (top left) to browse and open files manually.

### 🖊️ Annotate (`A`)
- Draw ROIs for cells in your chosen view (coronal, axial, or sagittal).
- The color of your drawing indicates the current cell index.
- Designed for dense annotation: only adds to unoccupied pixels.
- Switch between cell indices using left/right buttons.

### 🧽 Eraser (`E`)
- Erase parts of the currently active cell only.
- Functions like the brush tool, but in reverse.

### 🔢 Cell Index Navigation
- Use the left and right buttons to decrease or increase the currently active cell index.
- You can only move to a higher index after painting with your current index.
- Indexing starts at 1; the background is always saved as 0 in your mask.

### 📋 Copy & Paste Points (Right-Click)
- Copy points in one slice and paste them in the next using right-click
- Right-click once to copy points in the current slice.
- Right-click again in a neighboring slice to paste — great for quickly labeling similar-looking cells across slices.

### 🎯 Find Cell (`C`)
- Automatically navigates to the **median location** of the current cell.
- Helpful for resuming annotation after breaks.

### 🎭 Toggle Masks (`M`)
- Show or hide ROI masks to inspect the grayscale image without overlays.

### 🧭 View Finder (`V`)
- Displays crosshair lines in each view showing the current position in the other two views.
- Toggle on/off for better visibility.

### 🔎 Select Cell (`S`)
- Select a specific cell to edit.
- Required to draw/erase on an existing cell.
- Toggle off to resume creating new cells (resets to the highest index).

### ❌ Delete Cell (`D`)
- Click on a cell to delete it entirely once this mode is activated.

---

## 🧠 Notes
- Ensure all images loaded in a tab have **the same dimensions**. Concurrent viewing only works for equally sized volumes.
- Annotations are stored separately for each tab and must be saved individually.

---
