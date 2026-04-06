# Crystal Structure Visualizer (via Stlite)

> An interactive crystal structure visualization tool powered by WebAssembly and Python — runs entirely in your browser.

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue.svg)](https://github.com/DifferentialManifold922)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Live Demo
No installation required. Run directly in your browser (desktop & mobile supported):

https://DifferentialManifold922.github.io/Crystal-structure-visualizer-via-Stlite/webpage.html

---

## Overview
This project leverages **Stlite (Streamlit + WebAssembly via Pyodide)** to bring Python-based physical modeling entirely to the frontend.

It enables real-time, interactive visualization of crystal structures without any backend dependency.

---

## Key Features
- Real-time generation and rendering of common crystal structures:
  - SC (Simple Cubic)
  - BCC (Body-Centered Cubic)
  - FCC (Face-Centered Cubic)
  - HCP (Hexagonal Close-Packed)
  - Perovskite structures
- Interactive visualization of crystal planes via Miller indices $(hkl)$
- Wigner–Seitz cell construction using `scipy.spatial`
- Smooth 3D interaction (zoom, rotate, inspect atoms) powered by `Plotly`

---

## Tech Stack
- **Core Engine**: Stlite (Streamlit + WebAssembly / Pyodide)
- **Language**: Python 3
- **Libraries**:
  - Streamlit (UI framework)
  - NumPy & SciPy (numerical computation & geometry)
  - Plotly (3D visualization)

---

## Local Setup
To run or modify the project locally:

```bash
git clone https://github.com/DifferentialManifold922/Crystal-structure-visualizer-via-Stlite.git
cd Crystal-structure-visualizer-via-Stlite

```
(Optional but recommended) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows
```

Install dependencies and run
```bash
pip install -r requirements.txt
streamlit run webpage.py
```
