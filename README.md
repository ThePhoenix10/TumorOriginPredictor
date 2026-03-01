# TumorOriginPredictor
A clinically validated, cloud-deployed machine learning platform for tissue-of-origin identification in Cancer of Unknown Primary (CUP) using somatic mutation profiles.

---

## Overview
TumorOriginPredictor is a mutation-only machine learning framework designed to infer the most likely tissue of origin for cancers of unknown primary (CUP) using routinely collected targeted next-generation sequencing (NGS) data.

Unlike feature-intensive deep learning systems requiring multi-omic integration, this platform operates exclusively on somatic mutation profiles derived from standard clinical gene panels. The model has been externally validated using an independent precision oncology cohort and deployed as a real-time web application.

---

## Key Features
- Mutation-only classification framework
- Trained on a large, well-curated tumor genomics dataset
- Independent external clinical validation
- Cloud deployment for real-time inference
- Transparent probability outputs (Top-3 predictions)
- No requirement for additional tissue sampling
- No need for RNA-seq or multi-omic data
- Designed for translational clinical applicability

---

## Installation
### Clone the Repository
```bash
git clone https://github.com/ThePhoenix10/TumorOriginPredictor.git
cd TumorOriginPredictor
```
### System Requirements
- Python 3.8+
- pip
- (Optional) VS Code or other IDE

### Create Virtual Environment (Recommended)
```bash
python -m venv venv
```
**Activate:**

Mac/Linux:
```bash
source venv/bin/activate
```
Windows:
```bash
venv\Scripts\activate
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
Dependencies include: NumPy, pandas, scikit-learn, SciPy, Matplotlib, Flask

---

## Data
### Training Data
- Large curated tumor genomics dataset: [MSK-IMPACT 2017](https://www.cbioportal.org/study/summary?id=msk_impact_2017)
- Multi-cancer classification
- 338-gene clinical sequencing panel

### Validation
- Independent Gundersen Precision Oncology Cohort
- 770 de-identified patient samples

---

## Documentation
For detailed methodology, architecture, and validation workflow:
[Documentation PDF](https://drive.google.com/drive/u/1/folders/1RiH87W9Q7v8xfnbydSQOoxlSi-xjc7WP)

---

## Web Application
The live web application can be accessed at: [https://tumor-origin.com/](https://tumor-origin.com/)

This platform enables real-time tissue-of-origin prediction using somatic mutation profiles.

---

## Authors
- Saicharan Vellanki
- Paraic Kenny, PhD — Gundersen Medical Foundation

## Contact
- saivellanki10@gmail.com
- pakenny@emplifyhealth.org

---

## License
```
MIT License

Copyright (c) 2026 Saicharan Vellanki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
