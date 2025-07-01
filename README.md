# GeoPyWeb

<div align="center">

![GeoPyWeb Logo](https://img.shields.io/badge/GeoPyWeb-Advanced%20Geochemical%20Analysis-blue?style=for-the-badge)

**A powerful web-based platform for geochemical data analysis and visualization**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation) • [Contributing](#contributing)

</div>

## 🌍 Overview

GeoPyWeb is a comprehensive web-based platform for geochemical data analysis and visualization. Built as a web version of the popular GeoPyTool, it provides geoscientists with an intuitive interface to upload, analyze, and visualize geochemical datasets using industry-standard diagrams and classification methods.

## ✨ Features

### 📊 Classification Diagrams
- **TAS Diagram** - Total Alkali-Silica classification for volcanic rocks (Wilson et al., 1989)
- **QAPF Diagram** - Modal classification for plutonic rocks with CIPW norm support

### 📈 Variation Diagrams  
- **Harker Diagrams** - Multi-element variation plots against SiO2

### 🔬 Trace Element Analysis
- **REE Patterns** - Rare Earth Element spider diagrams with multiple normalization standards
  - C1 Chondrite (Sun & McDonough, 1989; Taylor & McLennan, 1985; Haskin et al., 1966; Nakamura, 1977)
  - MORB (Sun & McDonough, 1989)
  - UCC (Rudnick & Gao, 2003)
- **Trace Element Spider Diagrams** - Comprehensive incompatible element patterns
  - 6 normalization standards (PM, OIB, EMORB, C1, NMORB, UCC)
  - 2 element sequences (Cs-Lu 36 elements, Rb-Lu 26 elements)
  - Automatic K2O→K and TiO2→Ti conversions

### 🏔️ Tectonic Discrimination
- **Pearce Diagrams** - Granite tectonic setting discrimination (Pearce et al., 1984)
  - 4 sub-diagrams: Y+Nb vs Rb, Yb+Ta vs Rb, Y vs Nb, Yb vs Ta
  - Discriminates syn-COLG, VAG, WPG, and ORG granites

### 🧮 Norm Calculations
- **CIPW Norm** - Cross, Iddings, Pirsson, Washington normative mineral calculations
- Automatic integration with QAPF diagrams

### 🎨 Advanced Features
- **Smart Column Recognition** - Automatic standardization of geochemical column names
- **Flexible Data Input** - Support for CSV and Excel files
- **Color-coded Samples** - Intelligent sample grouping and visualization
- **Multiple Export Formats** - PNG and SVG output options
- **Professional Styling** - Publication-ready diagrams with proper referencing

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/GeoPyTool/GeoPyWeb.git
   cd GeoPyWeb
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python run.py
   ```

4. **Access the web interface**
   Open your browser and navigate to `http://localhost:5000`

### Docker Installation (Optional)

```bash
# Build the Docker image
docker build -t geopyweb .

# Run the container
docker run -p 5000:5000 geopyweb
```

## 📖 Usage

### 1. Data Upload
- Upload your geochemical data in CSV or Excel format
- Supported file extensions: `.csv`, `.xlsx`, `.xls`
- Maximum file size: 16MB

### 2. Data Format Requirements

#### Major Elements (weight %)
- **Required for TAS**: SiO2, Na2O, K2O
- **Required for CIPW/QAPF**: SiO2, TiO2, Al2O3, Fe2O3, FeO, MnO, MgO, CaO, Na2O, K2O, P2O5
- **Required for Harker**: SiO2 + any other major elements

#### Trace Elements (ppm)
- **REE Elements**: La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu
- **Pearce Elements**: Rb, Y, Nb, Yb, Ta
- **Full Trace Suite**: Cs, Tl, Rb, Ba, W, Th, U, Nb, Ta, K, La, Ce, Pb, Pr, Mo, Sr, P, Nd, F, Sm, Zr, Hf, Eu, Sn, Sb, Ti, Gd, Tb, Dy, Li, Y, Ho, Er, Tm, Yb, Lu

#### Optional Columns
- **Sample Grouping**: Color, Type, Label, Group, Category, Class, Formation, Unit, Lithology
- **Sample Information**: Age, Location, Description

### 3. Column Name Recognition

GeoPyWeb automatically recognizes and standardizes various column naming conventions:

```
Examples of recognized formats:
- SiO2, SIO2, sio2, SiO2(wt%), SiO2_wt%, sio2_weight%
- La, LA, la, La(ppm), la_ppm, La_ppb
- K2O → K conversion, TiO2 → Ti conversion (automatic)
```

### 4. Analysis Workflow

1. **Upload Data** → Automatic column standardization and preview
2. **Select Analysis Type** → Choose from available diagram types
3. **Configure Options** → Set normalization standards, element sequences, output format
4. **Generate Diagram** → View results with professional styling
5. **Download Results** → Export as PNG or SVG

## 🔧 Technical Stack

- **Backend**: Flask 2.3.3 (Python web framework)
- **Data Processing**: Pandas 2.0.3, NumPy 1.24.3
- **Visualization**: Matplotlib 3.7.2
- **File Handling**: OpenPyXL 3.1.2 (Excel support)
- **Frontend**: Bootstrap 5.3, HTML5, JavaScript ES6
- **Security**: Werkzeug 2.3.7 (secure file uploads)

## 📁 Project Structure

```
GeoPyWeb/
├── run.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/            
│   └── index.html        # Web interface template
├── uploads/              # Temporary file storage
├── Geochemistry.csv      # Sample dataset
├── README.md             # This file
└── LICENSE               # MIT License
```

## 🔬 Scientific References

The implemented methods are based on established geochemical literature:

- **TAS Classification**: Wilson, M. et al. (1989) *Journal of Petrology*
- **Pearce Diagrams**: Pearce, J.A. et al. (1984) *Journal of Petrology*, v.25, p.956-983
- **REE Normalization**: Sun, S.S. & McDonough, W.F. (1989); Rudnick, R.L. & Gao, S. (2003)
- **QAPF Classification**: Maitre, R.W.L. et al. (2004) *Cambridge University Press*

## 🤝 Contributing

We welcome contributions to GeoPyWeb! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run in development mode
export FLASK_ENV=development
python run.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original GeoPyTool development team
- The global geochemical community for standardized methods
- Contributors and users who provide feedback and improvements

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/GeoPyTool/GeoPyWeb/issues)
- **Documentation**: [Wiki](https://github.com/GeoPyTool/GeoPyWeb/wiki)
- **Community**: [Discussions](https://github.com/GeoPyTool/GeoPyWeb/discussions)

---

<div align="center">

**Made with ❤️ for the geoscience community**

</div>
