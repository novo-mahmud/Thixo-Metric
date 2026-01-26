
# Thixo-Metric v1.0
### Quantitative Geotechnical Stability Analysis

**Thixo-Metric** is an automated decision-support tool designed for the analysis of thixotropic strength recovery in remolded riverbank clays. It is specifically tailored for the Bangladesh Delta, providing engineers and researchers with data-driven "Wait-Time" analysis for safe post-disturbance construction.

---

## System Preview

![Thixo-Metric Dashboard](dashboard_preview.png)

*Figure: Interactive dashboard showing Recovery Curves, Sensitivity Analysis, and Spatial Risk Profiles.*

---

## Features

*   **Interactive Dashboard:** Real-time visualization of Factor of Safety (FoS), Recovery Curves, and Spatial Risk Profiles.
*   **Sensitivity Analysis:** Comparative analysis of "Baseline" vs. "Flood" scenarios to quantify Hydraulic Lag.
*   **Strategic Decision Support:** 
    *   Calculates Reach Failure Rate.
    *   Identifies Top 5 High-Risk Boreholes.
    *   Determines optimal construction wait-time based on 95% confidence intervals.
*   **Dynamic Reporting:** Automated generation of comprehensive Technical Reports (PDF) including tables, embedded charts, and vulnerability summaries.
*   **Granular Exports:** Downloadable options for CSV data, combined dashboards, or individual high-res graphs.

---

## Installation / Usage

This tool is designed to run seamlessly in **Google Colab**. 

1.  Click on "Open in Colab" button below (or upload the `.ipynb` file manually).
2.  Ensure the following libraries are installed (they are pre-installed in Colab):
    ```bash
    pip install pandas numpy matplotlib seaborn ipywidgets fpdf openpyxl
    ```

**[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/Thixo-Metric/blob/main/Thixo-Metric.ipynb)**

---

## Input Data Format

To use **Thixo-Metric**, you must upload a `.csv` or `.xlsx` file containing the following columns:

| Column Name | Description | Units |
| :--- | :--- | :--- |
| `Sample_ID` | Unique Borehole Identifier | Text (e.g., BH-001) |
| `Depth_m` | Depth of sample from surface | Meters (m) |
| `Undisturbed_Su` | Original Undisturbed Shear Strength | kPa |
| `Remolded_Su_S0` | Shear Strength at t=0 (Remolded) | kPa |
| `PI` | Plasticity Index | % |
| `Water_Content` | Natural Water Content | % |
| `Liquid_Limit_LL` | Liquid Limit | % |
| `is_submerged` | Hydrological Condition (Dry/Submerged) | Boolean (True/False) |

*Note: The system will automatically calculate `Soil_Type` (CH/CL) if not provided.*

---

## Geotechnical Logic & Model

The system utilizes the following log-linear recovery model to predict undrained shear strength over time:

 $$ S_u(t) = S_0 + [A \cdot \log_{10}(t)] $$ 
**Parameters:**
*   **$S_u(t)$**: Undrained shear strength at time $t$.
*   **$A$ (Recovery Constant):** Dynamically adjusted based on:
    *   **Soil Type:** Base $A$ is higher for CH (High Plasticity) clays than CL.
    *   **Liquidity Index (LI):** Penalizes recovery rates for highly liquid soils.
    *   **Submergence:** Applies a -20% penalty to submerged samples.
*   **Cap:** Strength is capped at 75% of Undisturbed $S_u$ to remain conservative.

**Strategic Metrics:**
*   **Hydraulic Lag:** The calculated time delay for submerged soil to reach the same Factor of Safety as dry soil.
*   **Wait-Time:** The minimum number of days required to ensure 95% of the borehole reach meets the Target FoS.

---

## Workflow

1.  **Upload:** Select your field data CSV/Excel file.
2.  **Analyze:** Click "Run Analysis" to process the data and render the dashboard.
3.  **Review:** Adjust the "Days Since Disturbance" slider to visualize recovery progress.
4.  **Generate:** Click "Generate Technical Report" to create the PDF bundle.
5.  **Download:** Download the PDF, CSV data, or individual graphs for your report.

---

## Roadmap & Updates

### Current Release
*   **Version:** v1.0.0
*   **Status:** Official Launch
*   **Details:** Includes PDF reporting, Sensitivity Analysis, and Interactive Dashboard.
*   See the [Official Release Page](../../releases) for the asset bundle.

### Feedback & Issues
If you encounter bugs, have feature requests, or find data discrepancies, please create an **[Issue](../../issues)**. We use the Issues tab to track development priorities and engage with the community.

---

## Citation

If you use this tool for your research, thesis, or engineering work, please cite:

> Thixo-Metric v1.0: A Time-Dependent Stability Framework for Deltaic Clays. (2023)

---

## Author & Contact

**Md. Mahmudul Hasan Novo**  

**Email:** [novomahmud@gmail.com](mailto:novomahmud@gmail.com)  

**LinkedIn:** [Connect with Author](https://www.linkedin.com/in/novomahmud/)  

**Affiliation:** BUET (Bangladesh University of Engineering and Technology)  

**Research Focus:** Hydro-Geotechnical Decision Support Systems for Riverbank Stability
