# Sustainability Intelligence Platform

### An AI-Powered Decision-Support Dashboard for Data Center Sustainability

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code for a comprehensive, real-time dashboard designed to monitor and improve the sustainability of data center operations. This capstone project goes beyond traditional monitoring by integrating an AI Co-Pilot, powered by Google's Gemini Pro, to provide automatic alerts, actionable recommendations, and interactive diagnostics.

---

## Live Demo

**[Link to your deployed Azure application]** `(<- Replace this with your public Azure URL)`

---

## 🌟 Key Features

This platform is a complete, end-to-end decision-support system with a wide range of advanced features:

* **📊 Comprehensive Multi-Tab Dashboard:** A clean, professional UI with seven distinct tabs for a 360-degree view of sustainability performance.
* **🤖 AI Co-Pilot:**
    * **Automatic Alerts:** The system uses a dynamic context engine to detect critical anomalies and generate expert-level recommendations automatically.
    * **Interactive Chat:** A floating chat widget allows operators to have a natural language conversation with the AI to "drill down" into problems and test hypotheses.
* **🔮 Predictive Forecasting:** A time-series model (Simple Exponential Smoothing) forecasts total power consumption for the next three hours, enabling proactive decision-making.
* **🔥 Rack Health Heatmap:** A visual thermal map of the data center floor that instantly highlights "hot spots" and potential equipment failures.
* **🔬 Interactive Correlation Plotter:** A powerful data exploration tool that allows users to plot any two metrics against each other to discover hidden relationships.
* **💡 "What-If" Scenario Simulator:** A strategic planning tool that uses complex mathematical models to calculate the long-term financial and carbon impact of major sustainability initiatives.

---

## 🛠️ Tech Stack & Architecture

The platform is built on a modern, multi-layered architecture designed for real-time data processing and intelligent analysis.

| Layer | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Frontend** | Dash, Plotly, Font Awesome | Building the interactive UI, visualizations, and professional icons. |
| **Backend & Data**| Python, Pandas, NumPy | Core application logic, data generation, and numerical calculations. |
| **Intelligence (AI/ML)** | Google Gemini Pro, Statsmodels | Powering the AI Co-Pilot and the predictive forecasting model. |
| **Deployment** | Microsoft Azure, Git/GitHub | Hosting the live web application and managing the CI/CD workflow. |


### System Architecture Diagram

```mermaid
graph TD
    subgraph "<b><i class='fas fa-database'></i> Data Layer</b>"
        A["fa:fa-file-csv dc_sustainability_data.csv<br/>(Simulated Real-time Feed)"] -->|Every 5 seconds| B
    end
    subgraph "<b><i class='fas fa-desktop'></i> Core Application (Python/Dash)</b>"
        B("fa:fa-cogs Dashboard Engine<br/>(Processes & Aggregates Data)") --> C & D & H
    end
    subgraph "<b><i class='fas fa-brain'></i> Intelligence & Analysis Layer</b>"
        D("fa:fa-exclamation-triangle Alerting Engine<br/>(Checks Data vs. Thresholds)") -- Anomaly Detected! --> E["Dynamic Prompt Engineering"]
        E --> F{"fa:fa-microchip AI Co-Pilot<br/>(Gemini Model)"}
        F -- Initial Recommendation --> G(["fa:fa-check-circle Automated Alert Panel"])
        H["Advanced Analytics Engine"] --> H1 & H2 & H3 & H4
        H1["fa:fa-chart-line Predictive Forecasting"]
        H2["fa:fa-fire-alt Rack Health Heatmap"]
        H3["fa:fa-search-dollar Correlation Plotter"]
        H4["fa:fa-lightbulb Scenario Simulator"]
    end
    subgraph "<b><i class='fas fa-user-cog'></i> User Interaction Layer</b>"
        I("fa:fa-user Data Center Operator") <--> C("fa:fa-window-maximize Multi-Tab UI")
        I --> J("fa:fa-comment-dots Floating Chat Widget") -- Asks follow-up --> K
    end
    subgraph "<b><i class='fas fa-cloud-upload-alt'></i> Deployment Layer</b>"
        DEP1["fa:fa-server Azure App Service"]
        DEP2["fa:fa-github GitHub Actions"]
        DEP3["fa:fa-link Custom Domain"]
    end
    K["Conversational Prompt Engineering<br/>(Appends question & history)"] --> F
    F -- Conversational Reply --> J


