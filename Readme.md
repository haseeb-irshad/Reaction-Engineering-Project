# Reaction Engineering Modelling
This repository contains my completed work for a **Reaction Engineering Project** undertaken under the guidance of the University of Cambridge. The project involved running simulations, analysing reactor behaviour, performing kinetic parameter identification and applying optimisation and surrogate modelling techniques using Python and Jupyter notebooks.  
The modelling framework was provided as part of the coursework; my contribution focused on understanding the system behaviour, running simulations, interpreting results and completing the analytical tasks.

---
## 📌 Tasks Completed

### **Batch Reactor Simulation (0_toy.ipynb)**
- Explored a simple batch reaction to understand model structure and Jupyter workflows.  
- Investigated how temperature, rate constants and initial concentrations influence concentration–time profiles.  
- Performed basic parameter calibration to match ground truth kinetics.

### **SNAr Plug Flow Reactor Modelling & Calibration (1_SNAr.ipynb)**
- Ran simulations of a multistep SNAr reaction inside a plug flow reactor.  
- Studied how temperature, reactant concentration and residence time affect product formation and side reactions.  
- Identified which kinetic parameters are unidentifiable based on which species were measured experimentally.  
- Performed parameter calibration using the **L-BFGS-B** optimisation method.  
- Generated Latin Hypercube samples and explored 3D product concentration landscapes.

### **Three-Phase Hydrogenation Modelling (2_hydrogenation.ipynb)**
- Modelled a heterogeneous reaction involving solid, liquid and gas phases with mass transfer limitations.  
- Applied two-film theory, Henry’s law and fitted saturation models for reactants.  
- Performed a sensitivity analysis to determine the rate-limiting step.  
- Calibrated kinetic and transport parameters using **Differential Evolution**.  
- Compared alternative mechanistic models and evaluated their predictive performance.

### **Optimisation & Surrogate Modelling**
- Used **Latin Hypercube Sampling** to efficiently explore multidimensional parameter spaces.  
- Built **Gaussian Process surrogate models** and applied **Bayesian Optimisation** to identify favourable operating conditions.  
- Compared surrogate predictions against the full mechanistic model.

---
## 🛠️ Technologies Used
- **Python**: NumPy, pandas, SciPy, scikit-learn  
- **Jupyter Notebook**  
- **Plotly** for interactive visualisation  
- **SciPy Optimisation Tools**: L-BFGS-B, Differential Evolution  
- **Gaussian Process Regression** for surrogate modelling  

---
## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/haseeb-irshad/Reaction-Engineering-Project.git
cd Reaction-Engineering-Project
```
2. Create and activate a Python environment:
```bash
conda create -n reaction-eng python=3.10
conda activate reaction-eng
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Open any `.ipynb` file in VS Code or JupyterLab and run the cells sequentially.
