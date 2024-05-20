# Purpose
This program, written in R Quarto, is intended to predict concentrations of molecules in an ELISA (enzyme-linked immunosorbent assay) by optical density (OD). The predictions are dependent on a standard curve to inform the linear model. 

# Basic Functionality
- The data is read in, then subsetted into the standard curve information and the experimental data
- Columns are renamed for the standard curve and sample
- The correaltion coefficient is calculated for strenght of curve
- Fit the curve and build a linear model basd on the standard curve
- Construct a dataframe to feed the predictor function, then predict the concentrations of antibodies present based on the OD
- Turn the predictions into a dataframe
- Perform a one-way ANOVA with Tukey's post-hoc
- Extract the significant relationships from the TPH
- Plot the TPH intervals for visual inspection
- Plot the concentrations by treatment group

# Plots Produced

<img src="images/SC.png" alt="Optical Density Standard Curve with LM Fit" width="400"/>  

<img src="images/ELISA_TPH.png" alt="Optical Density Standard Curve with LM Fit" width="400"/>



<img src="images/ELISA_Expression.png" alt="Optical Density Standard Curve with LM Fit" width="400"/>  

<img src="images/ELISA_ExpressionII.png" alt="Optical Density Standard Curve with LM Fit" width="400"/>


