# post-process-susc

this lIbrary is built following the default tiled file structures in output of the Annual_Wildfire_Susceptibility python package with the monthly configuration, avaialble at the following github:
https://github.com/GiorgioMeschi/Annual_Wildfire_Susceptibility

That package allows the computation of monthly susceptibility map on a certain domain subdivided in x tiles using ML algorithms. 
This Library can produce automatically several steps, with the option of running them step by step or all together through the built-in run_all() function based on the input settings. 

the main operations are the following:

- aggregate the tiled outcome producing a country map for each month of the selected years, including  required reprojection operations.

- computation of the threshold using a specific methodology and geenration of categorized susceptibility maps

- computation of the fuel type and the monthly fuel maps (12-classes output) following a proven methodology. 

- plotting of the susceptibility and fule maps including some statistics, and merge in a single png

upcoming:

default statistics focused on validating the outcome of the model, omparing fire occurences with the monthly outcome. 


the library includes an option to integrate the workflow directly on operational scripts that compute the current monthly fuel maps, used as input of the newest (2025) version of the fire danger rating system RISICO.


# Pre-requirements

this library uses functions included in the geospatial_tools python package:

python -m pip install --no-cache-dir -U git+https://github.com/GiorgioMeschi/geospatial_tools

# Installation 

python -m pip install --no-cache-dir -U git+https://github.com/GiorgioMeschi/post-process-susc


