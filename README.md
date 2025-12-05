# post-process-susc

this lIbrary is built on the default tiled file structures in output of the Annual_Wildfire_Susceptibility python package with the monthly configuration, avaialble at the following github:
https://github.com/GiorgioMeschi/Annual_Wildfire_Susceptibility

the mentioned package allows the computation of monthly susceptibility map on a certain domain subdivided in x tiles using ML algorithms. 
This Library can produce automatically several steps, with the option of running them step by step or all together through the built-in run_all() function based on the input settings. 

the main operations are the following:

- Aggregate the tiled outcome producing a country map for each month of the selected years, including  required reprojection operations.

- Computation of the threshold using a specific methodology and generation of categorized susceptibility maps

- Computation of the fuel type and the monthly fuel maps (12-classes output) following a proven methodology. 

- plotting of the susceptibility and fuel maps including some statistics, and merge in a single png

upcoming:

default statistics focused on validating the outcome of the model, omparing fire occurences with the monthly outcome. 


the library includes an option to integrate the workflow directly on operational scripts that compute the current monthly fuel maps, used as input of the newest (2025) version of the fire danger rating system RISICO.

NOTE: 
the run_all funtion runs a different post processing methodology based on the parameter "four_models".
If True, the susceptibility maps were produces (per tile) with a different ML model per fuel type. 4 fuel types are used, so 4 different Random Forests are trained. The post processing take into account all the separated outcome, categorize and reprojecting before merging them together to get a single monthly map, per month per year. 



# Pre-requirements

this library uses functions included in the geospatial_tools python package:

python -m pip install --no-cache-dir -U git+https://github.com/GiorgioMeschi/geospatial_tools

# Installation 

python -m pip install --no-cache-dir -U git+https://github.com/GiorgioMeschi/post-process-susc


# Example of usage

```python

from post_process_susc import post_process

DATAPATH, VS = '.data', 'v1'

# file structure of tiled suspebptibilities over the AOI, example for tile_1:
# {DATAPATH}/ML/tile_1/susceptibility/v3/2025_5/annual_maps/Annual_susc_2025_5.tif


input = {
        'datapath':             DATAPATH, 
        'vs':                   VS, 
        'years':                list(range(2011,2025)), 
        'months':               list(range(1,13)), 
        'tiles':                os.listdir(f'{DATAPATH}/ML'), 
        'dem_file':             f'{DATAPATH}/raw/dem/dem_100m_32632.tif', 
        'working_crs':          'EPSG:32632', 
        'fire_path':            f'{DATAPATH}/raw/burned_area/incendi_dpc_2007_2023_ita_32632_gt10ha.shp',
        'fires_col':            'date_iso', 
        'veg_path':             f'{DATAPATH}/raw/vegetation/corine_2019_32632.tif', 
        'mapping_path':         f'{DATAPATH}/raw/vegetation/veg_to_ft2.json',
        'settings_plt_susc':    dict(    xboxmin_hist= 0.2,
                                            yboxmin_hist= 0.1,
                                            xboxmin_pie= 0.6,
                                            yboxmin_pie= 0.7,
                                            pixel_to_ha_factor= 1,
                                            normalize_over_y_axis= 25,
                                            ncol=12,
                                            nrow=14
                                            ),
                      
        'cores': 1,
        'four_models':False
        }




pp = post_process.PostProcess(**input) 

pp.run_all()


# example run 1 task, plot susceptiility maps (alternative plot --> colr change for grassland covered areas) the merge png is a single table.

totalba = gpd.read_file(f'{DATAPATH}/raw/burned_area/incendi_dpc_2007_2023_ita_32632_gt10ha.shp').area.sum()/10000  # in ha
for year in range(2011, 2025):
    for month in range(1, 13):
        pp.plot_alternative_susc(totalba, year, month) # this will produce a plot in the whole AOI per month

# in alternative for  1 year:
# pp.plot_alternative_susc(None, 2011, 1)

# merge
pp.merge_all_pngs(susc = True, alternative = 'alternative') # merge susc (otherwise fuel maps) of alternative plots

```




