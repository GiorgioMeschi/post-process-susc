
import os
import logging
import json
import rasterio as rio
import numpy as np
import multiprocessing as mp
from rasterio.windows import Window
from scipy import ndimage
import geopandas as gpd

import matplotlib.pyplot as plt
import seaborn as sns


from geospatial_tools import geotools as gt
from geospatial_tools import FF_tools as ff

import pyproj
pyproj_path = pyproj.datadir.get_data_dir()
import os
os.environ["GTIFF_SRS_SOURCE"] = "EPSG"
os.environ["PROJ_DATA"] = pyproj_path


class PostProcess:
    def __init__(self, datapath: str, vs: str, years: list[int], months: list[int], tiles: list[str], 
                 dem_file: str, working_crs: str, fire_path: str,
                 fires_col: str, veg_path: str, mapping_path: str,
                 settings_plt_susc=dict(    xboxmin_hist= 0,
                                            yboxmin_hist= 0,
                                            xboxmin_pie= 0,
                                            yboxmin_pie= 0.7,
                                            pixel_to_ha_factor= 100,
                                            normalize_over_y_axis= 10,
                                            ncol=12,
                                            nrow=4
                                            ),
                      
                 cores= 25,
                 four_models=False,
                 op = False,
                 custom_fuel_filename = None,
                 smooth = True):

        '''
        Process susceptibility maps and get fuel maps based on the a default tiled files structure.

        Parameters
        ----------
        datapath : path to the main data folder.   
        vs : version of the analysis
        years : list of years to process
        months : list of months to process
        tiles : list of tiles names to process
        dem_file : path to a DEM file to use as reference for reprojection
        working_crs : crs code (eg. 'EPSG:32632') of the analysis
        fire_path : path to the fire shapefile
        fires_col : name of the column in the fire shapefile that contains the date of the fire
        veg_path : path to the vegetation raster file
        mapping_path : path to the json file that contains the mapping between vegetation and fuel types
        settings_plt_susc : dictionary with settings for the plotting functions
        cores : number of cores to use for multiprocessing
        four_models : if True process the four models outputs separately
        op : if True run in operational mode (no merged plots and no eval of thresholds)
        custom_fuel_filename : if provided save the fuel map also in this path

        use run_all() to execute all the processing steps. 

        '''
        self.vs = vs 
        self.datapath = datapath
        self.years = years
        self.months = months
        self.tiles = tiles
        self.dem_file = dem_file
        self.fire_path = fire_path
        self.working_crs = working_crs
        self.fires_col = fires_col
        self.veg_path = veg_path
        self.mapping_path = mapping_path
        self.settings_plt_susc = settings_plt_susc

        self.cores = cores
        self.four_models = four_models
        self.op = op
        self.custom_fuel_filename = custom_fuel_filename
        self.smooth = smooth

        self.R = gt.Raster()
        self.F = ff.FireTools()
        self.I = gt.Imtools()

        # logging to file datapath
        logging.basicConfig(
            filename=os.path.join(self.datapath, 'post_process.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='w'
        )

    
    def remove_tile_borders(self, tile: str, year: int, month: int, num_px_to_remove = 10, cl = ''):

        '''
        remove the borders to the buffered tiles in order to avoid issues when merging tiles
        the parameter cl governs the clipping when four_models is True
        '''

        path = os.path.join(self.datapath, 'ML', tile, 'susceptibility', self.vs, cl, f'{year}_{month}', 'annual_maps', f'Annual_susc_{year}_{month}.tif')
        try:
            with rio.open(path) as src:
                width = src.width
                height = src.height

                # removing pixels per side: 8 for x and y, so 16 in total to subtract to width and height
                removing = num_px_to_remove*2
                window = Window(num_px_to_remove, num_px_to_remove, width - removing, height - removing)

                # Read the data within the window
                data = src.read(1, window=window)
                profile = src.profile

                # Update the profile with new width, height, and transform
                profile.update({
                    'height': data.shape[0],
                    'width': data.shape[1],
                    'transform': src.window_transform(window)
                })

            # Save the clipped tile (overwrite)
            path2 = path.replace('.tif', '_clipped.tif')
            with rio.open(path2, 'w', **profile) as dst:
                dst.write(data, 1)
            
            if os.path.exists(path2):
                os.remove(path)

        except Exception as e:
            logging.info(f'error: {e}')
    
    
    def merge_susc_tiles(self, year: int, month: int, cl = '', clean = False):

        '''
        Merge the susceptibility tiles into a single country map. (for each class cl is four_models is True)
        '''

        outfile = f'{self.datapath}/susceptibility/{self.vs}/{cl}/susc_{year}_{month}.tif'
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        if not os.path.exists(outfile):
            # merge tiles
            files_to_merge = [os.path.join(self.datapath, 'ML', tile, 'susceptibility', self.vs, cl, f'{year}_{month}/annual_maps/Annual_susc_{year}_{month}_clipped.tif')
                            for tile in self.tiles]
            
            files_to_merge = [f for f in files_to_merge if os.path.exists(f)]

            if len(files_to_merge) != 0:
                out = self.R.merge_rasters(outfile, -1, 'last', *files_to_merge)

                # fix nan data (nan to -1)
                with rio.open(out) as src:
                    arr = src.read(1)
                    arr[np.isnan(arr)] = -1
                    out_meta = src.meta.copy()
                    out_meta.update({
                        'compress': 'lzw',
                        'tiled': True,
                        'blockxsize': 256,
                        'blockysize': 256,
                    })
                    with rio.open(out, 'w', **out_meta) as dst:
                        dst.write(arr, 1)
            
        if clean:
            if os.path.exists(outfile):
                os.remove(outfile)

    
    def reproj_merged_susc(self, yr: str, cl = ''):  

        '''
        Reproject the merged susceptibility map to the working CRS
        yr : string with year_month format (ie 2011_1)
        '''

        basep = f'{self.datapath}/susceptibility/{self.vs}/{cl}'
        try:  
            path = os.path.join(basep, f'susc_{yr}.tif')
            outpath = f'{basep}/reproj_susc_{yr}.tif'
            self.R.reproject_raster_as_v2(path, outpath, self.dem_file, self.working_crs, self.working_crs, 'near')
            os.rename(path, f'{basep}/susc_{yr}_raw.tif')
            os.rename(outpath, path)
            os.remove(f'{basep}/susc_{yr}_raw.tif')
        except Exception as e:
            logging.info(f'error in {yr}: {e}')
    
    def eval_thresholds(self, cl = ''):

        '''
        Compute the thresholds based on the burned area and susceptibility values for categorizing the susceptibility
        '''

        allyears = [f"{year}_{month}" for year in self.years for month in self.months]
        home = os.path.dirname(self.datapath)
        fires_p = self.fire_path.split('data')[1]
        # settings passed to the fire tools function optimized for multiple countries
        settings = dict(
            countries = [os.path.basename(home)],
            years = allyears,
            folder_before_country = os.path.dirname(home),
            folder_after_country = os.path.join('data', 'susceptibility', self.vs, cl),
            fires_paths = f'data/{fires_p}',
            name_susc_without_year = 'susc_',
            year_fires_colname = self.fires_col,
            crs = self.working_crs,
            year_in_name = True,
            allow_plot = False
        ) 

        data = self.F.eval_annual_susc_thresholds(**settings)

        # redefine tr for monthly 
        ba_list = data[3]
        high_vals_years = data[1]
        low_vals_years = data[2]

        avg_ba = np.mean(ba_list)
        mask_over_treashold = [1 if ba > avg_ba else 0 for ba in ba_list]

        # select values from high and  low vals
        mask = np.array(mask_over_treashold)
        high_val_over_tr =  high_vals_years[mask == 1]
        low_val_over_tr =  low_vals_years[mask == 1]
        lv2 = np.mean(high_val_over_tr)
        lv1 = np.mean(low_val_over_tr)

        # save
        thresholds = dict(lv1=lv1, lv2=lv2)
        out = os.path.join(self.datapath, 'susceptibility', self.vs, cl, 'thresholds')
        os.makedirs(out, exist_ok=True)
        with open(f'{out}/thresholds.json', 'w') as f:
            json.dump(thresholds, f, indent=4)


        # plot
        # Convert inputs to numpy arrays with consistent dtype
        high_vals_years = np.array(high_vals_years, dtype=float)
        low_vals_years = np.array(low_vals_years, dtype=float)
        high_val_over_tr = np.array(high_val_over_tr, dtype=float)
        low_val_over_tr = np.array(low_val_over_tr, dtype=float)

        # Means for values over threshold
        mean_high_over = np.mean(high_val_over_tr)
        mean_low_over = np.mean(low_val_over_tr)

        # === Plot: Over Threshold Distributions ===
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.histplot(high_val_over_tr, kde=True, stat='density', element='step', fill=True, label='High Values > Avg BA', ax=ax2)
        sns.histplot(low_val_over_tr, kde=True, stat='density', element='step', fill=True, label='Low Values > Avg BA', ax=ax2)
        ax2.axvline(mean_high_over, color='blue', linestyle='--', label=f'Mean High > Avg: {mean_high_over:.2f}')
        ax2.axvline(mean_low_over, color='orange', linestyle='--', label=f'Mean Low > Avg: {mean_low_over:.2f}')
        ax2.set_title("Distribution: High vs Low Values Over Burned Area Threshold")
        ax2.set_xlabel("Susceptibility Value")
        ax2.set_ylabel("Density")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        # save fig
        fig2.savefig(f'{out}/susceptibility_distributions_over_thresholds.png', dpi=100)

        return [lv1, lv2]


    def get_categoric_susc(self, susc_path: str, thresholds: list[float], cl = ''):

        '''
        Categorize the susceptibility map based on the provided thresholds
        '''

        susc_class_oufolder = os.path.join(self.datapath, 'susceptibility', self.vs, 'susc_classified', cl)
        os.makedirs(susc_class_oufolder, exist_ok=True)
        file1 = os.path.basename(susc_path)
        out1 = f'{susc_class_oufolder}/{file1}'

        if not os.path.exists(out1):
            susc = self.R.read_1band(susc_path)
            tr1, tr2 = thresholds
            susc_cl = self.R.categorize_raster(susc, [tr1, tr2], nodata = -1)
            # type int
            susc_cl = susc_cl.astype(np.int8)
            # save
            self.R.save_raster_as(susc_cl, out1, reference_file = susc_path, dtype = np.int8(), nodata = 0)
        

    def get_fuel_type(self):

        '''
        Compute the fuel type map based on the vegetation raster and the mapping file
        '''

        out_ft = f'{self.datapath}/fuel_type_4cl/{self.vs}/ft.tif'
        os.makedirs(os.path.dirname(out_ft), exist_ok=True)
        veg = self.R.read_1band(self.veg_path)
        mapping = json.load(open(self.mapping_path))
        fueltype = self.R.remap_raster(veg, mapping)
        self.R.save_raster_as(fueltype, out_ft, 
                            reference_file = self.dem_file, dtype = np.int8(), nodata = 0)


    def merge_cl_output(self, year: int, month: int, cls: list[str]):

        '''
        Merge the classified susceptibility maps from each class into a single map, for the case four_models is True
        '''

        folder_merged = f'{self.datapath}/susceptibility/{self.vs}/susc_classified'
        outfile = f'{folder_merged}/susc_{year}_{month}.tif'
        out_smooth = f'{folder_merged}/susc_{year}_{month}_smoothed.tif'
        out_repr = f'{folder_merged}/susc_{year}_{month}_reproj.tif'
        # merge tiles
        files_to_merge = [f"{self.datapath}/susceptibility/{self.vs}/susc_classified/{cl}/susc_{year}_{month}.tif"
                        for cl in cls]
        files_to_merge = [f for f in files_to_merge if os.path.exists(f)]

        if len(files_to_merge) != 0:
            out = self.R.merge_rasters(outfile, np.nan, 'max', *files_to_merge)
            
            # fix nan data (nan to -1)
            with rio.open(out) as src:
                arr = src.read(1)
                out_meta = src.meta.copy()
                out_meta.update({
                    'compress': 'lzw',
                    'tiled': True,
                    'blockxsize': 256,
                    'blockysize': 256,
                })
                with rio.open(out, 'w', **out_meta) as dst:
                    dst.write(arr, 1)
                # smooth
                arr = arr.astype(float)
                if self.smooth:
                    arr_smooth = ndimage.generic_filter(arr, np.nanmean, size=3, mode='nearest', cval=0)
                    arr_smooth = np.rint(arr_smooth).astype(arr.dtype)
                else:
                    arr_smooth = arr
                with rio.open(out_smooth, 'w', **out_meta) as dst:
                    dst.write(arr_smooth, 1)
                del arr, arr_smooth

            self.R.reproject_raster_as_v2(out_smooth, out_repr, self.dem_file, 
                                        input_crs=self.working_crs, working_crs = self.working_crs, interpolation = 'near')
            
            os.rename(outfile, f'{folder_merged}/susc_{year}_{month}_raw.tif')
            os.rename(out_repr, outfile)
            os.remove(f'{folder_merged}/susc_{year}_{month}_raw.tif')
            os.remove(out_smooth)



    # get hazards (fuel maps)
    def get_haz(self, year: int, month: int):

        '''
        Compute the fuel map (12 classes) based on the categorized susceptibility map and the fuel type map
        '''

        try:
            folder = f'{self.datapath}/susceptibility/{self.vs}/susc_classified'
            susc_path = f'{folder}/susc_{year}_{month}.tif'
            
            out_hazard_file = os.path.join(self.datapath, 'fuel_maps', self.vs, f'fuel_{year}_{month}.tif')
            os.makedirs(os.path.dirname(out_hazard_file), exist_ok=True)
            ft_path = f'{self.datapath}/fuel_type_4cl/{self.vs}/ft.tif'
            matrix = np.array([ [1, 4, 7, 10],
                                [2, 5, 8, 11],
                                [3, 6, 9, 12]])
            
            susc_cl = self.R.read_1band(susc_path)
            ft = self.R.read_1band(ft_path)

            hazard = self.R.contigency_matrix_on_array(susc_cl, ft, matrix, nodatax = 0, nodatay = 0)
            self.R.save_raster_as(hazard, out_hazard_file, 
                                reference_file = susc_path, dtype = np.int8(), nodata = 0)
            
            if self.custom_fuel_filename is not None:
                self.R.save_raster_as(hazard, self.custom_fuel_filename, 
                                reference_file = susc_path, dtype = np.int8(), nodata = 0)

        except Exception as e:
            logging.info(f'error get_haz for {year}_{month}: {e}\n')


    def plot_susc(self, total_ba, tr1, tr2, year, month):

        '''
        Plot the susceptibility map with the burned area statistics
        '''

        if self.op == True:
            total_ba = None
            allow_hist = False
            allow_fires = False
        else:
            allow_hist = True
            allow_fires = True

        try:
            path = f'{self.datapath}/susceptibility/{self.vs}/susc_classified/susc_{year}_{month}.tif'
            out = os.path.join( os.path.dirname(path), 'PNG')
            os.makedirs(out, exist_ok=True)
            # outputlike = f'{outfolder}/susc_plot_{year}{month}.png'
            # if not os.path.exists(outputlike):
            settings = dict(
                fires_file= self.fire_path,
                fires_col= self.fires_col, # 'finaldate',
                crs= self.working_crs,
                susc_path= path,
                xboxmin_hist= self.settings_plt_susc['xboxmin_hist'],
                yboxmin_hist= self.settings_plt_susc['yboxmin_hist'],
                xboxmin_pie= self.settings_plt_susc['xboxmin_pie'],
                yboxmin_pie= self.settings_plt_susc['yboxmin_pie'],
                threshold1= tr1,
                threshold2= tr2,
                out_folder= out,
                year= year,
                month= month,
                season= False,
                total_ba_period= total_ba,
                susc_nodata= 0,
                pixel_to_ha_factor= self.settings_plt_susc['pixel_to_ha_factor'],
                allow_hist= allow_hist,
                allow_pie= True,
                allow_fires= allow_fires,
                normalize_over_y_axis= self.settings_plt_susc['normalize_over_y_axis'],
                limit_barperc_to_show= 2,
                is_categorical = True
            )

            self.F.plot_susc_with_bars(**settings)
            plt.close('all')
        except Exception as e:
            logging.info(f'no susc map for - {year}_{month}: {e}')
        

    def plot_alternative_susc(self, total_ba, year: int, month: int):

        '''
        Plot an alternative version of the susceptibility map where class 1,2,3 are reclassified to 4,5,6 if fuel type is 1 (grass)
        '''

        try:
            path = f'{self.datapath}/susceptibility/{self.vs}/susc_classified/susc_{year}_{month}.tif'
            ft_path = f'{self.datapath}/fuel_type_4cl/{self.vs}/ft.tif'

            susc = self.R.read_1band(path)
            ft = self.R.read_1band(ft_path)

            # put class 4 5 and 6 when susc is 1 2 3 and ft is 1
            alt_susc = susc.copy()
            mask = (susc == 1) & (ft == 1)
            alt_susc[mask] = 4
            mask = (susc == 2) & (ft == 1)
            alt_susc[mask] = 4
            mask = (susc == 3) & (ft == 1)
            alt_susc[mask] = 4

            # save
            out_alt_susc = f'{self.datapath}/susceptibility/{self.vs}/susc_classified/susc_{year}_{month}_alternative.tif'
            self.R.save_raster_as(alt_susc, out_alt_susc, path, dtype = np.int8(), nodata = 0)

            # plot
            if self.op == True:
                total_ba = None
                allow_hist = False
                allow_fires = False
            else:
                allow_hist = True
                allow_fires = True

        
            out = os.path.join( os.path.dirname(path), 'PNG', 'alternative')
            os.makedirs(out, exist_ok=True)
            # outputlike = f'{outfolder}/susc_plot_{year}{month}.png'
            # if not os.path.exists(outputlike):
            settings = dict(
                fires_file= self.fire_path,
                fires_col= self.fires_col, 
                crs= self.working_crs,
                susc_path= out_alt_susc,
                xboxmin_hist= self.settings_plt_susc['xboxmin_hist'],
                yboxmin_hist= self.settings_plt_susc['yboxmin_hist'],
                xboxmin_pie= self.settings_plt_susc['xboxmin_pie'],
                yboxmin_pie= self.settings_plt_susc['yboxmin_pie'],
                threshold1= 0,
                threshold2= 0,
                out_folder= out,
                year= year,
                month= month,
                season= False,
                total_ba_period= total_ba,
                susc_nodata= 0,
                pixel_to_ha_factor= self.settings_plt_susc['pixel_to_ha_factor'],
                allow_hist= allow_hist,
                allow_pie= True,
                allow_fires= allow_fires,
                normalize_over_y_axis= self.settings_plt_susc['normalize_over_y_axis'],
                limit_barperc_to_show= 2,
                is_categorical = True,
                options = dict( array_classes = [-1,0.1,1.1,2.1,3.1,4.1],
                                array_names = ['No Data', 'Low', 'Medium', 'High', 'Grass mask'],
                                array_colors = ['#0bd1f700','green', 'yellow', 'red', "#5c605f20"], #  "#5be0ad", "#cece75", "#e081a2"
                               )
                )
            

            self.F.plot_susc_with_bars(**settings)
            plt.close('all')
        except Exception as e:
            logging.info(f'no susc map for - {year}_{month}: {e}')



    def plot_haz(self, year: int, month: int):

        '''
        Plot the fuel map with the burned area statistics
        '''

        if self.op == True:
            allow_hist = False
            allow_fires = False
        else:
            allow_hist = True
            allow_fires = True

        try:
            hazard_file = f'{self.datapath}/fuel_maps/{self.vs}/fuel_{year}_{month}.tif'
            os.makedirs(os.path.dirname(hazard_file), exist_ok=True)
            with rio.open(hazard_file) as haz:
                haz_arr = haz.read(1)
                haz_ndoata = haz.nodata
                unique, counts = np.unique(haz_arr, return_counts=True)
                total_pixels = np.where(haz_arr==haz_ndoata, 0, 1).sum()
                percentages = {int(k): int((v / total_pixels) * 100) for k, v in zip(unique, counts) if k != haz_ndoata}
            with open(f'{os.path.dirname(hazard_file)}/fuel_percentage_{year}_{month}.csv', 'w') as f:
                f.write('Fuel_Class,Percentage\n')
                for k, v in percentages.items():
                    f.write(f'{k},{v}\n')

            out = os.path.join( os.path.dirname(hazard_file), 'PNG')
            os.makedirs(out, exist_ok=True)

            settings = dict(
                fires_file=         self.fire_path,
                fires_col=          self.fires_col,
                crs=                self.working_crs,
                hazard_path=        hazard_file,
                xboxmin_hist= self.settings_plt_susc['xboxmin_hist'],
                yboxmin_hist= self.settings_plt_susc['yboxmin_hist'],
                xboxmin_pie= self.settings_plt_susc['xboxmin_pie'],
                yboxmin_pie= self.settings_plt_susc['yboxmin_pie'],
                out_folder=         out,
                year=               year,
                month=              month,
                season=             False,
                haz_nodata=         0,
                pixel_to_ha_factor= self.settings_plt_susc['pixel_to_ha_factor'],
                allow_hist=         allow_hist,
                allow_pie=          True,
                allow_fires=        allow_fires,
                show_compressed_legend = True
            )

            self.F.plot_haz_with_bars(**settings)
                

        except Exception as e:
            logging.info(f'no haz map for - {year}_{month}: {e}')



    def merge_all_pngs(self, susc = True, alternative = ''):

        '''
        Merge all the susceptibility or fuel png plots into a single image
        '''

        yearmonths = [f"{year}{month}" for year in self.years for month in self.months]
        name = 'susc' if susc else 'fuel'
        year_filenames = [f'{name}_plot_{yrm}' for yrm in yearmonths]
        if susc:
            basep = f'{self.datapath}/susceptibility/{self.vs}/susc_classified/PNG/{alternative}'
        else:
            basep = f'{self.datapath}/fuel_maps/{self.vs}/PNG/{alternative}'

        year_files = [f"{basep}/{filename}.png" for filename in year_filenames]
        year_files = [f for f in year_files if os.path.exists(f)]
        

        fig = self.I.merge_images(year_files, 
                                  ncol=self.settings_plt_susc['ncol'], 
                                  nrow=self.settings_plt_susc['nrow'])
        # save image (Image object)
        out = f'{basep}/MERGED'
        os.makedirs(out, exist_ok=True)
        fig.save(f"{out}/{name}_plot_merged.png")



    def run_all(self):

        '''
        Run all the post-processing steps based on the initialized parameters.
        The methodology depends on the four_models and op parameters.
        '''
        
        if self.four_models:
            cls = ['1','2','3','4']
            
            for cl in cls:
                logging.info(f'Processing class {cl}...')
                    
                logging.info('Removing tile borders...')
                with mp.Pool(self.cores) as pool:
                    pool.starmap(self.remove_tile_borders, [(tile, year, month, 10, cl) 
                                                            for tile in self.tiles for year in self.years for month in self.months])

                logging.info('Merging susc tiles...')
                with mp.Pool(self.cores) as pool:
                    pool.starmap(self.merge_susc_tiles, [(year, month, cl, False) 
                                                        for year in self.years for month in self.months])
                
                
                logging.info('Reprojecting merged susc...')
                yearmonths = [f'{year}_{month}' for year in self.years for month in self.months]
                with mp.Pool(processes=self.cores//2) as pool:
                    pool.starmap(self.reproj_merged_susc, [(yr, cl) for yr in yearmonths])

                if not self.op:
                    logging.info('Evaluating thresholds...')
                    thresholds = self.eval_thresholds(cl)
                else:
                    thresholds_d = json.load(open(f'{self.datapath}/susceptibility/{self.vs}/{cl}/thresholds/thresholds.json'))
                    thresholds = [thresholds_d['lv1'], thresholds_d['lv2']]

                logging.info('Getting categoric susc...')
                folder_susc = f'{self.datapath}/susceptibility/{self.vs}/{cl}' 
                susc_names = [i for i in os.listdir(folder_susc) if i.endswith('.tif') and not i.endswith('raw.tif')]
                with mp.Pool(processes=self.cores) as pool:
                    pool.starmap(self.get_categoric_susc, [(os.path.join(folder_susc, susc_name), thresholds, cl) 
                                                            for susc_name in susc_names])
                
        
            logging.info('Merging class outputs (all the cls)')
            # merge susc of different cl outputs
            with mp.Pool(processes=self.cores//2) as pool:
                pool.starmap(self.merge_cl_output, [(year, month, cls) for year in self.years for month in self.months])

            
        else:
            
            # insert some checks of all the susc tiles are present, if not stop the process.

            logging
            with mp.Pool(self.cores) as pool:
                pool.starmap(self.remove_tile_borders, [(tile, year, month) 
                                                        for tile in self.tiles for year in self.years for month in self.months])

            logging.info('Merging susc tiles...')
            with mp.Pool(self.cores) as pool:
                pool.starmap(self.merge_susc_tiles, [(year, month) 
                                                    for year in self.years for month in self.months])

            logging.info('Reprojecting merged susc...')    
            yearmonths = [f'{year}_{month}' for year in self.years for month in self.months]
            for yr in yearmonths:
                self.reproj_merged_susc(yr)
            
            if not self.op:
                logging.info('Evaluating thresholds...')
                thresholds = self.eval_thresholds()
            else:
                thresholds_d = json.load(open(f'{self.datapath}/susceptibility/{self.vs}/thresholds/thresholds.json'))
                thresholds = [thresholds_d['lv1'], thresholds_d['lv2']]

            logging.info('Getting categoric susc...')
            folder_susc = f'{self.datapath}/susceptibility/{self.vs}' #/cl
            susc_names = [i for i in os.listdir(folder_susc) if i.endswith('.tif') and not i.endswith('raw.tif')]
            with mp.Pool(processes=self.cores) as pool:
                pool.starmap(self.get_categoric_susc, [(os.path.join(folder_susc, susc_name), thresholds) 
                                                        for susc_name in susc_names])
            
        
        # same operation for cl and not
        logging.info('--------------common op----------------')

        if not os.path.exists(f'{self.datapath}/fuel_type_4cl/{self.vs}/ft.tif'):
            logging.info('Getting fuel type...')
            self.get_fuel_type()

        logging.info('Getting hazard maps...')
        with mp.Pool(processes=self.cores) as pool:
            pool.starmap(self.get_haz, [(year, month) 
                                        for year in self.years for month in self.months])
        
        
        logging.info('Plotting susc maps...')
        total_ba = gpd.read_file(self.fire_path).area.sum() / 10000
        with mp.Pool(processes=self.cores) as pool:
            pool.starmap(self.plot_susc, [(total_ba, thresholds[0], thresholds[1], year, month) 
                                        for year in self.years for month in self.months])


        
        # modify this with operational run, the name of the hazard to same as addition.    
        logging.info('Plotting hazard maps...')
        with mp.Pool(processes=self.cores) as pool:
            pool.starmap(self.plot_haz, [(year, month) 
                                        for year in self.years for month in self.months])
        
        if not self.op:
            logging.info('Merging all PNGs...')
            self.merge_all_pngs(susc = True)
            self.merge_all_pngs(susc = False) # fuel maps

        

            
        
            

                
            

            
                
            
            




