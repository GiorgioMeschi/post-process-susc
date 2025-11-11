
import os
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

class PostProcess:
    def __init__(self, datapath, vs, years, months, tiles, 
                 dem_file, working_crs, fire_path,
                 fires_col, veg_path, mapping_path,
                 settings_plt_susc=dict(    xboxmin_hist= 0,
                                            yboxmin_hist= 0,
                                            xboxmin_pie= 0,
                                            yboxmin_pie= 0.7,
                                            pixel_to_ha_factor= 100,
                                            normalize_over_y_axis= 10,
                                            ncol=12,
                                            nrow=4
                                            ),
                      
                 cores = 25,
                 four_models=False):

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
        self.R = gt.Raster()
        self.F = ff.FireTools()
        self.I = gt.ImageTools()

    
    def remove_tile_borders(self, tile, year, month, num_px_to_remove = 10, cl = ''):

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
            print(f'error: {e}')
    
    
    def merge_susc_tiles(self, year, month, cl = '', clean = False):

        print(f"{year}-{month:02d}...")
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

    
    def reproj_merged_susc(self, yr, cl = ''):  

        basep = f'{self.datapath}/susceptibility/{self.vs}/{cl}'
        try:  
            path = os.path.join(basep, f'susc_{yr}.tif')
            outpath = f'{basep}/reproj_susc_{yr}.tif'
            self.R.reproject_raster_as_v2(path, outpath, self.dem_file, self.working_crs, self.working_crs, 'near')
            os.rename(path, f'{basep}/susc_{yr}_raw.tif')
            os.rename(outpath, path)
            os.remove(f'{basep}/susc_{yr}_raw.tif')
        except Exception as e:
            print(f'error in {yr}: {e}')
    
    def eval_thresholds(self, cl = ''):

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


    def get_categoric_susc(self, susc_path, thresholds: list[float], cl = ''):
        susc = self.R.read_1band(susc_path)
        tr1, tr2 = thresholds
        susc_cl = self.R.categorize_raster(susc, [tr1, tr2], nodata = -1)
        # smooth with average filter 
        susc_cl_float = susc_cl.astype(float)
        # put -1 to nan for smoothing
        susc_cl_float[susc_cl_float == -1] = np.nan
        susc_cl_smoothed = ndimage.generic_filter(susc_cl_float, np.nanmean, size=3, mode='nearest')
        susc_cl_smoothed[np.isnan(susc_cl_smoothed)] = 0
        # type int
        susc_cl_smoothed = susc_cl_smoothed.astype(np.int8)
        # save
        susc_class_oufolder = os.path.join(self.datapath, 'susceptibility', self.vs, 'susc_classified', cl)
        os.makedirs(susc_class_oufolder, exist_ok=True)
        file1 = os.path.basename(susc_path)
        out1 = f'{susc_class_oufolder}/{file1}'
        self.R.save_raster_as(susc_cl, out1, reference_file = susc_path, dtype = np.int8(), nodata = 0)
        

    def get_fuel_type(self):
        out_ft = f'{self.datapath}/fuel_type_4cl/{self.vs}/ft.tif'
        os.makedirs(os.path.dirname(out_ft), exist_ok=True)
        veg = self.R.read_1band(self.veg_path)
        mapping = json.load(open(self.mapping_path))
        fueltype = self.R.remap_raster(veg, mapping)
        self.R.save_raster_as(fueltype, out_ft, 
                            reference_file = self.dem_file, dtype = np.int8(), nodata = 0)


    def merge_cl_output(self, year, month, cls):

        folder_merged = f'{self.datapath}/susceptibility/{self.vs}/susc_classified'
        outfile = f'{folder_merged}/susc_{year}_{month}.tif'
        outfile2 = outfile.replace('.tif', '_smoothed.tif')
        if not os.path.exists(outfile):
            # merge tiles
            files_to_merge = [f"{self.datapath}/susceptibility/{self.vs}/{cl}/susc_classified/susc_{year}_{month}.tif"
                            for cl in cls]

            out = self.R.merge_rasters(outfile, np.nan, 'max', *files_to_merge)

            # fix nan data (nan to -1)
            with rio.open(out) as src:
                arr = src.read(1)
                arr[np.isnan(arr)] = 0
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
                arr_float = arr.astype(float)
                arr_smooth = ndimage.generic_filter(arr_float, np.nanmean, size=3, mode='nearest', cval=0)
                arr_smooth = np.rint(arr_smooth).astype(arr.dtype)
                with rio.open(outfile2, 'w', **out_meta) as dst:
                    dst.write(arr_smooth, 1)

        self.R.reproject_raster_as_v2(outfile2, outfile2.replace('.tif', '_reproj.tif'), self.dem_file, 
                                    input_crs=self.working_crs, working_crs = self.working_crs, interpolation = 'near')
        
        os.rename(outfile2.replace('.tif', '_reproj.tif'), outfile)
        os.remove(outfile2)



    # get hazards (fuel maps)
    def get_haz(self, year, month):

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

        except Exception as e:
            print(f'no susc for {year}_{month}: {e}')


    def plot_susc(self, total_ba, tr1, tr2, year, month):

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
                susc_nodata= -1,
                pixel_to_ha_factor= self.settings_plt_susc['pixel_to_ha_factor'],
                allow_hist= True,
                allow_pie= True,
                allow_fires= True,
                normalize_over_y_axis= self.settings_plt_susc['normalize_over_y_axis'],
                limit_barperc_to_show= 2,
                is_categorical = True
            )

            self.F.plot_susc_with_bars(**settings)
            plt.close('all')
        except Exception as e:
            print(f'no susc map for - {year}_{month}: {e}')
        

    def merge_all_pngs(self):

        yearmonths = [f"{year}{month}" for year in self.years for month in self.months]
        year_filenames = [f'susc_plot_{yrm}' for yrm in yearmonths]
        basep = f'{self.datapath}/susceptibility/{self.vs}/susc_classified/PNG'
        year_files = [f"{basep}/{filename}.png" for filename in year_filenames]
        year_files = [f for f in year_files if os.path.exists(f)]
        

        fig = self.I.merge_images(year_files, 
                                  ncol=self.settings_plt_susc['ncol'], 
                                  nrow=self.settings_plt_susc['nrow'])
        # save image (Image object)
        out = f'{basep}/MERGED'
        os.makedirs(out, exist_ok=True)
        fig.save(f"{out}/susc_plot_merged.png")


    def run_all(self):

        if self.four_models:
            cls = [1,2,3,4]
            for cl in cls:
                print(f'Processing class {cl}...')
                
                # insert some checks of all the susc tiles are present, if not stop the process.
                

                with mp.Pool(25) as pool:
                    pool.starmap(self.remove_tile_borders, [(tile, year, month, cl) 
                                                            for tile in self.tiles for year in self.years for month in self.months])

                with mp.Pool(25) as pool:
                    pool.starmap(self.merge_susc_tiles, [(year, month, cl) 
                                                        for year in self.years for month in self.months])
                    
                yearmonths = [f'{year}_{month}' for year in self.years for month in self.months]
                for yr in yearmonths:
                    self.reproj_merged_susc(yr, cl)

                thresholds = self.eval_thresholds(cl)

                folder_susc = f'{self.datapath}/susceptibility/{self.vs}/{cl}' 
                susc_names = [i for i in os.listdir(folder_susc) if i.endswith('.tif') and not i.endswith('raw.tif')]
                with mp.Pool(processes=20) as pool:
                    pool.starmap(self.get_categoric_susc, [(os.path.join(folder_susc, susc_name), thresholds, cl) 
                                                            for susc_name in susc_names])
                

                # merge susc of different cl outputs
                with mp.Pool(processes=20) as pool:
                    pool.starmap(self.merge_cl_output, [(year, month) for year in self.years for month in self.months])


        else:
            
            # insert some checks of all the susc tiles are present, if not stop the process.

            with mp.Pool(self.cores) as pool:
                pool.starmap(self.remove_tile_borders, [(tile, year, month) 
                                                        for tile in self.tiles for year in self.years for month in self.months])

            with mp.Pool(self.cores) as pool:
                pool.starmap(self.merge_susc_tiles, [(year, month) 
                                                    for year in self.years for month in self.months])
                
            yearmonths = [f'{year}_{month}' for year in self.years for month in self.months]
            for yr in yearmonths:
                self.reproj_merged_susc(yr)

            thresholds = self.eval_thresholds()

            folder_susc = f'{self.datapath}/susceptibility/{self.vs}' #/cl
            susc_names = [i for i in os.listdir(folder_susc) if i.endswith('.tif') and not i.endswith('raw.tif')]
            with mp.Pool(processes=self.cores) as pool:
                pool.starmap(self.get_categoric_susc, [(os.path.join(folder_susc, susc_name), thresholds) 
                                                        for susc_name in susc_names])
            

        # same operation for cl and not
        print('------------------------------')
        self.get_fuel_type()
        with mp.Pool(processes=self.cores) as pool:
            pool.starmap(self.get_haz, [(year, month) 
                                        for year in self.years for month in self.months])
        
        # plot
        total_ba = gpd.read_file(self.fires_file).area.sum() / 10000
        with mp.Pool(processes=self.cores) as pool:
            pool.starmap(self.plot_susc, [(total_ba, thresholds[0], thresholds[1], year, month) 
                                        for year in self.years for month in self.months])
        
        self.merge_all_pngs()

        

            
        
            

                
            

            
                
            
            




