import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import psycopg2
import shapely.wkb as wkb
from geopandas import GeoSeries
from pathlib import Path
from netCDF4 import Dataset
from os import walk


import pdb

def check_data_gap(df):

    df.dropna(axis='rows', how='all', inplace=True)
    missing_dates = pd.date_range(df.index.min(), df.index.max()).difference(df.index)
    
    print(f'Date start: {df.index.min().strftime("%Y-%m-%d")}, date end: {df.index.max().strftime("%Y-%m-%d")}')

    if len(missing_dates) > 0:
        print(f"Missing dates: {', '.join(missing_dates.strftime('%Y-%m-%d'))}")
    else:
        print('No missing dates')
    # return missing_dates

"""
def interpolate_df(df):

    df = df.reindex(pd.date_range(df.index.min(), df.index.max()), fill_value=np.nan)
    return df.interpolate()
"""


def readnetcdf_in_shp_db(nc_fileName, STAT_CODE, res=5500, plot=False):
    
    # Open the netcdf file
    ds = xr.open_dataset(nc_fileName)
    
    # Open the shape file and reproject it to the MESCAN-Surfex grid (unit=meters)

    conn = psycopg2.connect(host="10.8.244.31",
                       database="climate_data",
                       user="ado_user",
                       password="hydro#ado",
                       port=5432)
                       
    cur = conn.cursor()
    
    # get the metadata
    query = f"""
            SELECT "geom" FROM "hydrology"."ado_basins_CCM2_final2" WHERE "id_station" = '{STAT_CODE}'    
            """
    df = pd.read_sql_query(query,conn)
    
    # close the connection when finished
    cur.close()
    conn.close()
    
    
    shp=GeoSeries(wkb.loads(df.geom[0], hex=True))
    shp=shp.set_crs("EPSG:4326")
    shp_reproj = shp.to_crs('+proj=lcc +lat_1=50 +lat_2=50 +lat_0=50 +lon_0=8 +x_0=2937018.5829291 +y_0=2937031.41074803 +a=6371229 +b=6371229')
    
    # Crop ds with the shapefile bounding box (bb)
    bb = shp_reproj.bounds.iloc[0]
    ds = ds.sel(x=slice(bb['minx']-res, bb['maxx']+res), 
                y=slice(bb['miny']-res, bb['maxy']+res))
    
    
    #0000 Mask all the points in ds where the grid box do not intersect or is in the shapefile
    for i in ds.x.values:
        for j in ds.y.values:
            gridbox = Point(i, j)#.buffer(res/2, cap_style=3)
            if not (gridbox.intersects(shp_reproj.loc[0])):
                for k in ds.data_vars.keys():
                    if not (k =='Lambert_Conformal' or k=='time_bnds'):
                        ds[k].loc[dict(x=i, y=j)] = np.nan
    ds = ds.dropna(dim='x', how='all')
    ds = ds.dropna(dim='y', how='all')
    counter=0                    
    # Plot the era5 gridbox and the shapefile if plot=True
    if plot:
        plt.figure(figsize=(25,25))
        for x in ds.x.values:
            for y in ds.y.values:
                gridbox = Point(x, y).buffer(res / 2, cap_style=3)
                gridbox_x, gridbox_y = gridbox.exterior.xy
                plt.plot(gridbox_x, gridbox_y, color='blue')
                for k in ds.data_vars.keys():
                    if not (k =='Lambert_Conformal' or k=='time_bnds'):
                        if not(ds[k].loc[dict(x=x, y=y)].isnull().all()):
                            plt.plot(x, y, marker='o', color='red')
                            counter=counter+1
        
        coords=[p.exterior.xy for p in shp_reproj.loc[0]]
        shp_x=coords[0][0]
        shp_y=coords[0][1]
        #shp_x, shp_y = *shp_reproj.loc[0].exterior.xy
        plt.plot(shp_x, shp_y, color='black')
        plt.axis('equal')                        
        print(f'n of pixels{counter}')  
    
    return ds


def xarray2df(xa, varnamedest,varnameor=False):
    if not varnameor:
        df = {}
        for i in range(xa.y.size):
            for j in range(xa.x.size):
                df[f'{varnamedest}x{j}y{i}'] = xa.isel(y=i, x=j).to_dataframe().iloc[:, 2]
                #pdb.set_trace()

    else:
        df = {}
        for i in range(xa.y.size):
            for j in range(xa.x.size):
                df[f'{varnamedest}x{j}y{i}'] = xa.isel(y=i, x=j).to_dataframe().loc[:,varnameor]
                #pdb.set_trace()

    frame=pd.DataFrame(df)
    return frame


def get_discharge_from_DB(STAT_CODE):
    #read the csv file with the daily discharge

    # establish connection using information supplied in documentation
    conn = psycopg2.connect(host="10.8.244.31",
                           database="climate_data",
                           user="ado_user",
                           password="hydro#ado",
                           port=5432)
    cur = conn.cursor()
    
    # get the metadata
    query = f"""
    SELECT date, discharge_m3_s FROM hydrology.discharge WHERE id_station = '{STAT_CODE}' ORDER BY date;
    """
    
    df = pd.read_sql_query(query, conn)
   
    #set the date in a proper format
    df.index = pd.to_datetime(df.date)
    df.drop(columns='date',inplace=True)
    # close the connection when finished
    cur.close()
    conn.close()
    return df
    
    
def spatial_stats_daily_input(daily_input):
    
    #allocate the returned  variable
    new_daily_input=pd.DataFrame()
        
    #for every variable compute the mean and the 4 quantiles stats.
    
    t_columns = [c for c in daily_input.columns if c[0] =='T']
    t_vars=daily_input[t_columns]
    new_daily_input.loc[:,'T']  = t_vars.mean(axis=1)
    new_daily_input.loc[:,'T5'] =t_vars.quantile(q=0.05,axis=1)
    new_daily_input.loc[:,'T25']=t_vars.quantile(q=0.25,axis=1)
    new_daily_input.loc[:,'T75']=t_vars.quantile(q=0.75,axis=1)
    new_daily_input.insert(loc=4,column='T95',value=t_vars.quantile(q=0.95,axis=1))
    
    
    s_columns = [c for c in daily_input.columns if c[0] =='S']
    s_vars=daily_input[s_columns]
    new_daily_input.loc[:,'S']  = s_vars.mean(axis=1)
    new_daily_input.loc[:,'S5'] =s_vars.quantile(q=0.05,axis=1)
    new_daily_input.loc[:,'S25']=s_vars.quantile(q=0.25,axis=1)
    new_daily_input.loc[:,'S75']=s_vars.quantile(q=0.75,axis=1)
    new_daily_input.loc[:,'S95']=s_vars.quantile(q=0.95,axis=1)
    
    p_columns = [c for c in daily_input.columns if c[0] =='P']
    p_vars=daily_input[p_columns]
    new_daily_input.loc[:,'P']  = p_vars.mean(axis=1)
    new_daily_input.loc[:,'P5'] =p_vars.quantile(q=0.05,axis=1)
    new_daily_input.loc[:,'P25']=p_vars.quantile(q=0.25,axis=1)
    new_daily_input.loc[:,'P75']=p_vars.quantile(q=0.75,axis=1)
    new_daily_input.loc[:,'P95']=p_vars.quantile(q=0.95,axis=1)

    e_columns = [c for c in daily_input.columns if c[0] =='E']
    e_vars=daily_input[e_columns]
    new_daily_input.loc[:,'E']  = e_vars.mean(axis=1)
    new_daily_input.loc[:,'E5'] =e_vars.quantile(q=0.05,axis=1)
    new_daily_input.loc[:,'E25']=e_vars.quantile(q=0.25,axis=1)
    new_daily_input.loc[:,'E75']=e_vars.quantile(q=0.75,axis=1)
    new_daily_input.loc[:,'E95']=e_vars.quantile(q=0.95,axis=1)
    
    
    if ('Q' in daily_input.columns):
        new_daily_input.loc[:,'Q']=(daily_input.Q)
        
        
    return new_daily_input
# ------------------------------------------------------------------------------------------------------------------





def readsnow_in_shp_db(src_folder,mask_file, STAT_CODE, res=5500, plot=False):


    # CONCATENATE THE SNOW MODEL FILES IN THE  src_folder
    
    filenames = next(walk(src_folder), (None, None, []))[2]  # [] if no file
    



    # OPEN THE MASK FILE AND CLIP-OUT THE GLACIER PIXELS
    
    mask=xr.open_dataset(mask_file)
    non_glacier_mask = np.logical_not(np.array(mask.snowgrid_mask6))
    
    # Open the shape file and reproject it to the MESCAN-Surfex grid (unit=meters)
    conn = psycopg2.connect(host="10.8.244.31",
                       database="climate_data",
                       user="ado_user",
                       password="hydro#ado",
                       port=5432)

    cur = conn.cursor()
    
    # get the OUTLINE
    query = f"""
            SELECT "geom" FROM "hydrology"."ado_basins_CCM2_final2" WHERE "id_station" = '{STAT_CODE}'
            """
    df = pd.read_sql_query(query,conn)
    
    # close the connection when finished
    cur.close()
    conn.close()
    
    
    shp=GeoSeries(wkb.loads(df.geom[0], hex=True))
    shp=shp.set_crs("EPSG:4326")
    shp_reproj = shp.to_crs('+proj=lcc +lat_1=50 +lat_2=50 +lat_0=50 +lon_0=8 +x_0=2937018.5829291 +y_0=2937031.41074803 +a=6371229 +b=6371229')
    bb = shp_reproj.bounds.iloc[0]

    c=0
    for i in filenames:
        if c==0:
            ds = xr.open_dataset(src_folder+i)
            ds=(ds.swe_tot.where(non_glacier_mask)).to_dataset()
   
            # Crop ds with the shapefile bounding box (bb)
            ds = ds.sel(x=slice(bb['minx']-res, bb['maxx']+res), 
                        y=slice(bb['miny']-res, bb['maxy']+res))

        else:
            ds_add = xr.open_dataset(src_folder+i)
            ds_add=(ds_add.swe_tot.where(non_glacier_mask)).to_dataset()            

            # Crop ds with the shapefile bounding box (bb)
            ds_add = ds_add.sel(x=slice(bb['minx']-res, bb['maxx']+res), 
                        y=slice(bb['miny']-res, bb['maxy']+res))
                        
            ds=xr.concat([ds, ds_add], dim="time")
            
        c=c+1
            
            
            
    #0000 Mask all the points in ds where the grid box do not intersect or is in the shapefile
    for i in ds.x.values:
        for j in ds.y.values:
            gridbox = Point(i, j)#.buffer(res/2, cap_style=3)
            if not (gridbox.intersects(shp_reproj.loc[0])):
                for k in ds.data_vars.keys():
                    if not (k =='Lambert_Conformal' or k=='time_bnds'):
                        ds[k].loc[dict(x=i, y=j)] = np.nan
    ds = ds.dropna(dim='x', how='all')
    ds = ds.dropna(dim='y', how='all')
    
    counter=0                    
    # Plot the era5 gridbox and the shapefile if plot=True
    if plot:
        plt.figure(figsize=(25,25))
        for x in ds.x.values:
            for y in ds.y.values:
                gridbox = Point(x, y).buffer(res / 2, cap_style=3)
                gridbox_x, gridbox_y = gridbox.exterior.xy
                plt.plot(gridbox_x, gridbox_y, color='blue')
                for k in ds.data_vars.keys():
                    if not (k =='Lambert_Conformal' or k=='time_bnds'):
                        if not(ds[k].loc[dict(x=x, y=y)].isnull().all()):
                            plt.plot(x, y, marker='o', color='red')
                            counter=counter+1
        print(f'n of pixels{counter}')

        coords=[p.exterior.xy for p in shp_reproj.loc[0]]
        shp_x=coords[0][0]
        shp_y=coords[0][1]
        #shp_x, shp_y = *shp_reproj.loc[0].exterior.xy
        plt.plot(shp_x, shp_y, color='black')
        plt.axis('equal')                        
    
    return ds
    