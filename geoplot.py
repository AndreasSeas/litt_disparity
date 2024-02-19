"""
Developer: 
    Andreas Seas

Last Updated:
    23.07.03
    
Environment init:
    conda activate lit_disparity
    
"""

# =============================================================================
# import critical packages
# =============================================================================
import pandas as pd
import pgeocode
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import censusdata
import pingouin as pg
import geopandas as gpd
import geodatasets
from shapely.geometry import Point
import xycmap
import os
# load usa geometry
usa = gpd.read_file(geodatasets.get_path('geoda.natregimes'))

homedir=os.getcwd()
savefig=True;

# =============================================================================
# functions
# =============================================================================
def pullacs5detail(subgroup,year,dataname):
       
    data_raw = censusdata.download('acs5', year,
               censusdata.censusgeo([('zip code tabulation area', '*')]),
              subgroup)
    
    data_raw.head()
    
    data = data_raw.copy()
    data.index = [str(x)[-5:] for x in data.index]
    data.index.name = 'zcta'
    data.columns = [dataname]

    return data

def pullacs5DP(subgroup,year,dataname):
       
    '''
    resources:
        variable descriptions:
            https://api.census.gov/data/2021/acs/acs5/profile/variables.html
        corresponding datatable:
            https://data.census.gov/table?d=ACS+5-Year+Estimates+Data+Profiles&tid=ACSDP5Y2021.DP05
        general stats handbook:
            https://www.census.gov/content/dam/Census/library/publications/2018/acs/acs_general_handbook_2018_ch07.pdf
        based off this example:
            https://jtleider.github.io/censusdata/example1.html
    '''
    
    # use this one in order to pull more accurate profile data
    
    print("reading acs5 data for {}".format(dataname))
    
    data_raw = censusdata.download('acs5', year,
               censusdata.censusgeo([('zip code tabulation area', '*')]),
              subgroup,tabletype='profile')
    
    data_raw.head()
    
    data = data_raw.copy()
    data.index = [str(x)[-5:] for x in data.index]
    data.index.name = 'zcta'
    data.columns = [dataname]

    return data

# =============================================================================
# Load patient zipcode data
# =============================================================================

name_littzipcodes='/Users/as822/Library/CloudStorage/Box-Box/McIntyreLab_Andreas/Writing/Working/W-Manuscript-LITTDisparity/LITT_Zip_Code_Data.csv'

df=pd.read_csv(name_littzipcodes)

# =============================================================================
# clean patient data and remove pts without clear race/ethnic identity
# =============================================================================
# first drop patients with refusal to respond to question of race/ethnicity
df=df.drop(np.where(df['Race']==6)[0])# 6 is the index used for NR for race
df=df.drop(np.where(df['Hispanic']==2)[0])# 2 is the index used for NR for ethnicity

# get the unique zipcodes among all patients
unique_zipcodes=set(df["Zip Code"].unique())

## identify the numer of individuals from each of the two groups identified here
# first make a dataframe to store the data
df_zip=pd.DataFrame(data=None,index=unique_zipcodes,
                    columns=['n_pts','n_white_nonhisp','n_other',
                             'n_population','pct_minority','zcta','shp'])
#   n_pts = number of patients from that region
#   n_white_nonhisp = number of white, non-hispanic individuals in that zipcode
#   n_other = npts-n_white_nonhisp
#   n_population = total population in that zipcode
#   pct_minority = percent minority
#   zcta = the zip code tabulate area != zip code!
#   shp = shapefile for that specific zipcode

for zipcode in unique_zipcodes:
    temp=df.loc[df["Zip Code"]==zipcode,:];
    
    df_zip.loc[zipcode,'n_pts']=len(temp)
    df_zip.loc[zipcode,'n_white_nonhisp']=((temp.Race==0) & (temp.Hispanic==0)).sum()
    df_zip.loc[zipcode,'n_other']=len(temp)-df_zip.loc[zipcode,'n_white_nonhisp']
    
# =============================================================================
# Load data on different groups
# =============================================================================
# doing 1-[white nonhispanic] population to calculate minority pop
# why?
# 	I just computed one example for ZCTA 25311
# 	    #tot = 10964
# 	    #white = 8343, i.e. 76.1% or 23.9% non-white
# 	    #black = 1664, #asian = 88, #hispanic=163
# 	     (1664+88+163)/10964 = 17.5%
# 	differential is 6%
# 	since the white population has only one error margin, by minimizing the 
#       number of calculations we do, we minimize error propagation

# white non hispanic
# subgroup=["DP05_0077E"]; 
# dataname = 'wnh'
# wnh_E=pullacs5DP(subgroup=subgroup,year=2019,dataname=dataname)

# Percent white non-hispanic
subgroup=["DP05_0077PE"]; 
dataname = 'wnh_PE'
wnh_PE=pullacs5DP(subgroup=subgroup,year=2019,dataname=dataname)

# total population (in order to potentially reweigh the values above)
subgroup=["DP05_0070E"]; 
dataname = 'tot'
tot=pullacs5DP(subgroup=subgroup,year=2019,dataname=dataname)

df_census=pd.concat([wnh_PE,tot],axis=1)

df_census.loc[:,'minority']=100-df_census.wnh_PE

# =============================================================================
# Cross-walk between ZCTA and Zip Code
# =============================================================================
# reference is here: 
#    https://udsmapper.org/zip-code-to-zcta-crosswalk/

name_crosswalk='/Users/as822/Library/CloudStorage/Box-Box/McIntyreLab_Andreas/Writing/Working/W-Manuscript-LITTDisparity/ZIPCodetoZCTACrosswalk2021UDS.xlsx'

df_cw=pd.read_excel(name_crosswalk);

# zcta_list=df_census.index.astype(int).to_list()# get all zipcodes as int
# zip_list=[None]*len(zcta_list);# create mpty list

# leng=0
# for i,zcta in enumerate(zcta_list):
    # zip_list[i]=df_cw.loc[df_cw.ZCTA==int(zcta),'ZIP_CODE'].values
    # if len(df_cw.loc[df_cw.ZCTA==int(zcta),'ZIP_CODE'].values)>1:
        # fdsa
    # need to use a list becasue some ZCTA have multiple Zip Codes

# =============================================================================
# collate into one larger dataframe for each zipcode
# =============================================================================

for zipcode in df_zip.index:
    df_zip.loc[zipcode,'zcta']=int(df_cw.loc[df_cw.ZIP_CODE==zipcode,'ZCTA'].values[0])
    # print(len(df_cw.loc[df_cw.ZIP_CODE==zipcode,'ZCTA']))
    zcta_temp="{0:0=5d}".format((df_zip.loc[zipcode,'zcta']))
    df_zip.loc[zipcode,'n_population']=df_census.loc[zcta_temp,'tot']
    df_zip.loc[zipcode,'pct_minority']=df_census.loc[zcta_temp,'minority']

# note that there are some instances where one zcta == 2 zip codes
# this is in instances only where there are large suppliers or clients

# get rid of rows with zero population (Naples Florida, 34101, is 0 people)
df_zip=df_zip.loc[df_zip.n_population>0,:]

# =============================================================================
# compute lat long and distance from Duke
# =============================================================================

df_zip["Latitude"]=np.nan
df_zip["Longitude"]=np.nan
df_zip["Distance"]=np.nan
###
nomi = pgeocode.Nominatim('US')

dukedata=nomi.query_postal_code("27710")
lat_duke=dukedata.latitude
lon_duke=dukedata.longitude

Lat2=lat_duke*np.pi/180
Long2=lon_duke*np.pi/180

#constant
R=6371e3;#metres

for i in df_zip.index:
    temp=nomi.query_postal_code(str(i))
    df_zip.loc[i,"Latitude"]=temp.latitude
    df_zip.loc[i,"Longitude"]=temp.longitude
    
    # Calculate distance using lat/lon:
        # https://keisan.casio.com/exec/system/1224587128
        # https://www.movable-type.co.uk/scripts/latlong.html
    Lat1=temp.latitude*np.pi/180
    Long1=temp.longitude*np.pi/180
    
    dLat=Lat1-Lat2
    dLong=Long1-Long2
    
    if np.abs(dLat)>0:
        a=np.sin(dLat/2)**2 + np.cos(Lat1) * np.cos(Lat2) * np.sin(dLong/2)**2
        c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
        dist=R*c
        df_zip.loc[i,'Distance']=dist/1000*0.621371
    else:
        df_zip.loc[i,'Distance']=np.nan 

# =============================================================================
# grab shape files from census
# =============================================================================
# source is here
# https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.2019.html#list-tab-1883739534
name_shape='/Users/as822/Library/CloudStorage/Box-Box/McIntyreLab_Andreas/Writing/Working/W-Manuscript-LITTDisparity/cb_2019_us_zcta510_500k/cb_2019_us_zcta510_500k.shp'
df_shp=(gpd.read_file(name_shape)
            .rename(columns={'ZCTA5CE10':'zipcode','geometry':'zcta_polygon'}))[['zipcode','zcta_polygon']].set_index('zipcode')

df_zip['shptype']=np.nan

for i,zipcode in enumerate(df_zip.index):
    temp_zip="{0:0=5d}".format(zipcode)
    temp_zcta="{0:0=5d}".format(df_zip.loc[zipcode,'zcta'])
    if (df_shp.index==temp_zip).sum()==0:
        # specific example of one zipcode not represented, where the zcta is actually more accurate
        # Hickory NC, ZCTA = 28603, zipcode 28602, but 28602 is located in 28603
        # in this instance, use the shapefile of the zcta
        if (df_shp.index==temp_zcta).sum()==1:
            print('getting zcta')
            df_zip.loc[zipcode,'shp']=df_shp.loc[temp_zcta,'zcta_polygon']
            df_zip.loc[zipcode,'shptype']='zcta'
        else:
            df_zip.loc[zipcode,'shptype']=np.nan
        
    else:
        df_zip.loc[zipcode,'shp']=df_shp.loc[temp_zip,'zcta_polygon']
        df_zip.loc[zipcode,'shptype']='zipcode'
        
df_zip=df_zip.dropna()# drop the na ones cus cant get a shape really
# none in this instance

# # =============================================================================
# # Plot scatterplot
# # =============================================================================

# #plot
# fig,ax=plt.subplots()
x=df_zip.pct_minority.astype(float)
y=(df_zip.n_pts/df_zip.n_population*100000).astype(float);# per 100k
s=df_zip.n_population.astype(float);
d=df_zip.Distance.astype(float);

# cm = plt.cm.get_cmap('magma')
# sc=ax.scatter(x=x,y=y,c=d,cmap=cm)
# # ax.plot(x,y,'or')

# lm = pg.linear_regression(x, y)
# lm2 = pg.linear_regression(x, y, weights=df_zip.n_population.astype(float))

# # df_zip.loc[:,'x']=x#.astype(float)
# # df_zip.loc[:,'y']=y#.astype(float)


# # sns.lmplot(data=df_zip,x='x',y='y',hue='n_population',)
# ax.set_xlabel('% Population Non-White and/or Hispanic')
# ax.set_ylabel('LITT Patients/100K')
# plt.colorbar(sc,)

# # also potentially plot with the color coinciding to distance from Duke?

# # =============================================================================
# # Plot geomap
# # =============================================================================
# # ax = df.plot(figsize=(10, 10), alpha=0.5, edgecolor="k")

# # world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

# fig, ax=plt.subplots(figsize=(16,8),ncols=1,nrows=1,)
# usa.plot(ax=ax,color=(0.9,0.9,0.9))
# usa.boundary.plot(ax=ax,color=(0.5,0.5,0.5),linewidth=0.1)

# sc=ax.scatter(x=df_zip.Longitude,y=df_zip.Latitude,c=x,cmap=cm,s=30,alpha=0.7)
# ax.plot(lon_duke,lat_duke,'xr');
# ax.set_xlim(-95,-72);
# ax.set_ylim(24.5,40);
# cbar=plt.colorbar(sc,)
# cbar.set_label('% Population Non-White and/or Hispanic')

# fig, ax=plt.subplots(figsize=(16,8),ncols=1,nrows=1,)
# usa.plot(ax=ax,color=(0.9,0.9,0.9))
# usa.boundary.plot(ax=ax,color=(0.5,0.5,0.5),linewidth=0.1)

# sc=ax.scatter(x=df_zip.Longitude,y=df_zip.Latitude,c=y,cmap=cm,s=30,alpha=0.7)
# ax.plot(lon_duke,lat_duke,'xr');
# ax.set_xlim(-95,-72);
# ax.set_ylim(24.5,40);
# cbar=plt.colorbar(sc,)
# cbar.set_label('LITT Patients/100K')

# sys.exit()

# fig, ax=plt.subplots(figsize=(16,8),ncols=1,nrows=1,)
# usa.plot(ax=ax,color=(0.9,0.9,0.9))
# usa.boundary.plot(ax=ax,color=(0.5,0.5,0.5),linewidth=0.1)

# df_zip.pct_minority=df_zip.pct_minority.astype(float)
# df_zip['yval']=y

# # plot kdeplot weighted by pct minority
# sns.kdeplot(
#     data=df_zip, 
#     x='Longitude', 
#     y="Latitude", 
#     fill=True, 
#     ax=ax,
#     palette="BrBG",
#     weight='pct_minority'
# )

# fig, ax=plt.subplots(figsize=(16,8),ncols=1,nrows=1,)
# usa.plot(ax=ax,color=(0.9,0.9,0.9))
# usa.boundary.plot(ax=ax,color=(0.5,0.5,0.5),linewidth=0.1)

# df_zip.pct_minority=df_zip.pct_minority.astype(float)
# df_zip['yval']=y

# # plot kdeplot weighted by pct minority
# sns.kdeplot(
#     data=df_zip, 
#     x='Longitude', 
#     y="Latitude", 
#     fill=True, 
#     ax=ax,
#     palette="BrBG",
#     weight='y'
# )

# fig, ax=plt.subplots(figsize=(8,8),ncols=1,nrows=1,subplot_kw={'projection':'3d'})
# # ax=plt.axes(projection='3d')

# s=ax.scatter3D(x,d,y,c=y,cmap='BrBG')
# ax.set_xlabel('% Population Non-White and/or Hispanic')
# ax.set_zlabel('LITT Patients/100K')
# ax.set_ylabel('Distance (miles)')
# cbar=plt.colorbar(s,orientation='vertical',label='LITT Patients/100K')
# plt.tight_layout()

# =============================================================================
# try xycmap
# =============================================================================
# https://github.com/rbjansen/xycmap
# fig, ax=plt.subplots(figsize=(10,10),ncols=1,nrows=1,)

fig, ax=plt.subplots(figsize=(5,7),ncols=1,nrows=2,
                      gridspec_kw={'height_ratios': [2, 1]},
                      squeeze=True)
ax_set=ax;
ax=ax_set[0]
usa.plot(ax=ax,color=(0.9,0.9,0.9))
usa.boundary.plot(ax=ax,color=(0.5,0.5,0.5),linewidth=0.1)

# make these floats
df_zip['xval']=df_zip.pct_minority.astype(float)
df_zip['yval']=y
xcmap=plt.cm.PiYG
ycmap=plt.cm.Greys

cmap = xycmap.mean_xycmap(xcmap=xcmap, ycmap=ycmap, n=(5,5))
colors = xycmap.bivariate_color(sx=df_zip['xval'], 
                                sy=df_zip['yval'], 
                                cmap=cmap,
                                xlims=(0,100),
                                ylims=(0,62))

sc=ax.scatter(x=df_zip.Longitude,y=df_zip.Latitude,c=colors,cmap=cmap,s=40,alpha=0.8,edgecolor='k',linewidth=0.5)

ax.plot(lon_duke,lat_duke,'xb',markersize=10,linewidth=8);
## for bigger map
# ax.set_xlim(-97,-72);
# ax.set_ylim(24,43);
## for focused
ax.set_xlim(-86,-72);
ax.set_ylim(30,40);

ax.axis('off')

ax=ax_set[1]
x=df_zip.pct_minority.astype(float)
y=(df_zip.n_pts/df_zip.n_population*100000).astype(float);# per 100k
s=df_zip.n_population.astype(float);
d=df_zip.Distance.astype(float);

sc=ax.scatter(x=x,y=y,c=colors,s=40,alpha=0.8,edgecolor='k',linewidth=0.5)
# ax.plot(x,y,'or')

lm = pg.linear_regression(x, y)
lm2 = pg.linear_regression(x, y, weights=df_zip.n_population.astype(float))

line1, =ax.plot([0,90],[lm.coef[0],lm.coef[0]+lm.coef[1]*90],'-k',
              label="unweighted p = {:.5f}".format(lm.pval[1]))
# line2, =ax.plot([0,90],[lm2.coef[0],lm2.coef[0]+lm2.coef[1]*90],'--k',
#               label="weighted p = {:.3f}".format(lm2.pval[1]))

ax.legend(handles=[line1])

# ax.text(60,30,"$\p = {}$".format(kappa_4),fontsize=16,va='center',ha='center',
#            bbox=dict(boxstyle="round", fc="white", ec="black", pad=0.2))

# sns.lmplot(data=df_zip,x='x',y='y',hue='n_population',)
ax.set_xlabel("% Non-White and/or Hispanic")
ax.set_ylabel('LITT Patients/100K')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

cax = fig.add_axes([0.70, 0.41, 0.2, 0.2])
cax = xycmap.bivariate_legend(ax=cax, sx=df_zip['xval'], sy=df_zip['yval'], 
                              cmap=cmap,
                              xlims=(0,100),
                              ylims=(0,62))
cax.set_xlabel('% Non-White and/or Hispanic')
cax.set_ylabel('LITT Patients/100K')
cax.set_xticks(cax.get_xlim())
cax.set_yticks(cax.get_ylim())

savefig=True
os.chdir('/Users/as822/Library/CloudStorage/Box-Box/McIntyreLab_Andreas/Writing/Working/W-Manuscript-LITTDisparity/figures/')
if savefig: 
    fig.savefig('finalfigure.png',dpi=600);
    fig.savefig('finalfigure.svg');
    tempdf=df_zip.drop('shp',axis=1)
    tempdf.to_csv('dataframe.csv')
    lm.to_csv('linear_unweighted.csv')
    lm2.to_csv('linear_weighted.csv')
os.chdir(homedir)
sys.exit()
# =============================================================================
# binarized thing
# =============================================================================
df_dist=df;

df_dist['Distance']=np.nan

for i in df_dist.index:
    temp_zip=df_dist.loc[i,"Zip Code"]
    # temp_zip="{0:0=5d}".format(df_dist.loc[i,"Zip Code"])
    if (df_zip.index==temp_zip).sum()>0:
        df_dist.loc[i,"Distance"] = df_zip.loc[temp_zip,'Distance']

df_dist=df_dist.dropna()

df_dist["NHW"]=(df_dist['Race']==0) & (df_dist['Hispanic']==0) 

sns.stripplot(data=df_dist,x='NHW',y='Distance')

sns.kdeplot(data=df_dist,x='Distance',hue='NHW')

# https://pingouin-stats.org/build/html/generated/pingouin.mwu.html
stats=pg.mwu(x=df_dist.loc[df_dist.NHW,'Distance'].values,y=df_dist.loc[df_dist.NHW==False,'Distance'].values,alternative='greater')

###############################################################################
###############################################################################
###############################################################################
# # sys.exit()
# # # sns.kdeplot(
# # #     data=df_zip, x='Distance', y="pct_minority", fill=True, ax=ax,palette="tab10",
# # #     hue='pct_minority'
# # # )

# # # fig, ax=plt.subplots(figsize=(8,8),ncols=1,nrows=1,)
# # # sns.kdeplot(
# # #     data=df_zip, x='', y="pct_minority", fill=True, ax=ax,palette="tab10",
# # #     hue=
# # # )

# # sc=ax.scatter(x=df_zip.Longitude,y=df_zip.Latitude,c='k',s=15,alpha=0.7)
# # ax.plot(lon_duke,lat_duke,'xr');
# # ax.set_xlim(-95,-72);
# # ax.set_ylim(24.5,40);


# # # cbar=plt.colorbar(sc,)
# # cbar.set_label('LITT Patients/100K')

# # sys.exit()

# # df=df.rename(columns={'Race: 0 - White, 1 - Black, 2 - Asian, 3 - American Indian or AK native, 4 - Pacific Islander/Native Hawaian, 5 - Other/Multiracial, 6 - NR/Declined': "Race"})

# # sns.kdeplot(
# #     data=df, x="Longitude", y="Latitude", fill=True, ax=ax,palette="tab10",
# #     # hue="Race"
# # )

# # plot heatmap

# # ax = geoplot.kdeplot(
# #     collisions.head(1000),
# #     clip=boroughs.geometry,
# #     shade=True,
# #     cmap="Reds",
# #     projection=geoplot.crs.AlbersEqualArea(),
# # )
# # geoplot.polyplot(boroughs, ax=ax, zorder=1)

# ax.set_xlim(-85,-72);
# ax.set_ylim(30,40);

# sys.exit()
# # #total population
# # subgroup=["DP05_0070E"]; 
# # dataname = 'tot'
# # tot=pullacs5DP(subgroup=subgroup,year=2019,dataname=dataname)

# # #non-hispanic black
# # subgroup=["DP05_0078E"]; 
# # dataname = 'blk'
# # blk=pullacs5DP(subgroup=subgroup,year=2019,dataname=dataname)

# # #non-hispanic asian
# # subgroup=["DP05_0080E"]; 
# # dataname = 'asi'
# # asi=pullacs5DP(subgroup=subgroup,year=2019,dataname=dataname)

# # #allhispanic
# # subgroup=["DP05_0071E"]; 
# # dataname = 'his'
# # his=pullacs5DP(subgroup=subgroup,year=2019,dataname=dataname)



# # DP05_0077E
# #(DP05_0078E (non-hispanic black/aa alone) 
# #+DP05_0080E (non-hispanic asian) 
# #+ DP05_0071E (Hispanic or Latino of any race))/DP05_0070E (total popluation)

# sys.exit()
# # =============================================================================
# # Load data on total population
# # =============================================================================
# # maingroup="B01003" # looks at all races
# # censusdata.printtable(censusdata.censustable('acs5', 2019, maingroup))

# subgroup=["B01003_001E"]
# dataname = 'total'
# total=pullacs5DP(subgroup=subgroup,year=2019,dataname=dataname)


# censusdata.printtable(censusdata.censustable('acs5', 2019, 'DP05'))

# subgroup=["DP05_0077E"]; #white non hispanic



# data_raw = censusdata.download('acs5', 
#                                2019, 
#                                censusdata.censusgeo([('zip code tabulation area', '*')]),
#                                    subgroup,tabletype='profile')



# # B98012 - 


# sys.exit()
# # =============================================================================
# # Identify zipcodes
# # =============================================================================
# df["Latitude"]=np.nan
# df["Longitude"]=np.nan

# ###
# nomi = pgeocode.Nominatim('US')

# for i in df.index:
#     temp=nomi.query_postal_code(str(df.loc[i,"Zip Code"]))
#     df.loc[i,"Latitude"]=temp.latitude
#     df.loc[i,"Longitude"]=temp.longitude
#     # print(temp)

# dukedata=nomi.query_postal_code("27710")
# lat_duke=dukedata.latitude
# lon_duke=dukedata.longitude

# # https://geopandas.org/en/stable/docs/user_guide/projections.html
# import geopandas
# import geodatasets
# from shapely.geometry import Point

# usa = geopandas.read_file(geodatasets.get_path('geoda.natregimes'))
# # world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

# fig, ax=plt.subplots(figsize=(16,8),ncols=1,nrows=1,)
# # world.plot(ax=ax,color=(0.9,0.9,0.9))
# # world.boundary.plot(color=(0.3,0.3,0.3),linewidth=0.5,ax=ax)
# usa.plot(ax=ax,color=(0.9,0.9,0.9))
# usa.boundary.plot(ax=ax,color=(0.5,0.5,0.5),linewidth=0.1)

# ax.plot(df.Longitude,df.Latitude,'.k');
# ax.plot(lon_duke,lat_duke,'or');

# df=df.rename(columns={'Race: 0 - White, 1 - Black, 2 - Asian, 3 - American Indian or AK native, 4 - Pacific Islander/Native Hawaian, 5 - Other/Multiracial, 6 - NR/Declined': "Race"})

# # sns.kdeplot(
# #     data=df, x="Longitude", y="Latitude", fill=True, ax=ax,palette="tab10",
# #     # hue="Race"
# # )

# # plot heatmap

# # ax = geoplot.kdeplot(
# #     collisions.head(1000),
# #     clip=boroughs.geometry,
# #     shade=True,
# #     cmap="Reds",
# #     projection=geoplot.crs.AlbersEqualArea(),
# # )
# # geoplot.polyplot(boroughs, ax=ax, zorder=1)

# ax.set_xlim(-85,-72);
# ax.set_ylim(30,40);

# sys.exit()

# ## plot


# import geopandas
# from shapely.geometry import Point


# # https://onelinerhub.com/python-matplotlib/how-to-fill-countries-with-colors-using-world-map

# world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
# # from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/

# all_coords=pd.DataFrame(data=None,columns=['Unique ID','Geometry']);
# for df in dflist:
#     df['Geometry']=np.nan;
#     for i in df.index:
#         df.loc[i,'Geometry']=Point(df.loc[i,'Longitude'],df.loc[i,'Latitude'])
        
#     all_coords=pd.concat([all_coords,df[['Unique ID','Geometry']]])

# all_coords=all_coords.reset_index(drop=True);
# all_coords['Country']=np.nan

# # go thru each country
# for i in range(len(world)):
#     # print(world.loc[i,'name'])
#     geom=world.loc[i,'geometry']
#     for j in range(len(all_coords)):
#         pt=all_coords.loc[j,'Geometry']
#         if geom.contains(pt):
#             all_coords.loc[j,'Country']=world.loc[i,'name'];
#         # print(geom.contains(pt))
        
# # make sure to only keep unique countries
# world['count']=0;
# unique_coords=pd.DataFrame(data=None,columns=['Country'],index=all_coords['Unique ID'].unique());

# countmult=0
# for i, idx in enumerate(unique_coords.index):

#     ctemp=all_coords.loc[all_coords['Unique ID']==idx,'Country']
    
#     for loc in ctemp.unique():
#         world.loc[world.name == loc,'count']=world.loc[world.name == loc,'count']+1
    
#     if len(ctemp.unique())==1:
#         unique_coords.loc[idx,'Country']=ctemp.iloc[0]
        
#     else:
#         unique_coords.loc[idx,'Country']="multiple"
#         # print(ctemp)
#         # countmult=countmult+1

# # world['count']=0;

# # polygon.contains(point)

# plt.rcParams['font.size'] = '12'
# plt.rcParams['font.family'] = 'serif'

# fig, ax=plt.subplots(figsize=(16,8),ncols=1,nrows=1,)
# world.plot(column='count',legend=True,ax=ax,vmin=0,vmax=10,cmap='Blues',
#            legend_kwds={"label": "# Participants", "orientation": "vertical"},)

# # import matplotlib.colors as colors
# # world.plot(column='count',legend=True,ax=ax,vmin=1,vmax=10,cmap='Blues',
# #            norm=colors.LogNorm(vmin=1,vmax=world['count'].max()))
# world[world.name == 'United States of America'].plot(color=[0.5,0.5,0.5],ax=ax)

# world.boundary.plot(color=[0.5,0.5,0.5],linewidth=0.5,ax=ax,)
# # plt.scatter(all_coords.Geometry)

# # world[world.name == 'Ukraine'].plot(color='yellow',ax=ax)

# # plt.show()

# # markerlist=['x','o','x','o','.'];
# colorlist=['#e70843','#90052a','#658b6c','#094614','#391164'];

# for i,df_ in enumerate(dflist):
#     # print(df_.shape)
#     x_=df_.loc[:,'Longitude']
#     y_=df_.loc[:,'Latitude']
#     # ax.plot(x_,y_,markerlist[i],label=dfname[i],);
#     ax.scatter(x_,y_,50,colorlist[i],alpha=0.3,label=dfname[i],)

# ax.legend(loc='lower center',ncol=5,bbox_to_anchor=[0.5,-0.1])


# plt.tight_layout()
# plt.axis('off')

# os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
# if savefig: fig.savefig('Figures/Fig_map_v2.png',dpi=600);
# os.chdir(homedir)
