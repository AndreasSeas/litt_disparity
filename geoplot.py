# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

conda activate flxsus
"""
import pandas as pd
import pgeocode
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

df=pd.read_csv('/Users/as822/Library/CloudStorage/Box-Box/McIntyreLab_Andreas/Writing/Manuscript-LITTDisparity/LITT_Zip_Code_Data.csv')
df["Latitude"]=np.nan
df["Longitude"]=np.nan

###
nomi = pgeocode.Nominatim('US')

for i in df.index:
    temp=nomi.query_postal_code(str(df.loc[i,"Zip Code"]))
    df.loc[i,"Latitude"]=temp.latitude
    df.loc[i,"Longitude"]=temp.longitude
    # print(temp)

dukedata=nomi.query_postal_code("27710")
lat_duke=dukedata.latitude
lon_duke=dukedata.longitude

# https://geopandas.org/en/stable/docs/user_guide/projections.html
import geopandas
import geodatasets
from shapely.geometry import Point

usa = geopandas.read_file(geodatasets.get_path('geoda.natregimes'))
# world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

fig, ax=plt.subplots(figsize=(16,8),ncols=1,nrows=1,)
# world.plot(ax=ax,color=(0.9,0.9,0.9))
# world.boundary.plot(color=(0.3,0.3,0.3),linewidth=0.5,ax=ax)
usa.plot(ax=ax,color=(0.9,0.9,0.9))
usa.boundary.plot(ax=ax,color=(0.5,0.5,0.5),linewidth=0.1)

ax.plot(df.Longitude,df.Latitude,'.k');
ax.plot(lon_duke,lat_duke,'or');

df=df.rename(columns={'Race: 0 - White, 1 - Black, 2 - Asian, 3 - American Indian or AK native, 4 - Pacific Islander/Native Hawaian, 5 - Other/Multiracial, 6 - NR/Declined': "Race"})

# sns.kdeplot(
#     data=df, x="Longitude", y="Latitude", fill=True, ax=ax,palette="tab10",
#     # hue="Race"
# )

# plot heatmap

# ax = geoplot.kdeplot(
#     collisions.head(1000),
#     clip=boroughs.geometry,
#     shade=True,
#     cmap="Reds",
#     projection=geoplot.crs.AlbersEqualArea(),
# )
# geoplot.polyplot(boroughs, ax=ax, zorder=1)

ax.set_xlim(-85,-72);
ax.set_ylim(30,40);

sys.exit()

## plot


import geopandas
from shapely.geometry import Point


# https://onelinerhub.com/python-matplotlib/how-to-fill-countries-with-colors-using-world-map

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
# from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/

all_coords=pd.DataFrame(data=None,columns=['Unique ID','Geometry']);
for df in dflist:
    df['Geometry']=np.nan;
    for i in df.index:
        df.loc[i,'Geometry']=Point(df.loc[i,'Longitude'],df.loc[i,'Latitude'])
        
    all_coords=pd.concat([all_coords,df[['Unique ID','Geometry']]])

all_coords=all_coords.reset_index(drop=True);
all_coords['Country']=np.nan

# go thru each country
for i in range(len(world)):
    # print(world.loc[i,'name'])
    geom=world.loc[i,'geometry']
    for j in range(len(all_coords)):
        pt=all_coords.loc[j,'Geometry']
        if geom.contains(pt):
            all_coords.loc[j,'Country']=world.loc[i,'name'];
        # print(geom.contains(pt))
        
# make sure to only keep unique countries
world['count']=0;
unique_coords=pd.DataFrame(data=None,columns=['Country'],index=all_coords['Unique ID'].unique());

countmult=0
for i, idx in enumerate(unique_coords.index):

    ctemp=all_coords.loc[all_coords['Unique ID']==idx,'Country']
    
    for loc in ctemp.unique():
        world.loc[world.name == loc,'count']=world.loc[world.name == loc,'count']+1
    
    if len(ctemp.unique())==1:
        unique_coords.loc[idx,'Country']=ctemp.iloc[0]
        
    else:
        unique_coords.loc[idx,'Country']="multiple"
        # print(ctemp)
        # countmult=countmult+1

# world['count']=0;

# polygon.contains(point)

plt.rcParams['font.size'] = '12'
plt.rcParams['font.family'] = 'serif'

fig, ax=plt.subplots(figsize=(16,8),ncols=1,nrows=1,)
world.plot(column='count',legend=True,ax=ax,vmin=0,vmax=10,cmap='Blues',
           legend_kwds={"label": "# Participants", "orientation": "vertical"},)

# import matplotlib.colors as colors
# world.plot(column='count',legend=True,ax=ax,vmin=1,vmax=10,cmap='Blues',
#            norm=colors.LogNorm(vmin=1,vmax=world['count'].max()))
world[world.name == 'United States of America'].plot(color=[0.5,0.5,0.5],ax=ax)

world.boundary.plot(color=[0.5,0.5,0.5],linewidth=0.5,ax=ax,)
# plt.scatter(all_coords.Geometry)

# world[world.name == 'Ukraine'].plot(color='yellow',ax=ax)

# plt.show()

# markerlist=['x','o','x','o','.'];
colorlist=['#e70843','#90052a','#658b6c','#094614','#391164'];

for i,df_ in enumerate(dflist):
    # print(df_.shape)
    x_=df_.loc[:,'Longitude']
    y_=df_.loc[:,'Latitude']
    # ax.plot(x_,y_,markerlist[i],label=dfname[i],);
    ax.scatter(x_,y_,50,colorlist[i],alpha=0.3,label=dfname[i],)

ax.legend(loc='lower center',ncol=5,bbox_to_anchor=[0.5,-0.1])


plt.tight_layout()
plt.axis('off')

os.chdir('/Users/as822/Library/CloudStorage/Box-Box/!Research/FLXSUS/')
if savefig: fig.savefig('Figures/Fig_map_v2.png',dpi=600);
os.chdir(homedir)
