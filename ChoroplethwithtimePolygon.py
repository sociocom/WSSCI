import jismesh.utils as ju
import geopandas as gpd
import pandas as pd
import glob
from geojson import Polygon,Feature,FeatureCollection
import folium
from folium.plugins import TimeSliderChoropleth
import branca.colormap as cm
import numpy as np

all_files = glob.glob("./20200316/pros0123-0305/*.txt")
all_files.sort()
csvs = []
c_name = []
for filename in all_files:
    readtables=pd.read_table(filename,header = None,usecols=[0,2])
    readt = readtables.astype(int)
    grouped=readt.groupby(0).sum()
    csvs.append(grouped)
    c_name.append(filename.replace('./20200316/pros0123-0305/cov_', '').replace('.txt', ''))
df = pd.concat(csvs,axis=1)
df.sort_index()
df.columns = c_name
df2=df.fillna(0).astype(int)


index_list=list(df2.index)
Mesh = []
counter = 0
for mesh in index_list:
    meshnum = int(mesh)
    lat_sw, lon_sw = ju.to_meshpoint(meshnum, lat_multiplier=0, lon_multiplier=0)
    lat_se, lon_se = ju.to_meshpoint(meshnum, lat_multiplier=0, lon_multiplier=1)
    lat_nw, lon_nw = ju.to_meshpoint(meshnum, lat_multiplier=1, lon_multiplier=0)
    lat_ne, lon_ne = ju.to_meshpoint(meshnum, lat_multiplier=1, lon_multiplier=1)
    meshst = str(mesh)
    Mesh.append(Polygon([[(lon_ne, lat_ne), (lon_nw, lat_nw), (lon_sw, lat_sw), (lon_se, lat_se)]]))
Meshs = gpd.GeoDataFrame({'id': index_list, 'geometry': Mesh})
src = Meshs.to_json()

linear = cm.LinearColormap(['white','blue','aqua','lime','yellow','orange','red'],index=[0,5,10,50,100,500],vmin=0,vmax=600).to_step(200)
linear.caption = 'Suspicious User of infected'

Heatdata_7d = []

for yy in range(0, len(df2.index)):
    Heatdata_7d.append([])

for index_name, item in df2.iterrows():

    meshnum = int(index_name*10)

    for z in range(0,len(df2.columns)-6):
        diff_7d = df2.iloc[counter,z:z+6].sum()
        Heatdata_7d[counter].append(diff_7d)

    counter += 1

df_7d = pd.DataFrame(Heatdata_7d)

nn = len(df_7d.columns)
dayslist = []
for passday in range(nn):
    dayslist.append(1580223600+(passday*86400))
dayslist=np.array(dayslist)
dayslist=dayslist.astype('int32')

data = {}
recordindex = 0

for pref in df_7d.index:
    d = pd.DataFrame(
        {'color':df_7d.loc[pref,:].apply(linear).tolist(),'opacity':np.full(nn,0.7),},
    index=dayslist
    )
    data[recordindex] = d
    recordindex += 1
styledict = {pref: v.to_dict(orient='index') for pref, v in data.items()}

m = folium.Map(location=[43.067392,141.351137],zoom_start=5,tiles='cartodbpositron')
g = TimeSliderChoropleth(src,styledict=styledict,overlay=True).add_to(m)
folium.Marker(location = [43.804788, 143.831717],popup = "北見綜合卸センター　Event 2/13-15 FirstOutbreak　2/16　FirstReport 2/22 Cluster 2/27").add_to(m)
#folium.Marker(location = [43.054273, 141.353794],popup = "SingSingSing", ).add_to(m)
folium.Marker(location = [43.060967, 141.348851],popup = "さっぽろ雪まつり　Event2/1-12 FirstOutbreak 2/8 FirstReport 2/19").add_to(m)
#folium.Polygon(locations=[[43.0,141.5],[43.0,141.25],[43.1666,141.25],[43.1666,141.5]],color = 'black',weight=3).add_to(m)
#folium.Polygon(locations=[[43.9166,143.625],[43.6666,143.625],[43.6666,144.0],[43.9166,144.0]],color = 'black',weight=3).add_to(m)
m.add_child(linear)
m.save('Prospective_3D.html')
