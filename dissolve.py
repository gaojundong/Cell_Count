import os
import json
from pathlib import Path
import numpy as np
import geopandas as gp
import networkx as nx
from prometheus_client import Counter
import tqdm

from math import ceil
from shapely.geometry import Polygon,LineString
from shapely.ops import unary_union
from matplotlib import pyplot as plt
import shapely.ops as so

slide_list = []

def get_iou(obj1,obj2):
    inter = obj1.geometry.intersection(obj2.geometry)
    inter_area = inter.area
    union = obj1.geometry.union(obj2.geometry)
    union_area = union.area
    IoU = inter_area/union_area
    return IoU, inter_area/obj1.geometry.area, inter_area/obj2.geometry.area


def delete_overlap():

    deleted = set()

    for (idx_left,idx_right) in tqdm.tqdm(vertices, desc = 'delete overlap'):
        if idx_left == idx_right:continue

        obj1 = gpd_geoms.iloc[idx_left]
        obj2 = gpd_geoms.iloc[idx_right]
        IoU = get_iou(obj1,obj2)
        if IoU > 0.88:
            if float(obj1.at['measurements'][0]['prop']) > float(obj2.at['measurements'][0]['prop']):
                # gpd_geoms.drop(index=idx_right)
                deleted.add(idx_right)
            else:
                # gpd_geoms.drop(index=idx_left)
                deleted.add(idx_left)

    return deleted 



def merge_adjcent(obj1,obj2):
    
    hw = 8
    
    bbox1 = obj1.geometry.bounds
    bbox2 = obj2.geometry.bounds
    
    xmin,ymin,xmax,ymax = min(bbox1[0],bbox2[0]), min(bbox1[1],bbox2[1]),max(bbox1[2],bbox2[2]),max(bbox1[3],bbox2[3])

    position_x = ceil(xmin/256)*256
    position_y = ceil(ymin/256)*256
    
    
    while position_x < xmax:

        square = Polygon([[position_x-hw,ymin],[position_x+hw,ymin],
                          [position_x+hw,ymax],[position_x-hw,ymax]])

        L1 = square.intersection(obj1.geometry).bounds
        L2 = square.intersection(obj2.geometry).bounds
        
        if L1 and L2 and max(bbox1[2],bbox2[2])>position_x+hw and min(bbox1[0],bbox2[0])<position_x-hw:
            
            if L1[3]-L2[1]<= 0 or L2[3]-L1[1]<=0:
                return False

            line = [min(L1[1],L2[1]),max(L1[3],L2[3])]
            length_overlap = min(L1[3]-L2[1],L2[3]-L1[1])

            p = length_overlap/(line[1]-line[0])

            if p > 0.6:
                return True
        
        position_x += 256
            
    
    while position_y < ymax:

        square = Polygon([[xmin, position_y - hw],[xmax,position_y - hw],
                  [xmax,position_y + hw],[xmin, position_y + hw]])

        L1 = square.intersection(obj1.geometry).bounds
        L2 = square.intersection(obj2.geometry).bounds

        if not L1 or not L2:
            _ = 0
            
        elif max(bbox1[3],bbox2[3])< position_y+hw or min(bbox1[1],bbox2[1])>position_y-hw:
            _ = 0
        
        else:
        
            if L1[2]-L2[0]<= 0 or L2[2]-L1[0]<=0:
                return False

            line = [min(L1[0],L2[0]),max(L1[2],L2[2])]
            length_overlap = min(L1[2]-L2[0], L2[2]-L1[0])

            p=length_overlap/(line[1]-line[0])

            if p > 0.6:
                return True
        
        position_y += 256
    
    return False

def process_multiploygon(new_geom):
    
    l = len(new_geom)
    area_list = [new_geom.area for geom in new_geom]
    area_rate = [a/max(area_list) for a in area_list]
    all_geoms = [i for i in range(l) if area_rate[i] > 0.3 and area_list[i] > 200 ]
    valid = len(all_geoms)
    if not valid:
        new_geom = new_geom.geoms[0]
        return new_geom
    if valid == 1:
        new_geom = new_geom.geoms[all_geoms[0]]
    elif valid == 2:
        dist = new_geom.geoms[all_geoms[0]].distance(new_geom.geoms[all_geoms[1]])
        if dist > 30 :
            new_geom = new_geom.geoms[area_list.index(max(area_list))]
            valid = 1
        else:
            new_geom = unary_union([new_geom.geoms[0].buffer(20),new_geom.geoms[1].buffer(20)]).buffer(-18)
            valid = 1
            if new_geom.type != 'Polygon':
                valid = len(new_geom.geoms)
    if valid >=2:
        new_geom = new_geom.geoms[area_list.index(max(area_list))]
    
    return new_geom


if __name__ == '__main__':
    
    record = open('/scratch/gaojud96/inference_data/record.txt','a+')

    for id in slide_list:

        file_path = f'/scratch/gaojud96/inference_data/{id}/predict_result.json'
        with open(file_path) as f:
            bbox = json.load(f)

        geom_list = bbox
        gpd_geoms = gp.GeoDataFrame.from_features(geom_list)
        gpd_geoms.loc[~gpd_geoms.is_valid,'geometry']= gpd_geoms.loc[~gpd_geoms.is_valid, 'geometry'].buffer(0)
        gpd_geoms.loc[gpd_geoms.is_valid,'geometry']= gpd_geoms.loc[gpd_geoms.is_valid, 'geometry'].buffer(4)

        polygon_connections = gp.sjoin_nearest(gpd_geoms, gpd_geoms, how='left', max_distance = 8,distance_col='distance')
        polygons_connected = polygon_connections[~polygon_connections['index_right'].isna()]
        vertices = [(a, int(b)) for (a, b) in zip(
            polygons_connected.index.values, polygons_connected['index_right'].values)]
        gpd_geoms.loc[gpd_geoms.is_valid,'geometry']= gpd_geoms.loc[gpd_geoms.is_valid, 'geometry'].buffer(-4)

        G = nx.Graph()
        dissolved_list = []
        deleted_list = []

        for (id_1, id_2) in tqdm.tqdm(vertices, desc = 'built graph'):

            if id_1 == id_2: continue
            IoU, iou_1, iou_2 = get_iou(gpd_geoms.iloc[id_1],gpd_geoms.iloc[id_2])
            if merge_adjcent(gpd_geoms.iloc[id_1],gpd_geoms.iloc[id_2]):
                G.add_edge(id_1,id_2)
        
        for group in tqdm.tqdm(list(nx.connected_components(G)),desc = 'dissolve'):
            
            #union
            geoms = []
            max_areas = 0
            for idx in group:
                geom = gpd_geoms.loc[idx]['geometry']
                tmp = geom.area
                if tmp > max_areas:
                    max_areas = tmp
                    class_idx = idx
                geoms.append(geom.buffer(16))
                
            new_geom = unary_union(geoms).buffer(-16)
            
            #delete multiploygon
            if new_geom.type != 'Polygon':
                new_geom = process_multiploygon(new_geom)
            
            #save
            dissolved_list.append(
                {
                'geometry':new_geom,
                'classification':{
                    'name': gpd_geoms.loc[class_idx]['classification']['name']
                    },
                'measurements':[]
            }
            )
            deleted_list.extend(group)

        deleted_list = list(set(deleted_list))
        dissolved_gpd = gp.GeoDataFrame(dissolved_list)
        dissolved_polygons_gpd = gpd_geoms.drop(deleted_list).append(dissolved_gpd,ignore_index=True)
    
        save_path = f'/scratch/gaojud96/inference_data/{id}/result/'+'dissolved.txt'
        f.close()

    record.close()
