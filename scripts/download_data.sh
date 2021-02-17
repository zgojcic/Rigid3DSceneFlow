#!/usr/bin/env bash

DATASET=$1

function download() {
    if [ ! -d "data" ]; then
        mkdir -p "data"
    fi
    cd data 

    url="https://share.phys.ethz.ch/~gsg/weakly_supervised_3D_rigid_scene_flow/data/"
   
    extension=".tar" 
    
    wget --no-check-certificate --show-progress "$url$DATASET$extension"
    tar -xf "$DATASET$extension"
    rm "$DATASET$extension"
    cd ../


}


function download_all() {

    for DATASET in "stereo_kitti" "lidar_kitti" "semantic_kitti" "waymo_open" "fyling_things_3d"
    do
        download
    done	
}

function main() {
    if [ -z "$DATASET" ]; 
    then
        echo "No dataset selcted. All data will be downloaded"
        download_all
    elif [ "$DATASET" == "stereo_kitti" ]  || [ "$DATASET" == "lidar_kitti" ] || [ "$DATASET" == "semantic_kitti" ] || [ "$DATASET" == "waymo_open" ] || [ "$DATASET" == "fyling_things_3d" ] 
    then
        download
    else
        echo "Wrong dataset selected must be one of  [stereo_kitti, lidar_kitti, waymo_open, semantic_kitti, fyling_things_3d]."
    fi
}

main;
