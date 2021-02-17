#!/usr/bin/env bash


function download_models() {
    if [ ! -d "logs" ]; then
        mkdir -p "logs"
    fi
    cd logs 

    url="https://share.phys.ethz.ch/~gsg/weakly_supervised_3D_rigid_scene_flow/pretrained_models/"
   
    model_file="pretrained_models_ablations.tar" 
    
    wget --no-check-certificate --show-progress "$url$model_file"
    tar -xf "$model_file"
    rm "$model_file"
    cd ../
}


download_models;
