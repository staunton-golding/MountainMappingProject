# MountainMappingProject

## Description
Do you ever look at a mountain range and think "So pretty! What mountain is that?" I do, all the time (I'm horrible at geography). This repository was made to answer that question - it identifies peaks in user-uploaded photos. It is currently still in progress. Also currently only for use in Colorado, but will be expanded to the lower 48 of the US shortly (I have only queried NGIS for Colorado mountain peaks).

All code and necessary files are housed in the `mmp` folder. The mmp_main.ipynb notebook takes ~2-4 minutes to run.

## Project Overview
A high level overview of the pipeline (both completed and yet to be implemented):
  - Scrape photo metadata for coordinates of image
  - Download ~ +-1° lat and long DEM data centered around photo location (horizon line very rarely further than this)
  - Filter Colorado Peaks by those within the bounding box of the downloaded DEM data
  - Convert 3d cartesian DEM data to 2d adjust polar coordinates
  - "Unwrap" polar DEM, creating a rendered 360 view of ridgelines surrounding the site of the photo
  - Annotate rendered view with peaks, ignoring peaks hidden by larger mountains.
  - *NOTE - everything below is still to be done*
  - Train a u-net on image segmentation of mountain ranges (segmenting the mountains from the sky)
  - Deploy u-net, segment photograph
  - Delineate photo's ridgeline and panoramic rendered ridgeline
  - Register photo's ridgeline to the rendered ridgeline
  - Crop and inverse transform rendered ridgeline (and peak annotations) to photo
  - Return photo with annotations describing every mountain peak

Some general notes
  - Canny, otsu, waterfill, and semantic segmentation with grounding SAM for photo ridgeline segmentation all have issues (all tested).
    - Canny, otsu and produce inconsistent point clouds, and fail to ignore foreground trees.
    - Waterfill fails to produce sharp / correct ridgelines when clouds, snow or (sometimes) trees are present.
    - Semantic segmentation is slow and inconsistent (although sometimes excellent)
  - Currently have ~300 images of mountains/ridgelines (scraped from unsplash free-license images) with segmentation masks (produced with groundingSAM, refined with a gui)
    - Will train u-net with these images
  - Once model is trained and deployed, I'll add the scripts I used for scraping images, segmenting images with groundingSAM, attempting to segment / edge detect with other methods, and refining masks with a gui.

## Getting Started 
Dependencies for this repository include:
  - python=3.9 
  - gdal
  - pyproj
  - numpy
  - jupyter
  - matplotlib
  - scipy
  - geopandas
  - adjustText
  - pillow_heif
  - pillow
  - numba
  - ipython

I've supplied a .yaml file for use in creating a conda env for this project. To use, run this from the command line:

`conda env create -f mmp.yaml`

To activate the environemnt, run the following from the command line:

`conda activate mmp_env`

After activating, download this repository (used for downloading DEM data from The National Map): https://code.usgs.gov/gecsc/dem_getter. Place the dem_getter folder in your chosen directory (you really only need dem_getter.py).

To create a kernel from this environment, activate the environment, then run the following from the command line:

`python -m ipykernel install --name=mmp_env`

## Installation

you can either download a zip file, and unzip it to the folder of your choosing (this will extract a single folder called mmp_main), or use git to clone this repository with 
`git clone https://github.com/staunton-golding/MountainMappingProject.git`

## Usage
To use this repository, upload a photo of some pretty mountains, open mmp_main.ipynb, find where you are prompted to put the path to your image, and run each cell in the notebook.

## Future Plans
First, this is only the first half of the project. Right now, a 360 view of the surrounding ridgeline is rendered, with each visible peak being noted. Finishing the second half of this pipeline is the main priority. Those steps are included above.

Other plans include:
  - Add support for non-geotagged images
  - optimize panoramic rendering speeds
When I finish this project, I am sure there will be more things to improve.

I also suspect that I'll need to revamp the pseudo-polar transform from 3D DEM to an unwrapped 2D 360 panoramic rendering of the ridgeline. Going to investigate this after I complete first iteration of this pipeline.
