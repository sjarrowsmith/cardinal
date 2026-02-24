# Cardinal
Seismic and geoacoustic array processing

## Installing
To install and activate Cardinal:
Navigate to Cardinal directory
- conda env create -f cardinal_env.yml
- conda activate cardinal or source activate cardinal
- pip install tensorflow==2.19.0
- pip install cx_Oracle and pip install oracledb

If you encounter conflicts during install, run these first:
- conda config --add channels conda-forge
- conda config --add channels defaults
- conda config --set channel_priority flexible

## Run Cardinal
Navigate to Cardinal directory
## Jupyter Notebook
- jupyter nbclassic&
## Command-Line Client
- python run_cardinal.py --paramfile cl_cardinal_params.json --starttime 2015-01-03T13:47:00 --endtime 2015-01-03T13:49:00 # example earthquakes at PFO

## Examples
The notebook provided, 1_Bolide_Infrasound.ipynb, outlines how to implement Cardinal's algorithm on example bolide events.

## Algorithm
1. Segmentor
2. Adaptive Array
3. Array Processor
4. Aggregator
