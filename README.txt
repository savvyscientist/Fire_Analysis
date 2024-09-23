ANALYSIS AND EVALUATION OF pyrE

Description: Code to generate time series and seasonality graphs for observational products and pyrE. 
	     Code to generate netcdfs to plot maps on Panoply.
	     Code to align generated output in panels
             Code to regrid raw dataset into required resolution


Input Netcdf files: Input netcdf files must be in the format where dims are time,mask 
 where time is for monthly timely step and mask represents the mask value (GFED REGION)

File to run : /discover/nobackup/projects/giss_ana/users/ssenthil/code/Plotting.py 

How to run: python Plotting.py

1)User will be prompted to enter BA or Lightning to indicate if they want to analyse Lightning or Burned Area
2) Enter if you want to include observational product (GFED,WGLC etc)
3) if yes the product is loaded.
4)Enter the number of simulations to compare. List will include available simualtions for analysis.
5) Enter diagnostics to extract from netcdf. 
	i)BA has the option : BA_Mha, BA_Mha_shrub,BA_Mha_grass,BA_tree_Mha
       ii)Lightning : For comparison with observational product enter f_ignCG. 
      iii)For comparison between ignition and human, enter f_ignCG,f_ignHUMAN
      iv) For precipitation enter FLAMM_prec
6) edit outputpath and unit in code.

To include new dataset:
1) Add new netcdf path
2) Add model simualtions paths and new diagnostics to list
3) Add an additional 'if' cluase that specifies the files ot read for new analysis

MAPS
To run:
python maps.py

This will generate netcdfs in specified paths that can be plotted on panoply. 

PANELS

1) Edit source of files
2)Edit number of rows and columns
3) Edit output path.

REGRIDS)

To run: python <filename>

GUIDE FOR NETCDF GEN

1) To make netcdf files for observational and model simulation, follow structure in gfed_regrid.py in code/GFED/
2) Extract mask and apply it to each time step and store in a xarray with time,mask as dims



Note for GFED: for observation GFED product, the 15th mask region (all regions combimed) is a separate netcdf file so the time series and seasonality graphs for Total are generated separately in the code.

For any questions or doubts please email: sanjana.sk08@gmail.com
