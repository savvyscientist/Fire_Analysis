#This script plots a time series of one year of data (seasonality)
#of specified emissions sources from two simulations and also calculates
#the difference between the two
import numpy as np 
import netCDF4 as nc 
import matplotlib.pyplot as plt

year_i = 1996
year_e = 1996
months=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
n_months=len(months)
years = np.arange(year_i, year_e+1)
n_years = (year_e - year_i + 1) 
species=['NOx','OCB','BCB','NH3','SO2','Alkenes','Paraffin','CO']
sectors = [ "pyrE_src_hemis", "biomass_src_hemis"]
simulation=['E6TomaF40intpyrEtest','E6TomaF40intpyrEtest2']
legend_array= ["pyrE","defo", "biomass","pyrE+defo"]
color_array = ["black","blue","red","magenta","orange","green"]

n_sectors = len(sectors)
#Initalize Data array 
data = np.zeros((n_sectors*len(simulation), n_months))


#read axyp 
filename = "/discover/nobackup/kmezuman/E6TomaF40intpyrEtest/JAN1996.taijE6TomaF40intpyrEtest.nc"
ds = nc.Dataset(filename, "r")
area = ds.variables["axyp"]
earth_surface_area = np.sum(area)
seconds_in_year = 60.0*60.0*24.0*365.0 
kg_to_g = 10.0**3
for sp, specie in enumerate(species):
    for m, mon in enumerate(months):
        l_ind=0
        for s, sim in enumerate(simulation):
            #Construct file name 
            filename = "/discover/nobackup/kmezuman/"+sim+"/"+mon+"1996.taij"+sim+".nc"
            try: 
                ds = nc.Dataset(filename, 'r') 
                for sec, sector in enumerate(sectors): 
                    if sector == "pyrE_src_hemis" and sim == 'E6TomaF40intpyrEtest2':
                        continue 
                    diag = specie + '_' + sector
                    hemis_val = ds.variables[diag]
                    global_val = hemis_val[2,]
                    #if sector == "CO2n_pyrE_src_hemis" or sector == "CO2n_biomass_src_hemis":
                    #    unit_factor = 10.0**-12
                    #else:
                    #    unit_factor = 10.0**-15
                    data[l_ind, m] = global_val * earth_surface_area * seconds_in_year * kg_to_g #* unit_factor
                    l_ind += 1 
                ds.close()
            except FileNotFoundError: 
                print(f"File {filename} not found.")
            except Exception as e: 
                print(f"Error reading from {filename}: {str(e)}")
        data[-1, m] = data[0, m] + data[1, m]
    
#plotting 
#label_array = []
#for s, sector in enumerate(sectors):
 #   string = sector.replace("CO2n_", "") + "[Pg]"
#  label_array.append(string) 
    plt.figure(figsize=(12,8))
    for l, legend in enumerate(legend_array):
        plt.plot(months, data[l, :], label= legend, marker = 'o', color=color_array[l])
    plt.title(specie+f" Emissions by Sector ({year_i} - {year_e})")
    plt.xlabel("Month")
    plt.ylabel(specie+" Emissions [Pg]")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/discover/nobackup/kmezuman/plots/fire_repository/Develpment/'+specie+'_emissions_by_sector.eps')
    #plt.show()
