from netCDF4 import  Dataset
import numpy as np
import pandas as pd

#reading in the netCDF4 file path+filename 
data1 = Dataset(r'/discover/nobackup/kmezuman/E6TpyrEPD/ANN2010.taijE6TpyrEPD.nc', 'r')
data2 = Dataset(r'/discover/nobackup/kmezuman/E6TpyrEPD/ANN2010.aijE6TpyrEPD.nc', 'r') 
data3 = Dataset(r'/discover/nobackup/kmezuman/CCycle_E6obioF40/ANN2009.taijCCycle_E6obioF40.nc', 'r')
#displaying all the names of the varibles  
#print(data.variables.keys())
#accessing the varaibles needed 
 
CO2n_pyrE_src_hemis = data3.variables['CO2n_pyrE_src_hemis']
#print(CO2n_pyrE_src_hemis[2,])
CO2n_pyrE_src = data3.variables['CO2n_pyrE_src']
CO2n_pyrE_src_units = CO2n_pyrE_src.units
#print(CO2n_pyrE_src_units)

#CO2n_biomass_src_hemis = data1.variables['CO2n_biomass_src_hemis']
#print(CO2n_biomass_src_hemis[2,])

#CO_biomass_src_hemis = data1.variables['CO_biomass_src_hemis']
#print(CO_biomass_src_hemis[2,])

CO2n_Total_Mass_hemis = data1.variables['CO2n_Total_Mass_hemis']
#print(CO2n_Total_Mass_hemis[2,])
CO2n_Total_Mass = data1.variables['CO2n_Total_Mass']
CO2n_Total_Mass_units = CO2n_Total_Mass.units
#print(CO2n_Total_Mass_units)


#CO_Total_Mass_hemis = data1.variables['CO_Total_Mass_hemis']
#print(CO_Total_Mass_hemis[2,])

C_lab_hemis = data2.variables['C_lab_hemis']
#print(C_lab_hemis[2,])
C_lab = data2.variables['C_lab']
C_lab_units = C_lab.units
#print(C_lab_units)

soilCpool_hemis = data2.variables['soilCpool_hemis']
#print(soilCpool_hemis[2,])
soilCpool = data2.variables['soilCpool']
soilCpool_units = soilCpool.units
#print(soilCpool_units)

gpp_hemis = data2.variables['gpp_hemis']
#print(gpp_hemis[2,])
gpp = data2.variables['gpp']
gpp_units = gpp.units
#print(gpp_units)

rauto_hemis = data2.variables['rauto_hemis']
#print(rauto_hemis[2,])
rauto = data2.variables['rauto']
rauto_units = rauto.units
#print(rauto_units)

soilresp_hemis = data2.variables['soilresp_hemis']
#print(soilresp_hemis[2,])
soilresp = data2.variables['soilresp']
soilresp_units = soilresp.units
#print(soilresp_units)

ecvf_hemis = data2.variables['ecvf_hemis']
#print(ecvf_hemis[2,])
ecvf = data2.variables['ecvf']
ecvf_units = ecvf.units
#print(ecvf_units)

net_flex_down = gpp_hemis[2,] - rauto_hemis[2,] - soilresp_hemis[2,] - ecvf_hemis[2,]
df = pd.DataFrame({
    'Varaiable':['Biomass emissions','Atmospheric column load', 'Vegatation Carbon', 'Soil Carbon','GPP','Autotrophic Respiration', 'Soil Respiration', 'EXCESS C FLUX DUE TO VEG FRACTIONS CHANGE','   Net Flux = GPP - rauto - soilresp -ecvf', 'Ocean Carbon'],
   # 'CO': [(CO_biomass_src_hemis[2,]),(CO_Total_Mass_hemis[2,]), '-','-', '-'],  
   'Units':[(CO2n_pyrE_src_units),(CO2n_Total_Mass_units),(C_lab_units),(soilCpool_units),(gpp_units),(rauto_units),(soilresp_units),(ecvf_units),(ecvf_units),'-'],
   'CO2':[(CO2n_pyrE_src_hemis[2,]),(CO2n_Total_Mass_hemis[2,]),'-','-','-','-','-','-','-','-'],
   'Total Carbon': ['-','-',(C_lab_hemis[2,]),(soilCpool_hemis[2,]),(gpp_hemis[2,]),(rauto_hemis[2,]),(soilresp_hemis[2,]),(ecvf_hemis[2,]), net_flex_down,'-']
    })
data = print(df)
formatted_table = df.to_string(index = False)

output_file_path = 'carbon_budget.txt'
with open(output_file_path, 'w') as file:
    file.write(formatted_table)

