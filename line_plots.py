from glob import glob
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import matplotlib.dates as mdates
import itertools
import seaborn as sns

gfed_cover_labels = {
        0: "Ocean",
        1: "BONA",
        2: "TENA",
        3: "CEAM",
        4: "NHSA",
        5: "SHSA",
        6: "EURO",
        7: "MIDE",
        8: "NHAF",
        9: "SHAF",
        10: "BOAS",
        11: "CEAS",
        12: "SEAS",
        13: "EQAS",
        14: "AUST",   15: "Total"
    }




land_cover_labels = {
        0: "Water",
        1: "Boreal forest",
        2: "Tropical forest",
        3: "Temperate forest",
        4: "Temperate mosaic",
        5: "Tropical shrublands",
        6: "Temperate shrublands",
        7: "Temperate grasslands",
        8: "Woody savanna",
        9: "Open savanna",
        10: "Tropical grasslands",
        11: "Wetlands",
        12: "",
        13: "Urban",
        14: "",
        15: "Snow and Ice",
        16: "Barren",
        17: "Sparse boreal forest",
        18: "Tundra",
        19: ""
    }


#read from netcdf
def Data_TS_Season_Regional(opened_datasets, diagnostics_list,val,unit,output_path):
    

    
    #open datasets as variables
    #open diagnsotic (model only)
    #gfed check cond
    # do ilct stuf or else just plot the model stuff
    
    distinct_colors = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
    '#FFA500', '#008000', '#800080', '#008080', '#800000', '#000080',
    '#808000', '#800080', '#FF6347', '#00CED1', '#FF4500', '#DA70D6',
    '#32CD32', '#FF69B4', '#8B008B', '#7FFF00', '#FFD700', '#20B2AA',
    '#B22222', '#FF7F50', '#00FA9A', '#4B0082', '#ADFF2F', '#F08080'
]
    
    


    mask_lis =  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    
   
    for i in mask_lis:
        fig, ax = plt.subplots(figsize=(12, 8),tight_layout=True)
        
        color_map = plt.get_cmap('tab20')
        
        color_iter = iter(distinct_colors)
      
       
        for idx in opened_datasets.keys():
            
            if idx == 'gfed':
                fig_t, ax_t = plt.subplots(figsize=(12, 8),tight_layout=True)
                color_map = plt.get_cmap('tab20')
                gd = opened_datasets['gfed'].data_vars
                gfed_diag = list(gd.keys())
                
                time_dt =  opened_datasets[idx]['time'].values
                time_gf = pd.to_datetime(time_dt) 
                
                
                unique_land_cover_types = opened_datasets['gfed']['ilct'].values

                    # Iterate through each ilct type
                #if 'Total' in target_data.variables and 'Norm' in target_data.variables:
                
                if i == 15:
                    d = xr.open_dataset('/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/15th_region.nc')
                    time_dt =  d['time'].values
                    time_15 = pd.to_datetime(time_dt)
                    gd = d.data_vars
                    gfed_diag = list(gd.keys())
                    for ilct_idx, ilct in enumerate(unique_land_cover_types):
                        total_burn_area_ilct_15 = d[gfed_diag[0]][:,ilct_idx]

                        ax_t.plot(time_15, total_burn_area_ilct_15, label=f"iLCT {ilct}: {land_cover_labels[ilct]}", color=color_map(ilct_idx))
                    for diag_idx,diag_name in enumerate(gfed_diag[1:]):
                        color = next(color_iter)

                        ax.plot(time_15, d[diag_name][:], label=f'{diag_name}', color=color, linewidth=1.5)
                    total_burn_area_total_ilct = np.sum(d[gfed_diag[0]][:, :], axis=-1)
                    ax_t.plot(time_gf, total_burn_area_total_ilct, label="Total ILCT", color='black', linewidth=1.5) 
            
                    
                    
                    
                else:
                    for ilct_idx, ilct in enumerate(unique_land_cover_types):
                        total_burn_area_ilct = opened_datasets[idx][gfed_diag[0]][:, i,ilct_idx]

                        ax_t.plot(time_gf, total_burn_area_ilct, label=f"iLCT {ilct}: {land_cover_labels[ilct]}", color=color_map(ilct_idx))

                    for diag_idx,diag_name in enumerate(gfed_diag[1:]):
                        color = next(color_iter)

                        ax.plot(time_gf, opened_datasets[idx][gfed_diag[diag_idx+1]][:,i], label=f'{diag_name}', color=color, linewidth=1.5)
                    total_burn_area_total_ilct = np.sum(opened_datasets[idx][gfed_diag[0]][:,i, :], axis=-1)
                ax_t.plot(time_gf, total_burn_area_total_ilct, label="Total ILCT", color='black', linewidth=1.5) 
                ax.plot(time_gf, total_burn_area_total_ilct, label="NAT", color='black', linewidth=1.5) 
                ax_t.set_xlabel('Time')
                ax_t.set_ylabel('Total Burned Area [Mha]')
                ax_t.set_title(f"Total Burned Area for Mask Value {gfed_cover_labels[i]} and Different iLCT Types")
                ax_t.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=5)
                ax_t.xaxis.set_major_locator(mdates.YearLocator())
                ax_t.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                #plt.xticks(rotation=45)
                ax_t.tick_params(axis='x', rotation=45)
                plt.subplots_adjust(hspace=1.5)
                file_name_t = f"NATBA_TS_gfed{gfed_cover_labels[i]}_1997_2020.png"
                file_path_t = os.path.join(output_path, file_name_t)
                plt.savefig(file_path_t, dpi=300, bbox_inches='tight')
                plt.close()
                
            elif idx == 'wglc':
                print("wglc")
                time_dt =  opened_datasets[idx]['time'].values
                time = pd.to_datetime(time_dt)
                
                wg = opened_datasets['wglc'].data_vars
                wd_diag = list(wg.keys())
                for diag_idx,diag_name in enumerate(wd_diag):

                    
                    color = next(color_iter)
                    ax.plot(time, opened_datasets[idx][diag_name][:,i], label="Lightning Density WGLC", color=color, linewidth=1.5)
                
                
                
            else:

                for idn,diag in enumerate(diagnostics_list):
                    time_dt =  opened_datasets[idx]['time'].values
                    time = pd.to_datetime(time_dt)
                    #print(opened_datasets[idx][diag][:, i])


                    color = next(color_iter)
                    ax.plot(time, opened_datasets[idx][diag][:, i], label=f'{idx} {diag}', color=color, linewidth=1.5)
               
        
      

        ax.set_xlabel('Time')
        ax.set_ylabel(f'Total {val} {unit}')
        ax.set_title(f"Total {val} for Mask Value {gfed_cover_labels[i]}")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=5)






                 # Set x-axis tick positions to one year interval
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            #plt.xticks(rotation=45)
        ax.tick_params(axis='x', rotation=45)
        plt.subplots_adjust(hspace=1.5)






            #BA_TS_<region name>_<startyear>_<endyear>



        file_name = f"{val}_{idx}_TS_{gfed_cover_labels[i]}_1997_2020.png"
        file_path = os.path.join(output_path, file_name)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the first figure after saving
        print("Complete ",file_name)

        #plt.show()

        
#seasonality
def Data_TS_Season_model_gfed(opened_datasets, diagnostics_list,val,unit,output_path):
    
    distinct_colors = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
    '#FFA500', '#008000', '#800080', '#008080', '#800000', '#000080',
    '#808000', '#800080', '#FF6347', '#00CED1', '#FF4500', '#DA70D6',
    '#32CD32', '#FF69B4', '#8B008B', '#7FFF00', '#FFD700', '#20B2AA',
    '#B22222', '#FF7F50', '#00FA9A', '#4B0082', '#ADFF2F', '#F08080'
]
    


    mask_lis =  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    
    for i in mask_lis:
        fig, ax = plt.subplots(figsize=(12, 8),tight_layout=True)
        
        color_map = plt.get_cmap('tab20')
        
        color_iter = iter(distinct_colors)
      
       
        for idx in opened_datasets.keys():
            if idx == 'gfed':
                fig_t, ax_t = plt.subplots(figsize=(12, 8),tight_layout=True)
                color_map = plt.get_cmap('tab20')
                gd = opened_datasets['gfed'].data_vars
                gfed_diag = list(gd.keys())
                
                time_dt =  opened_datasets[idx]['time'].values
                time_gf = pd.to_datetime(time_dt) 
                
                
                unique_land_cover_types = opened_datasets['gfed']['ilct'].values
                
                
                if i==15:
                    std_error_15 = np.zeros(12)
                    plot_std = np.zeros(12)
                   
                    monthly_burn_15 = np.zeros((len(unique_land_cover_types), len(months)))
                    monthly_burn_count_15 = np.zeros((len(unique_land_cover_types), len(months)))
                    d = xr.open_dataset('/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/15th_region.nc')
                    time_dt =  d['time'].values
                    time_15 = pd.to_datetime(time_dt) 
                    gd = d.data_vars
                    gfed_diag = list(gd.keys())
                    for ilct_idx, ilct in enumerate(unique_land_cover_types):
                        total_burn_area_ilct = d[gfed_diag[0]][:,ilct_idx]
                        for month in range(len(months)):
               
                            monthly_burn[ilct_idx, month] = np.mean(total_burn_area_ilct[month::12])
                            monthly_burn_count[ilct_idx,month] = np.count_nonzero(total_burn_area_ilct[month::12])

                            std_error[month] = np.std(total_burn_area_ilct[month::12]) / np.sqrt(np.count_nonzero(monthly_burn_count[ilct_idx]))
                            if ilct_idx == 9:
                               plot_std[month]=std_error[month]
                        # Plot the burn area line for each land cover type
                        ax_t.plot(months, monthly_burn[ilct_idx], label=f"iLCT {ilct}: {land_cover_labels[ilct]}", color=color_map(ilct_idx))

                        ax_t.errorbar(months, monthly_burn[ilct_idx], yerr=std_error, fmt='none', capsize=9, color=color_map(ilct_idx),
                                    elinewidth=1)
                  
                    total_burned_across_ilct = np.sum(monthly_burn[1:,:], axis=0)
                   
                    # Plot the total burned area across all ilct types
                    ax_t.plot(months, total_burned_across_ilct, label="Total iLCT", color='black')
                 
                    #ax_t.errorbar(months, total_burned_across_ilct, yerr=total_burned_std_error, fmt='o', capsize=5, color='black', label="Error Bars")
                    #ax.plot(months, total_burned_across_ilct, label="NAT", color='black')
                    
                    # Plot the total burned area across all ilct types
                    ax_t.plot(months, total_burned_across_ilct, label="Total iLCT", color='black')
                 
                    #ax_t.errorbar(months, total_burned_across_ilct, yerr=total_burned_std_error, fmt='o', capsize=5, color='black', label="Error Bars")
                    ax.plot(months, total_burned_across_ilct, label="GFED5", color='black')
                    print(plot_std)
                    ax.errorbar(months, total_burned_across_ilct, yerr=plot_std, fmt='none', capsize=9, color='black',elinewidth=1)
                  
                    total_burned_across_ilct = np.sum(monthly_burn[1:,:], axis=0)
                    print(total_burned_across_ilct.shape)
                   
                    # Plot the total burned area across all ilct types
                    ax_t.plot(months, total_burned_across_ilct, label="Total iLCT", color='black')
                 
                    #ax_t.errorbar(months, total_burned_across_ilct, yerr=total_burned_
                    ax_t.set_xlabel('Time')
                    ax_t.set_ylabel(f'Total Burned Area [{unit}]')
                    ax_t.set_title(f"Total Burned Area for Mask Value {gfed_cover_labels[i]} and Different iLCT Types")
                    ax_t.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=5)
                    ax_t.tick_params(axis='x', rotation=45)
       
                    plt.subplots_adjust(hspace=1.5)
                    file_name = f"NATBA_SEASON_{gfed_cover_labels[i]}_1997_2020.png"
                    file_path = os.path.join(output_path, file_name)
                    plt.grid(True)
                    plt.savefig(file_path, dpi=300, bbox_inches='tight')

                    plt.close() 
                    
                    for diag_idx,diag_name in enumerate(gfed_diag[1:]):
                        std_error_total = np.zeros(12)
              
                
                        monthly_burn_total = np.zeros( len(months))
                        monthly_burn_count_total = np.zeros( len(months))
                        color = next(color_iter)
                        for month in range(len(months)):
                            total_burn_area_mask_15 = d[diag_name][:]
                            monthly_burn_total[month] = np.mean(total_burn_area_mask_15[month::12])
                            monthly_burn_count_total[month] = np.count_nonzero(total_burn_area_mask_15[month::12])
                            std_error_total[month] = np.std(total_burn_area_mask_15[month::12]) / np.sqrt(monthly_burn_count_total[month])
                            
                        #ax.plot(months, monthly_burn_total, label=f"{diag_name} Burned Area {unit}", color=color)
                        #ax.errorbar(months, monthly_burn_total, yerr=std_error_total, fmt='none', capsize=9, color=color,
                        #    elinewidth=1)
                    
                    
                else:
                    std_error = np.zeros(12)
                    plot_std = np.zeros(12)
                    std_error_list = []
                    monthly_burn = np.zeros((len(unique_land_cover_types), len(months)))
                    monthly_burn_count = np.zeros((len(unique_land_cover_types), len(months)))
                    nat_burn_area = np.zeros(12)
                    
                    
                    
                    for ilct_idx, ilct in enumerate(unique_land_cover_types):
                        total_burn_area_ilct = opened_datasets[idx][gfed_diag[0]][:, i,ilct_idx]
                        print('total_burn_area_ilct.shape')
                        print(total_burn_area_ilct.shape)
                        exit()
                        if ilct_idx > 0:
                           nat_burn_area += total_burn_area_ilct
                        for month in range(len(months)):
               
                            monthly_burn[ilct_idx, month] = np.mean(total_burn_area_ilct[month::12])
                            monthly_burn_count[ilct_idx,month] = np.count_nonzero(total_burn_area_ilct[month::12])

                            std_error[month] = np.std(total_burn_area_ilct[month::12]) / np.sqrt(np.count_nonzero(monthly_burn_count[ilct_idx]))
                            if ilct_idx == 9:
                                plot_std[month] = std_error[month]
                                print('ilct=9, std_error')
                                print(std_error)
                            
                        # Plot the burn area line for each land cover type
                        ax_t.plot(months, monthly_burn[ilct_idx], label=f"iLCT {ilct}: {land_cover_labels[ilct]}", color=color_map(ilct_idx))

                        ax_t.errorbar(months, monthly_burn[ilct_idx], yerr=std_error, fmt='none', capsize=9, color=color_map(ilct_idx),
                                    elinewidth=1)
                  
                    total_burned_across_ilct = np.sum(monthly_burn[1:,:], axis=0) #ignore "water" start from 1
                    

                   
                    # Plot the total burned area across all ilct types
                    ax_t.plot(months, total_burned_across_ilct, label="Total iLCT", color='black')
                 
                    #ax_t.errorbar(months, total_burned_across_ilct, yerr=total_burned_std_error, fmt='o', capsize=5, color='black', label="Error Bars")
                    ax.plot(months, total_burned_across_ilct, label="GFED5", color='black')
                    print('GFED5 error')
                    print(plot_std)
                    ax.errorbar(months, total_burned_across_ilct, yerr=plot_std, fmt='none', capsize=9, color='black',elinewidth=1)

                    
                    
                    
                    #ax_t.plot(months, monthly_burn_sum, label=f"Total", color='black')
                    #ax.plot(months, monthly_burn_sum, label=f"NAT", color='black')

                  
                    
                    ax_t.set_xlabel('Time')
                    ax_t.set_ylabel(f'Total Burned Area [{unit}]')
                    ax_t.set_title(f"Total Burned Area for Mask Value {gfed_cover_labels[i]} and Different iLCT Types")
                    ax_t.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=5)
                    ax_t.tick_params(axis='x', rotation=45)
       
                    plt.subplots_adjust(hspace=1.5)
                    file_name = f"NATBA_SEASON_{gfed_cover_labels[i]}_1997_2020.png"
                    file_path = os.path.join(output_path, file_name)
                    plt.grid(True)
                    plt.savefig(file_path, dpi=300, bbox_inches='tight')

                    plt.close() 
        
                        
                    for diag_idx,diag_name in enumerate(gfed_diag[1:]):
                        std_error_total = np.zeros(12)
              
                
                        monthly_burn_total = np.zeros( len(months))
                        monthly_burn_count_total = np.zeros( len(months))
                        color = next(color_iter)
                        for month in range(len(months)):
                            total_burn_area_mask = opened_datasets[idx][gfed_diag[diag_idx+1]][:, i]
                            monthly_burn_total[month] = np.mean(total_burn_area_mask[month::12])
                            monthly_burn_count_total[month] = np.count_nonzero(total_burn_area_mask[month::12])
                            std_error_total[month] = np.std(total_burn_area_mask[month::12]) / np.sqrt(monthly_burn_count_total[month])
                            
                        #ax.plot(months, monthly_burn_total, label=f"{diag_name} Burned Area {unit}", color=color)
                        #ax.errorbar(months, monthly_burn_total, yerr=std_error_total, fmt='none', capsize=9, color=color,
                        #    elinewidth=1)
            elif idx =='wglc':
                print("wglc")
                time_dt =  opened_datasets[idx]['time'].values
                time = pd.to_datetime(time_dt)
                
                wg = opened_datasets['wglc'].data_vars
                wd_diag = list(wg.keys())
                
                for diag_idx,diag_name in enumerate(wd_diag):
                        std_error_total = np.zeros(12)
              
                
                        monthly_burn_total = np.zeros( len(months))
                        monthly_burn_count_total = np.zeros( len(months))
                        color = next(color_iter)
                        for month in range(len(months)):
                            total_burn_area_mask = opened_datasets[idx][wd_diag[diag_idx]][:132, i]
                            monthly_burn_total[month] = np.mean(total_burn_area_mask[month::12])
                            monthly_burn_count_total[month] = np.count_nonzero(total_burn_area_mask[month::12])
                            std_error_total[month] = np.std(total_burn_area_mask[month::12]) / np.sqrt(monthly_burn_count_total[month])
                            
                        ax.plot(months, monthly_burn_total, label=f"{idx}{diag_name}", color=color)
                        ax.errorbar(months, monthly_burn_total, yerr=std_error_total, fmt='none', capsize=9, color=color,
                            elinewidth=1)
                

                
                        
                    
            else:
                
                std_error_total = np.zeros(12)
                monthly_burn_total_model = np.zeros( len(months))
                monthly_burn_count_total_model= np.zeros( len(months))
                
                for idn,diag in enumerate(diagnostics_list):
                    time_dt =  opened_datasets[idx]['time'].values
                    time = pd.to_datetime(time_dt)
                    
                    color = next(color_iter)
                    for month in range(len(months)):
                        if val == 'Lightning':
                             total_burn_area_mask_nudge = opened_datasets[idx][diag][156:, i]
                        else:
                            total_burn_area_mask_nudge = opened_datasets[idx][diag][:, i]
                        monthly_burn_total_model[month] = np.mean(total_burn_area_mask_nudge[month::12])
                        monthly_burn_count_total_model[month] = np.count_nonzero(total_burn_area_mask_nudge[month::12])
                        
                        std_error_total[month] = np.std(total_burn_area_mask_nudge[month::12]) / np.sqrt(monthly_burn_count_total_model[month])
                       
                    ax.plot(months, monthly_burn_total_model, label=f"Model", color=color)

                    print('model error')
                    print(std_error_total)
                    ax.errorbar(months, monthly_burn_total_model, yerr=std_error_total, fmt='none', capsize=9, color=color,
                        elinewidth=1)
               
        ax.set_xlabel('Month',fontsize=18)
        ax.set_ylabel(f'Total {val} {unit}',fontsize=18)
        ax.set_title(f"Natural {val} for {gfed_cover_labels[i]} ")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=5)
        plt.xticks(rotation=45, fontsize=18)
        plt.yticks(fontsize=18)
        plt.yscale('linear')
        #ax.set_xlim(0.5, 12.5) 




        ax.tick_params(axis='x', rotation=45)
        plt.subplots_adjust(hspace=1.5)





        #NATBA_TS_<region name>_<startyear>_<endyear>
        file_name_t = f"{val}_{idx}_SEASON_{gfed_cover_labels[i]}_1997_2020.png"
        file_path_t = os.path.join(output_path, file_name_t)
        plt.grid(True)
        plt.savefig(file_path_t, dpi=300, bbox_inches='tight')

        plt.close()

                
######################################################
#                    BASE                            #
######################################################

netcdf_paths = {'model':"/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/converted_model_not_nudged.nc",
                       'nudged':"/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/converted_model_nudged.nc",
                   'gfed':"/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/GFED_Mha.nc",
             'lightning_model':"/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC/gfed_model_lightning.nc",
               'lightning_nudged':'/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC/gfed_model_nudged_lightning.nc',
                 'wglc': '/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC/WLGC_GFED_region_data.nc',
                 'model_anthro':"/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC/lightning_model_result_model.nc",
                'nudged_anthro':"/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC/lightning_model_result_nudged.nc",
                'precip_model':"/discover/nobackup/projects/giss_ana/users/kmezuman/Precip/model_precipitation.nc",
                'precip_nudged':"/discover/nobackup/projects/giss_ana/users/kmezuman/Precip/nudged_precipitation.nc"
                 
                
                }
    


#val = input("Enter BA or Lightning or Precip: ")
val='BA'
#set units per product/diagnostic
unit = ''
#set output path 
output_path = '/discover/nobackup/kmezuman/plots/fire_repository/Hackathon/'
if val == 'BA':
    opened_datasets = {}
    #obs = input("Observation product: yes or no ")
    obs='yes'


    if obs.lower() =='yes':
        unit= 'Mha'
        
        print("loading gfed")
        file_path = netcdf_paths['gfed']
        gfed = xr.open_dataset(file_path)
        opened_datasets['gfed']=gfed
        print("loaded")
        
    list_of_sim = ['model', 'nudged']

    # Number of simulations to choose
    #num_inputs = int(input("How many simulations? "))
    num_inputs = 1

    #print("List of simulations to choose from:", list_of_sim)

    # Initialize a dictionary to store the opened datasets


    # Loop to get simulation inputs
    for i in range(num_inputs):
        #user_input = input(f"Enter simulation {i+1}: ")
        user_input = 'model'
        if user_input in list_of_sim:
            file_path = netcdf_paths[user_input]
            model = xr.open_dataset(file_path)
            model['BA_Mha'] = model['BA_tree_Mha']+model['BA_grass_Mha']+model['BA_shrub_Mha']
            opened_datasets[user_input] = model


        else:
            print(f"Invalid simulation: {user_input}")
    #diag_input = input('Specify diagnostics from {BA_Mha, BA_shrub_Mha, BA_grass_Mha, BA_tree_Mha}')
    diag_input = 'BA_Mha'
    diagnostics_list = diag_input.split(',')
        
        
    print("Plotting...")
    output_path = output_path + 'BA/'
    #Data_TS_Season_Regional(opened_datasets, diagnostics_list,val,unit,output_path)    
    Data_TS_Season_model_gfed(opened_datasets, diagnostics_list,val,unit,output_path)
elif val == 'Lightning':
    unit = '[strokes/km²/day]'
    opened_datasets = {}
    obs = input("Observation product: yes or no ")
    if obs.lower() == 'yes':
        
        print("loading wglc")
        file_path = netcdf_paths['wglc']
        wglc = xr.open_dataset(file_path)
        opened_datasets['wglc']=wglc
        print("loaded")
        
        # List of available simulations
    list_of_sim = ['lightning_model', 'lightning_nudged','model_anthro','nudged_anthro']

    # Number of simulations to choose
    num_inputs = int(input("How many simulations? "))

    print("List of simulations to choose from:", list_of_sim)



    # Loop to get simulation inputs
    for i in range(num_inputs):
        user_input = input(f"Enter simulation {i+1}: ")
        if user_input in list_of_sim:
            file_path = netcdf_paths[user_input]
            model = xr.open_dataset(file_path)
            opened_datasets[user_input] = model
            print(f"Opened {user_input} dataset")
        else:
            print(f"Invalid simulation: {user_input}")

    diag_input = input('Specify diagnostics from {f_ignCG for modelVs nudged and f_ignHUMAN for anthro}')
    diagnostics_list = diag_input.split(',')

# Trim and convert to lowercase for consistency
    diagnostics_list = [diag.strip() for diag in diagnostics_list]
    output_path = output_path + 'Lightning/'
    Data_TS_Season_Regional(opened_datasets, diagnostics_list,val,unit,output_path)
    Data_TS_Season_model_gfed(opened_datasets, diagnostics_list,val,unit,output_path)
    
    


elif val == 'Precip':
    unit = '[mm/day]'
    opened_datasets = {}
    list_of_sim = ['precip_model', 'precip_nudged']
    # Number of simulations to choose
    num_inputs = int(input("How many simulations? "))

    print("List of simulations to choose from:", list_of_sim)



    # Loop to get simulation inputs
    for i in range(num_inputs):
        user_input = input(f"Enter simulation {i+1}: ")
        if user_input in list_of_sim:
            file_path = netcdf_paths[user_input]
            model = xr.open_dataset(file_path)
            opened_datasets[user_input] = model
            print(f"Opened {user_input} dataset")
        else:
            print(f"Invalid simulation: {user_input}")

    diag_input = input('Specify diagnostics from {FLAMM_prec}')
    diagnostics_list = diag_input.split(',')

# Trim and convert to lowercase for consistency
    diagnostics_list = [diag.strip() for diag in diagnostics_list]
    output_path = output_path + 'Precip/'
    Data_TS_Season_Regional(opened_datasets, diagnostics_list,val,unit,output_path)
    Data_TS_Season_model_gfed(opened_datasets, diagnostics_list,val,unit,output_path)
else:
    print("Invalid input for 'BA' or 'Lightning' or 'Precip'")
    

                
                
