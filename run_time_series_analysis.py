def run_time_series_analysis(folder_data_list, time_analysis_figure_data, annual=False, save_netcdf=False):
    """
    Run time series analysis for multiple datasets, creating both long time series and seasonality plots
    
    Parameters:
    -----------
    folder_data_list : list
        List of folder data dictionaries
    time_analysis_figure_data : dict
        Figure metadata and settings
    annual : bool, optional
        If True, show annual totals for data
        If False, show monthly resolution data points 
        Default is False
    save_netcdf : bool, optional
        Whether to save NetCDF files
    """
    
    print(f"\n=== CREATING BOTH TIME SERIES AND SEASONALITY PLOTS ===")
    
    # Create TWO separate figures: one for time series, one for seasonality
    fig_timeseries, axis_timeseries = plt.subplots(figsize=(16, 6))
    fig_seasonality, axis_seasonality = plt.subplots(figsize=(15, 6))
    
    # Configure seasonality plot
    axis_seasonality.set_xlabel("Month")
    axis_seasonality.set_ylabel(time_analysis_figure_data["ylabel"]) 
    axis_seasonality.set_title(f"Seasonal Analysis - {time_analysis_figure_data['title']}")
    axis_seasonality.set_xticks(range(1, 13))
    axis_seasonality.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    global_year_max = 0
    global_year_min = 9999

    # Make sure the output directory exists before saving files
    output_dir = time_analysis_figure_data['figs_folder']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensuring output directory exists: {output_dir}")

    # Get logmapscale setting from configuration (default to True if not specified)
    logmapscale = time_analysis_figure_data.get('logmapscale', True)
    print(f"Using logmapscale setting: {logmapscale}")

    # Initialize dictionary to store all datasets for combined NetCDF
    all_datasets = {}

    # Process each dataset
    for index, folder_data in enumerate(folder_data_list):
        # Create individual map figure for this dataset
        map_figure, map_axis = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(18, 10),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

        folder_path, figure_data, file_type, variables = (
            folder_data["folder_path"],
            folder_data["figure_data"],
            folder_data["file_type"],
            folder_data["variables"],
        )

        print(f"\nProcessing dataset {index+1}/{len(folder_data_list)}: {file_type}")
        print(f"Variables: {variables}")
        print(f"Folder path: {folder_path}")
        print(f"Annual aggregation: {annual}")

        # Get data for this dataset
        (
            time_mean_data,
            data_per_year_stack,
            longitude,
            latitude,
            units,
            start_year,
            end_year,
        ) = obtain_time_series_xarray(
            NetCDF_folder_Path=folder_path,
            NetCDF_Type=file_type,
            variables=variables,
            annual=annual,
            save_netcdf=save_netcdf
        )

        # Skip if data retrieval failed
        if data_per_year_stack is None:
            print(f"Skipping dataset {index+1} due to data retrieval failure")
            plt.close(map_figure)
            continue

        figure_label = f"{figure_data['label']} ({start_year}-{end_year})"

        print(f"=== DEBUGGING MAP DATA for {file_type} ===")
        print(f"time_mean_data shape: {time_mean_data.shape if hasattr(time_mean_data, 'shape') else 'No shape attr'}")
        print(f"longitude shape: {longitude.shape if hasattr(longitude, 'shape') else 'No shape attr'}")
        print(f"latitude shape: {latitude.shape if hasattr(latitude, 'shape') else 'No shape attr'}")
        print(f"units: {units}")
        print(f"figure_label: {figure_label}")
        print(f"=== END DEBUGGING MAP DATA ===\n")

        # Create and save individual map
        if time_mean_data is not None:
            map_plot(
                figure=map_figure,
                axis=map_axis,
                axis_length=1,
                axis_index=0,
                decade_data=time_mean_data,
                longitude=longitude,
                latitude=latitude,
                subplot_title=figure_label,
                units=units,
                cbarmax=figure_data["cbarmax"],
                logMap=logmapscale,
            )
        else:
            print(f"Warning: time_mean_data is None for dataset {index+1}")

        # Save and close individual map
        map_figure.savefig(f"{time_analysis_figure_data['figs_folder']}/map_figure_{index}")
        plt.close(map_figure)
        print(f"Saved and closed: map_figure_{index}")

        # Plot BOTH time series and seasonality for this dataset
        if data_per_year_stack is not None and len(data_per_year_stack) > 0:
            
            # 1. Plot long time series
            time_series_plot(
                axis=axis_timeseries,
                data=data_per_year_stack,
                marker=figure_data["marker"],
                line_style=figure_data["line_style"],
                color=figure_data["color"],
                label=figure_label,
            )
            
            # 2. Calculate and plot seasonality
            seasonal_data = calculate_seasonal_statistics(data_per_year_stack)
            seasonal_time_series_plot(
                axis=axis_seasonality,
                seasonal_data=seasonal_data,
                marker=figure_data["marker"],
                line_style=figure_data["line_style"],
                color=figure_data["color"],
                label=figure_label,
            )
            
            print(f"Added dataset {index+1} to both time series and seasonality plots")

        # Update global year range for time series plot
        if data_per_year_stack is not None and len(data_per_year_stack) > 0:
            if np.issubdtype(type(data_per_year_stack[0, 0]), np.floating):
                year_max = int(np.ceil(data_per_year_stack[:, 0].max()))
                year_min = int(np.floor(data_per_year_stack[:, 0].min()))
            else:
                year_max = int(end_year) if end_year is not None else 2023
                year_min = int(start_year) if start_year is not None else 2010
                
            global_year_max = max(global_year_max, year_max)
            global_year_min = min(global_year_min, year_min)

        # Store dataset info for combined NetCDF
        if data_per_year_stack is not None and len(data_per_year_stack) > 0:
            all_datasets[f"{file_type}_{variables}"] = {
                'data': np.sum(data_per_year_stack[:, 1]) if len(data_per_year_stack.shape) > 1 else np.sum(data_per_year_stack),
                'time': list(range(int(start_year) if start_year else 2010, int(end_year) + 1 if end_year else 2024)),
                'units': units,
                'start_year': start_year,
                'end_year': end_year,
                'variables': variables,
                'file_type': file_type,
                'folder_path': folder_path
            }

    # Save combined NetCDF file with all datasets
    if save_netcdf and all_datasets:
        try:
            save_combined_netcdf(all_datasets, output_dir, annual, global_year_min, global_year_max)
        except Exception as e:
            print(f"Warning: Could not save NetCDF file: {e}")

    print(f"\n=== FINALIZING PLOTS ===")

    # ========== FINALIZE TIME SERIES PLOT ==========
    print("Finalizing time series plot...")
    
    # Set labels for time series plot
    if annual:
        xlabel = f"Yearly Data ({global_year_min}-{global_year_max})"
    else:
        xlabel = f"Monthly Data ({global_year_min}-{global_year_max})"
    
    axis_timeseries.set_title(time_analysis_figure_data["title"])
    axis_timeseries.set_xlabel(xlabel)
    axis_timeseries.set_ylabel(time_analysis_figure_data["ylabel"])
    
    # Format x-axis for time series
    if annual:
        years = range(global_year_min, global_year_max + 1)
        if len(years) <= 10:
            axis_timeseries.set_xticks(years)
            axis_timeseries.set_xticklabels([str(year) for year in years])
        else:
            axis_timeseries.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    else:
        axis_timeseries.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))

    # Add legend for time series
    handles_ts, labels_ts = axis_timeseries.get_legend_handles_labels()
    if handles_ts:
        legend_ts = axis_timeseries.legend(
            handles_ts, labels_ts,
            loc="center left",
            fontsize='medium',
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fancybox=True,
            shadow=True
        )

    # Layout and save time series
    plt.figure(fig_timeseries.number)
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    plt.savefig(f"{time_analysis_figure_data['figs_folder']}/time_analysis_figure", 
                bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Saved: time_analysis_figure")

    # ========== FINALIZE SEASONALITY PLOT ==========
    print("Finalizing seasonality plot...")
    
    # Add legend for seasonality
    handles_seas, labels_seas = axis_seasonality.get_legend_handles_labels()
    if handles_seas:
        legend_seas = axis_seasonality.legend(
            handles_seas, labels_seas,
            loc="center left",
            fontsize='medium',
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fancybox=True,
            shadow=True
        )

    # Layout and save seasonality
    plt.figure(fig_seasonality.number)
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    plt.savefig(f"{time_analysis_figure_data['figs_folder']}/time_analysis_figure_seasonality", 
                bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Saved: time_analysis_figure_seasonality")

    # Show both plots
    plt.figure(fig_timeseries.number)
    plt.show()
    
    plt.figure(fig_seasonality.number)
    plt.show()

    # Create difference maps if more than one dataset is provided (OPTIONAL)
    create_difference_maps = False  # Change to True if you want difference maps

    if len(folder_data_list) > 1 and create_difference_maps:
        print("\n=== Creating difference maps between datasets ===")
        
        # Get the first two datasets for comparison
        first_selection = 0
        second_selection = 1
        
        folder_data_one = folder_data_list[first_selection]
        folder_data_two = folder_data_list[second_selection]
        
        try:
            # Get data for both datasets
            (
                time_mean_data_one,
                data_per_year_stack_one,
                longitude_one,
                latitude_one,
                units_one,
                start_year_one,
                end_year_one,
            ) = obtain_time_series_xarray(
                NetCDF_folder_Path=folder_data_one["folder_path"],
                NetCDF_Type=folder_data_one["file_type"],
                variables=folder_data_one["variables"],
                annual=annual,
                save_netcdf=False
            )
            
            (
                time_mean_data_two,
                data_per_year_stack_two,
                longitude_two,
                latitude_two,
                units_two,
                start_year_two,
                end_year_two,
            ) = obtain_time_series_xarray(
                NetCDF_folder_Path=folder_data_two["folder_path"],
                NetCDF_Type=folder_data_two["file_type"],
                variables=folder_data_two["variables"],
                annual=annual,
                save_netcdf=False
            )
            
            # Check if datasets are compatible for difference calculation
            if (time_mean_data_one is not None and time_mean_data_two is not None and
                time_mean_data_one.shape == time_mean_data_two.shape and
                np.array_equal(longitude_one, longitude_two) and
                np.array_equal(latitude_one, latitude_two)):
                
                # Calculate difference
                time_mean_data_diff = time_mean_data_one - time_mean_data_two
                longitude_diff = longitude_one
                latitude_diff = latitude_one
                
                # Create difference map
                map_figure, map_axis = plt.subplots(
                    nrows=1,
                    ncols=1,
                    figsize=(18, 10),
                    subplot_kw={"projection": ccrs.PlateCarree()},
                )
                
                figure_label_diff = f"Difference: {folder_data_one['figure_data']['label']} - {folder_data_two['figure_data']['label']}"
                units_diff = units_one
                
                map_plot(
                    figure=map_figure,
                    axis=map_axis,
                    axis_length=1,
                    axis_index=0,
                    decade_data=time_mean_data_diff,
                    longitude=longitude_diff,
                    latitude=latitude_diff,
                    subplot_title=figure_label_diff,
                    units=units_diff,
                    cbarmax=None,
                    is_diff=True,
                )
                
                map_figure.savefig(
                   f"{time_analysis_figure_data['figs_folder']}/figure{first_selection}_and_figure{second_selection}_diff_map"
                )
                
                plt.close(map_figure)
                print(f"Difference map saved (not displayed): figure{first_selection}_and_figure{second_selection}_diff_map")
            else:
                print("Warning: Datasets have incompatible dimensions for difference calculation")
                
        except Exception as e:
            print(f"Error creating difference maps: {e}")

    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Generated files:")
    print(f"  - Individual maps: map_figure_0, map_figure_1, ...")
    print(f"  - Time series plot: time_analysis_figure")
    print(f"  - Seasonality plot: time_analysis_figure_seasonality")
    if create_difference_maps:
        print(f"  - Difference maps: figure0_and_figure1_diff_map")
