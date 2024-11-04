from os import listdir
import os
from os.path import join, isfile
import re
import warnings
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import json
import matplotlib.dates as mdates
import xarray as xr
from glob import glob
import h5py
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from utilityGlobal import (
    SCRIPTS_ENV_VARIABLES,
    MONTHLIST,
    DISTINCT_COLORS,
    MASK_LIST,
    MONTHS_NUM,
    GFED_COVER_LABELS,
    LAND_COVER_LABELS,
    NUM_MONTHS,
    MARKER,
    SECONDS_IN_A_YEAR,
    KILOGRAMS_TO_GRAMS,
    COLOR_MAP,
    SQM_TO_SQHA,
    KM_NEG_2TOM_NEG_2,
    KM_SQUARED_TO_METERS_SQUARED,
    DAYS_TO_SECONDS,
    EARTH_RADIUS,
)


######################################################
#                    PANELS                          #
######################################################
def plotPanel(output_directory):
    plt.tight_layout()
    plt.savefig(
        join(output_directory, f"Model_Combined_Season.png"), dpi=600
    )  # specify fle name
    plt.close()


def panelRunner(rows, columns, total_plots, plot_figure_size, output_directory):
    img_paths = []
    fig, axs = plt.subplots(rows, columns, figsize=plot_figure_size)
    for row_index in range(rows):
        for column_index in range(columns):
            index = row_index * columns + column_index

            if index >= total_plots:
                break  # Stop if all plots are processed

            img_path = img_paths[index]
            img = mpimg.imread(img_path)
            axs[row_index, column_index].imshow(img)
            axs[row_index, column_index].axis("off")
            # Add any additional customizations to the subplots if needed

    # Hide any empty subplots
    for i in range(total_plots, rows * columns):
        axs.flatten()[i].axis("off")

    plotPanel(output_directory)


######################################################
#                    CARBON BUDGET                   #
######################################################
def getCarbonBudgetVariables(input_files, output_file, df_variable_names) -> None:
    try:
        dataset_one = nc.Dataset(input_files[0], "r")
        dataset_two = nc.Dataset(input_files[1], "r")
        dataset_three = nc.Dataset(input_files[2], "r")

        output_file = output_file

        CO2n_pyrE_src_hemis = dataset_three.variables["CO2n_pyrE_src_hemis"]
        CO2n_pyrE_src = dataset_three.variables["CO2n_pyrE_src"]
        CO2n_pyrE_src_units = CO2n_pyrE_src.units

        CO2n_Total_Mass_hemis = dataset_one.variables["CO2n_Total_Mass_hemis"]
        CO2n_Total_Mass = dataset_one.variables["CO2n_Total_Mass"]
        CO2n_Total_Mass_units = CO2n_Total_Mass.units

        C_lab_hemis = dataset_two.variables["C_lab_hemis"]
        C_lab = dataset_two.variables["C_lab"]
        C_lab_units = C_lab.units

        soilCpool_hemis = dataset_two.variables["soilCpool_hemis"]
        soilCpool = dataset_two.variables["soilCpool"]
        soilCpool_units = soilCpool.units

        gpp_hemis = dataset_two.variables["gpp_hemis"]
        gpp = dataset_two.variables["gpp"]
        gpp_units = gpp.units

        rauto_hemis = dataset_two.variables["rauto_hemis"]
        rauto = dataset_two.variables["rauto"]
        rauto_units = rauto.units

        soilresp_hemis = dataset_two.variables["soilresp_hemis"]
        soilresp = dataset_two.variables["soilresp"]
        soilresp_units = soilresp.units

        ecvf_hemis = dataset_two.variables["ecvf_hemis"]
        ecvf = dataset_two.variables["ecvf"]
        ecvf_units = ecvf.units

        destination_variable = df_variable_names
        destination_units = [
            (CO2n_pyrE_src_units),
            (CO2n_Total_Mass_units),
            (C_lab_units),
            (soilCpool_units),
            (gpp_units),
            (rauto_units),
            (soilresp_units),
            (ecvf_units),
            (ecvf_units),
            "-",
        ]
        destination_CO2 = [
            (CO2n_pyrE_src_hemis[2,]),
            (CO2n_Total_Mass_hemis[2,]),
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
        ]
        total_carbon = [
            "-",
            "-",
            (C_lab_hemis[2,]),
            (soilCpool_hemis[2,]),
            (gpp_hemis[2,]),
            (rauto_hemis[2,]),
            (soilresp_hemis[2,]),
            (ecvf_hemis[2,]),
            (gpp_hemis[2,] - rauto_hemis[2,] - soilresp_hemis[2,] - ecvf_hemis[2,]),
            "-",
        ]
        return (destination_variable, destination_units, destination_CO2, total_carbon)
    except Exception as error:
        print("[-] User must input three files for this script", error)
        return ()


def saveDataframe(formatted_table, output_file):
    with open(output_file, "w") as file:
        file.write(formatted_table)


def createDataframe(
    destination_variable_names, destination_units, destination_CO2, total_carbon
):
    df = pd.DataFrame(
        {
            "Varaiable": destination_variable_names,
            # 'CO': [(CO_biomass_src_hemis[2,]),(CO_Total_Mass_hemis[2,]), '-','-', '-'],
            "Units": destination_units,
            "CO2": destination_CO2,
            "Total Carbon": total_carbon,
        }
    )
    saveDataframe(df.to_string(index=False))


def carbonBudgetRunner(input_files, output_file, df_variable_names):
    (
        destination_variable_names,
        destination_units,
        destination_CO2,
        total_carbon,
    ) = getCarbonBudgetVariables(
        input_files=input_files,
        output_file=output_file,
        pd_variables=df_variable_names,
    )
    createDataframe(
        destination_variable_names,
        destination_units,
        destination_CO2,
        total_carbon,
    )


######################################################
#                    LINE PLOT                       #
######################################################
# xlabel_name = Time or Month
def linePlotMask(
    axis,
    output_path,
    xlabel_name,
    ylabel_name,
    title_name,
    output_file_name,
    rotation_type,
):
    axis.set_xlabel(xlabel_name, fontsize=18)
    axis.set_ylabel(ylabel_name, fontsize=18)
    axis.set_title(title_name)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        fancybox=True,
        shadow=True,
        ncol=5,
    )

    if rotation_type == "Seasonal":
        plt.xticks(rotation=45, fontsize=18)
        plt.yticks(fontsize=18)
        plt.yscale("linear")
    elif rotation_type == "Regional":
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # ax.set_xlim(0.5, 12.5)

    axis.tick_params(axis="x", rotation=45)
    plt.subplots_adjust(hspace=1.5)

    # NATBA_TS_<region name>_<startyear>_<endyear>
    file_path = os.path.join(output_path, output_file_name)
    plt.grid(True)
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Complete ", output_file_name)


def wglcLinePlots(dataset, axis, mask_index, dataset_key_index, monthly=True):
    print("wglc")
    color_iter = iter(DISTINCT_COLORS)
    time_dt = dataset[dataset_key_index]["time"].values
    time = pd.to_datetime(time_dt)

    wg = dataset["wglc"].data_vars
    wd_diag = list(wg.keys())
    if monthly == True:
        for diag_idx, diag_name in enumerate(wd_diag):
            std_error_total = np.zeros(12)

            monthly_burn_total = np.zeros(len(MONTHS_NUM))
            monthly_burn_count_total = np.zeros(len(MONTHS_NUM))
            color = next(color_iter)
            for month in range(len(MONTHS_NUM)):
                total_burn_area_mask = dataset[dataset_key_index][wd_diag[diag_idx]][
                    :132, mask_index
                ]
                monthly_burn_total[month] = np.mean(total_burn_area_mask[month::12])
                monthly_burn_count_total[month] = np.count_nonzero(
                    total_burn_area_mask[month::12]
                )
                std_error_total[month] = np.std(
                    total_burn_area_mask[month::12]
                ) / np.sqrt(monthly_burn_count_total[month])

                axis.plot(
                    MONTHS_NUM,
                    monthly_burn_total,
                    label=f"{dataset_key_index}{diag_name}",
                    color=color,
                )
                axis.errorbar(
                    MONTHS_NUM,
                    monthly_burn_total,
                    yerr=std_error_total,
                    fmt="none",
                    capsize=9,
                    color=color,
                    elinewidth=1,
                )

    elif monthly == False:
        for diag_idx, diag_name in enumerate(wd_diag):
            color = next(color_iter)
            axis.plot(
                time,
                dataset[dataset_key_index][diag_name][:, mask_index],
                label="Lightning Density WGLC",
                color=color,
                linewidth=1.5,
            )


def line_plot_gfed_diagonal_calculation(
    gfed_diag, dataset, dataset_key_index, mask_index
):
    color_iter = iter(DISTINCT_COLORS)
    for diag_idx, diag_name in enumerate(gfed_diag[1:]):
        std_error_total = np.zeros(12)

        monthly_burn_total = np.zeros(len(MONTHS_NUM))
        monthly_burn_count_total = np.zeros(len(MONTHS_NUM))
        color = next(color_iter)
        for month in range(len(MONTHS_NUM)):
            # ternary operation distinguishing this function between the seasonal and monthly
            total_burn_area_mask = (
                dataset[diag_name][:]
                if mask_index == 15
                else dataset[dataset_key_index][gfed_diag[diag_idx + 1]][:, mask_index]
            )
            monthly_burn_total[month] = np.mean(total_burn_area_mask[month::12])
            monthly_burn_count_total[month] = np.count_nonzero(
                total_burn_area_mask[month::12]
            )
            std_error_total[month] = np.std(total_burn_area_mask[month::12]) / np.sqrt(
                monthly_burn_count_total[month]
            )

        # ax.plot(MONTHS_NUM, monthly_burn_total, label=f"{diag_name} Burned Area {unit}", color=color)
        # ax.errorbar(MONTHS_NUM, monthly_burn_total, yerr=std_error_total, fmt='none', capsize=9, color=color,
        #    elinewidth=1)


def Data_TS_Season_model_gfed_handleGFEDLandCoverTypes(
    ax_t, unique_land_cover_types, data_set, gfed_diag
):
    std_error = np.zeros(12)
    plot_std = np.zeros(12)
    monthly_burn = np.zeros((len(unique_land_cover_types), len(MONTHS_NUM)))
    monthly_burn_count = np.zeros((len(unique_land_cover_types), len(MONTHS_NUM)))

    for ilct_idx, ilct in enumerate(unique_land_cover_types):
        total_burn_area_ilct = data_set[gfed_diag[0]][:, ilct_idx]
        for month in range(len(MONTHS_NUM)):

            monthly_burn[ilct_idx, month] = np.mean(total_burn_area_ilct[month::12])
            monthly_burn_count[ilct_idx, month] = np.count_nonzero(
                total_burn_area_ilct[month::12]
            )

            std_error[month] = np.std(total_burn_area_ilct[month::12]) / np.sqrt(
                np.count_nonzero(monthly_burn_count[ilct_idx])
            )
            if ilct_idx == 9:
                plot_std[month] = std_error[month]
        # Plot the burn area line for each land cover type
        ax_t.plot(
            MONTHS_NUM,
            monthly_burn[ilct_idx],
            label=f"iLCT {ilct}: {LAND_COVER_LABELS[ilct]}",
            color=COLOR_MAP(ilct_idx),
        )

        ax_t.errorbar(
            MONTHS_NUM,
            monthly_burn[ilct_idx],
            yerr=std_error,
            fmt="none",
            capsize=9,
            color=COLOR_MAP(ilct_idx),
            elinewidth=1,
        )

    total_burned_across_ilct = np.sum(monthly_burn[1:, :], axis=0)
    return (total_burned_across_ilct, monthly_burn, plot_std)


def Data_TS_Season_model_gfed_handleGFEDPlot(
    axis,
    ax_t,
    unit,
    mask_index,
    output_path,
    plot_std,
    monthly_burn,
    total_burned_across_ilct,
):
    # Plot the total burned area across all ilct types
    ax_t.plot(
        MONTHS_NUM,
        total_burned_across_ilct,
        label="Total iLCT",
        color="black",
    )

    # ax_t.errorbar(MONTHS_NUM, total_burned_across_ilct, yerr=total_burned_std_error, fmt='o', capsize=5, color='black', label="Error Bars")
    # ax.plot(MONTHS_NUM, total_burned_across_ilct, label="NAT", color='black')

    # Plot the total burned area across all ilct types
    ax_t.plot(
        MONTHS_NUM,
        total_burned_across_ilct,
        label="Total iLCT",
        color="black",
    )

    # ax_t.errorbar(MONTHS_NUM, total_burned_across_ilct, yerr=total_burned_std_error, fmt='o', capsize=5, color='black', label="Error Bars")
    axis.plot(
        MONTHS_NUM,
        total_burned_across_ilct,
        label="GFED5",
        color="black",
    )
    print(plot_std)
    axis.errorbar(
        MONTHS_NUM,
        total_burned_across_ilct,
        yerr=plot_std,
        fmt="none",
        capsize=9,
        color="black",
        elinewidth=1,
    )

    total_burned_across_ilct = np.sum(monthly_burn[1:, :], axis=0)
    print(total_burned_across_ilct.shape)

    # Plot the total burned area across all ilct types
    ax_t.plot(
        MONTHS_NUM,
        total_burned_across_ilct,
        label="Total iLCT",
        color="black",
    )

    linePlotMask(
        axis=ax_t,
        output_path=output_path,
        xlabel_name="Time",
        ylabel_name=f"Total Burned Area [{unit}]",
        title_name=f"Total Burned Area for Mask Value {GFED_COVER_LABELS[mask_index]} and Different iLCT Types",
        output_file_name=f"NATBA_SEASON_{GFED_COVER_LABELS[mask_index]}_1997_2020.png",
        rotation_type=None,
    )


def Data_TS_Season_Regional_handleGFEDPlot(
    axis, mask_index, output_path, ax_t, time_gf, total_burn_area_total_ilct
):
    ax_t.plot(
        time_gf,
        total_burn_area_total_ilct,
        label="Total ILCT",
        color="black",
        linewidth=1.5,
    )
    axis.plot(
        time_gf,
        total_burn_area_total_ilct,
        label="NAT",
        color="black",
        linewidth=1.5,
    )
    linePlotMask(
        axis=ax_t,
        output_path=output_path,
        xlabel_name="Time",
        ylabel_name="Total Burned Area [Mha]",
        title_name=f"Total Burned Area for Mask Value {GFED_COVER_LABELS[mask_index]} and Different iLCT Types",
        output_file_name=f"NATBA_TS_gfed{GFED_COVER_LABELS[mask_index]}_1997_2020.png",
        rotation_type="Regional",
    )


def Data_TS_Season_model_gfed_handleGFEDLandCoverTypesGen(
    dataset, dataset_key_index, gfed_diag, unique_land_cover_types, mask_index
):
    std_error = np.zeros(12)
    plot_std = np.zeros(12)
    monthly_burn = np.zeros((len(unique_land_cover_types), len(MONTHS_NUM)))
    monthly_burn_count = np.zeros((len(unique_land_cover_types), len(MONTHS_NUM)))
    nat_burn_area = np.zeros(12)

    for ilct_idx, ilct in enumerate(unique_land_cover_types):
        total_burn_area_ilct = dataset[mask_index][gfed_diag[0]][
            :, dataset_key_index, ilct_idx
        ]
        print("total_burn_area_ilct.shape")
        print(total_burn_area_ilct.shape)
        exit()
        if ilct_idx > 0:
            nat_burn_area += total_burn_area_ilct
        for month in range(len(MONTHS_NUM)):

            monthly_burn[ilct_idx, month] = np.mean(total_burn_area_ilct[month::12])
            monthly_burn_count[ilct_idx, month] = np.count_nonzero(
                total_burn_area_ilct[month::12]
            )

            std_error[month] = np.std(total_burn_area_ilct[month::12]) / np.sqrt(
                np.count_nonzero(monthly_burn_count[ilct_idx])
            )
            if ilct_idx == 9:
                plot_std[month] = std_error[month]
                print("ilct=9, std_error")
                print(std_error)

        # Plot the burn area line for each land cover type
        ax_t.plot(
            MONTHS_NUM,
            monthly_burn[ilct_idx],
            label=f"iLCT {ilct}: {LAND_COVER_LABELS[ilct]}",
            color=COLOR_MAP(ilct_idx),
        )

        ax_t.errorbar(
            MONTHS_NUM,
            monthly_burn[ilct_idx],
            yerr=std_error,
            fmt="none",
            capsize=9,
            color=COLOR_MAP(ilct_idx),
            elinewidth=1,
        )
    total_burned_across_ilct = np.sum(
        monthly_burn[1:, :], axis=0
    )  # ignore "water" start from 1
    return (total_burned_across_ilct, monthly_burn, plot_std)


def Data_TS_Season_model_gfed_handle_GFED(
    dataset, mask_index, dataset_key_index, netcdf_path, axis, unit, output_path
):
    fig_t, ax_t = plt.subplots(figsize=(12, 8), tight_layout=True)
    color_map = plt.get_cmap("tab20")
    gd = dataset["gfed"].data_vars
    gfed_diag = list(gd.keys())

    time_dt = dataset[dataset_key_index]["time"].values
    unique_land_cover_types = dataset["gfed"]["ilct"].values

    if mask_index == 15:
        data_set = xr.open_dataset(netcdf_path)
        time_dt = data_set["time"].values
        time_15 = pd.to_datetime(time_dt)
        gd = data_set.data_vars
        gfed_diag = list(gd.keys())

        total_burned_across_ilct, monthly_burn, plot_std = (
            Data_TS_Season_model_gfed_handleGFEDLandCoverTypes(
                ax_t, unique_land_cover_types, data_set, gfed_diag
            )
        )

        Data_TS_Season_model_gfed_handleGFEDPlot(
            axis,
            ax_t,
            unit,
            mask_index,
            output_path,
            plot_std,
            monthly_burn,
            total_burned_across_ilct,
        )

        line_plot_gfed_diagonal_calculation(
            gfed_diag=gfed_diag,
            dataset=data_set,
            dataset_key_index=dataset_key_index,
            mask_index=mask_index,
        )

    else:

        total_burned_across_ilct, monthly_burn, plot_std = (
            Data_TS_Season_model_gfed_handleGFEDLandCoverTypesGen(
                dataset,
                dataset_key_index,
                gfed_diag,
                unique_land_cover_types,
                mask_index,
            )
        )

        Data_TS_Season_model_gfed_handleGFEDPlot(
            axis,
            ax_t,
            unit,
            mask_index,
            output_path,
            plot_std,
            monthly_burn,
            total_burned_across_ilct,
        )

        line_plot_gfed_diagonal_calculation(
            dataset=dataset,
            dataset_key_index=dataset_key_index,
            mask_index=mask_index,
            gfed_diag=gfed_diag,
        )

    pass


def Data_TS_Season_model_gfed_handle_else(
    dataset, diagnostics_list, value, mask_index, axis, dataset_key_index
):
    color_iter = iter(DISTINCT_COLORS)
    std_error_total = np.zeros(12)
    monthly_burn_total_model = np.zeros(len(MONTHS_NUM))
    monthly_burn_count_total_model = np.zeros(len(MONTHS_NUM))

    for idn, diag in enumerate(diagnostics_list):
        time_dt = dataset[dataset_key_index]["time"].values
        time = pd.to_datetime(time_dt)

        color = next(color_iter)
        for month in range(len(MONTHS_NUM)):
            if value == "Lightning":
                total_burn_area_mask_nudge = dataset[dataset_key_index][diag][
                    156:, mask_index
                ]
            else:
                total_burn_area_mask_nudge = dataset[dataset_key_index][diag][
                    :, mask_index
                ]
            monthly_burn_total_model[month] = np.mean(
                total_burn_area_mask_nudge[month::12]
            )
            monthly_burn_count_total_model[month] = np.count_nonzero(
                total_burn_area_mask_nudge[month::12]
            )

            std_error_total[month] = np.std(
                total_burn_area_mask_nudge[month::12]
            ) / np.sqrt(monthly_burn_count_total_model[month])

        axis.plot(
            MONTHS_NUM,
            monthly_burn_total_model,
            label=f"Model",
            color=color,
        )

        print("model error")
        print(std_error_total)
        axis.errorbar(
            MONTHS_NUM,
            monthly_burn_total_model,
            yerr=std_error_total,
            fmt="none",
            capsize=9,
            color=color,
            elinewidth=1,
        )


# seasonality
def Data_TS_Season_model_gfed(
    opened_datasets, diagnostics_list, val, unit, output_path, figure_shape=(12, 8)
):

    for i in MASK_LIST:
        fig, ax = plt.subplots(figsize=figure_shape, tight_layout=True)

        # Not needed anymore due to making them available in each individual function
        # color_map = plt.get_cmap("tab20")
        # color_iter = iter(DISTINCT_COLORS)

        for idx in opened_datasets.keys():
            if idx == "gfed":
                Data_TS_Season_model_gfed_handle_GFED(
                    dataset=opened_datasets,
                    mask_index=i,
                    dataset_key_index=idx,
                    netcdf_path="/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/15th_region.nc",
                    axis=ax,
                    unit=unit,
                    output_path=output_path,
                )
            elif idx == "wglc":
                wglcLinePlots(
                    dataset=opened_datasets,
                    mask_index=i,
                    dataset_key_index=idx,
                    axis=ax,
                )

            else:
                Data_TS_Season_model_gfed_handle_else(
                    dataset=opened_datasets,
                    diagnostics_list=diagnostics_list,
                    value=val,
                    mask_index=i,
                    axis=ax,
                    dataset_key_index=idx,
                )

        linePlotMask(
            axis=ax,
            output_path=output_path,
            xlabel_name="Month",
            ylabel_name=f"Total {val} {unit}",
            title_name=f"Natural {val} for {GFED_COVER_LABELS[idx]} ",
            output_file_name=f"{val}_{idx}_SEASON_{GFED_COVER_LABELS[i]}_1997_2020.png",
            rotation_type="Seasonal",
        )


def Data_TS_Season_Regional_handleGFEDLandCoverTypes(
    axis, ax_t, netcdf_filepath, time_gf, unique_land_cover_types
):
    color_iter = iter(DISTINCT_COLORS)
    dataset = xr.open_dataset(netcdf_filepath)
    time_dt = dataset["time"].values
    time_15 = pd.to_datetime(time_dt)
    gd = dataset.data_vars
    gfed_diag = list(gd.keys())
    for ilct_idx, ilct in enumerate(unique_land_cover_types):
        total_burn_area_ilct_15 = dataset[gfed_diag[0]][:, ilct_idx]

        ax_t.plot(
            time_15,
            total_burn_area_ilct_15,
            label=f"iLCT {ilct}: {LAND_COVER_LABELS[ilct]}",
            color=COLOR_MAP(ilct_idx),
        )
    for diag_idx, diag_name in enumerate(gfed_diag[1:]):
        color = next(color_iter)

        axis.plot(
            time_15,
            d[diag_name][:],
            label=f"{diag_name}",
            color=color,
            linewidth=1.5,
        )
    total_burn_area_total_ilct = np.sum(dataset[gfed_diag[0]][:, :], axis=-1)
    ax_t.plot(
        time_gf,
        total_burn_area_total_ilct,
        label="Total ILCT",
        color="black",
        linewidth=1.5,
    )


def Data_TS_Season_Regional_handleGFEDLandCoverGen(
    dataset,
    axis,
    mask_index,
    dataset_key_index,
    ax_t,
    gfed_diag,
    time_gf,
    unique_land_cover_types,
):
    color_iter = iter(DISTINCT_COLORS)
    for ilct_idx, ilct in enumerate(unique_land_cover_types):
        total_burn_area_ilct = dataset[dataset_key_index][gfed_diag[0]][
            :, mask_index, ilct_idx
        ]

        ax_t.plot(
            time_gf,
            total_burn_area_ilct,
            label=f"iLCT {ilct}: {LAND_COVER_LABELS[ilct]}",
            color=COLOR_MAP(ilct_idx),
        )

    for diag_idx, diag_name in enumerate(gfed_diag[1:]):
        color = next(color_iter)

        axis.plot(
            time_gf,
            dataset[dataset_key_index][gfed_diag[diag_idx + 1]][:, i],
            label=f"{diag_name}",
            color=color,
            linewidth=1.5,
        )
    total_burn_area_total_ilct = np.sum(
        dataset[dataset_key_index][gfed_diag[0]][:, mask_index, :], axis=-1
    )
    return total_burn_area_total_ilct


def Data_TS_Season_Regional_handle_GFED(
    dataset,
    axis,
    mask_index,
    dataset_key_index,
    output_path,
    netcdf_filepath="/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/15th_region.nc",
):
    fig_t, ax_t = plt.subplots(figsize=(12, 8), tight_layout=True)
    color_map = plt.get_cmap("tab20")
    gd = dataset["gfed"].data_vars
    gfed_diag = list(gd.keys())

    time_dt = dataset[dataset_key_index]["time"].values
    time_gf = pd.to_datetime(time_dt)

    unique_land_cover_types = dataset["gfed"]["ilct"].values

    # Iterate through each ilct type
    # if 'Total' in target_data.variables and 'Norm' in target_data.variables:

    if mask_index == 15:
        Data_TS_Season_Regional_handleGFEDLandCoverTypes(
            axis, ax_t, netcdf_filepath, time_gf, unique_land_cover_types
        )

    else:
        total_burn_area_total_ilct = Data_TS_Season_Regional_handleGFEDLandCoverGen(
            dataset,
            axis,
            mask_index,
            dataset_key_index,
            ax_t,
            gfed_diag,
            time_gf,
            unique_land_cover_types,
        )

    Data_TS_Season_Regional_handleGFEDPlot(
        axis, mask_index, output_path, ax_t, time_gf, total_burn_area_total_ilct
    )


def Data_TS_Season_Regional_handle_else(
    dataset, diagnostics_list, axis, mask_index, dataset_key_index
):
    color_iter = iter(DISTINCT_COLORS)
    for idn, diag in enumerate(diagnostics_list):
        time_dt = dataset[dataset_key_index]["time"].values
        time = pd.to_datetime(time_dt)
        # print(opened_datasets[idx][diag][:, i])

        color = next(color_iter)
        axis.plot(
            time,
            dataset[dataset_key_index][diag][:, mask_index],
            label=f"{dataset_key_index} {diag}",
            color=color,
            linewidth=1.5,
        )


# read from netcdf
def Data_TS_Season_Regional(
    opened_datasets, diagnostics_list, val, unit, output_path, figure_shape=(12, 8)
):

    for i in MASK_LIST:
        _, ax = plt.subplots(figsize=figure_shape, tight_layout=True)
        for idx in opened_datasets.keys():

            if idx == "gfed":
                Data_TS_Season_Regional_handle_GFED(
                    dataset=opened_datasets,
                    axis=ax,
                    mask_index=i,
                    dataset_key_index=idx,
                    output_path=output_path,
                    netcdf_filepath="/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/15th_region.nc",
                )

            elif idx == "wglc":
                wglcLinePlots(
                    dataset=opened_datasets,
                    axis=ax,
                    mask_index=i,
                    dataset_key_index=idx,
                    monthly=False,
                )

            else:
                Data_TS_Season_Regional_handle_else(
                    dataset=opened_datasets,
                    diagnostics_list=diagnostics_list,
                    axis=ax,
                    mask_index=i,
                    dataset_key_index=idx,
                )
    linePlotMask(
        axis=ax,
        output_path=output_path,
        xlabel_name="Time",
        ylabel_name=f"Total {val} {unit}",
        title_name=f"Total {val} for Mask Value {GFED_COVER_LABELS[idx]} ",
        output_file_name=f"{val}_{idx}_TS_{GFED_COVER_LABELS[i]}_1997_2020.png",
        rotation_type="Regional",
    )


def linePlotsRunner(netcdf_paths, output_path, val, unit):
    if val == "BA":
        opened_datasets = {}
        # obs = input("Observation product: yes or no ")
        obs = "yes"

        if obs.lower() == "yes":
            unit = "Mha"
            print("loading gfed")
            file_path = netcdf_paths["gfed"]
            gfed = xr.open_dataset(file_path)
            opened_datasets["gfed"] = gfed
            print("loaded")

        list_of_sim = ["model", "nudged"]

        # Number of simulations to choose
        # num_inputs = int(input("How many simulations? "))
        num_inputs = 1

        # print("List of simulations to choose from:", list_of_sim)

        # Initialize a dictionary to store the opened datasets

        # Loop to get simulation inputs
        for i in range(num_inputs):
            # user_input = input(f"Enter simulation {i+1}: ")
            user_input = "model"
            if user_input in list_of_sim:
                file_path = netcdf_paths[user_input]
                model = xr.open_dataset(file_path)
                model["BA_Mha"] = (
                    model["BA_tree_Mha"] + model["BA_grass_Mha"] + model["BA_shrub_Mha"]
                )
                opened_datasets[user_input] = model

            else:
                print(f"Invalid simulation: {user_input}")
        # diag_input = input('Specify diagnostics from {BA_Mha, BA_shrub_Mha, BA_grass_Mha, BA_tree_Mha}')
        diag_input = "BA_Mha"
        diagnostics_list = diag_input.split(",")

        print("Plotting...")
        output_path = output_path + "BA/"
        # Data_TS_Season_Regional(opened_datasets, diagnostics_list,val,unit,output_path)
        Data_TS_Season_model_gfed(
            opened_datasets, diagnostics_list, val, unit, output_path
        )
    elif val == "Lightning":
        unit = "[strokes/km²/day]"
        opened_datasets = {}
        obs = input("Observation product: yes or no ")
        if obs.lower() == "yes":

            print("loading wglc")
            file_path = netcdf_paths["wglc"]
            wglc = xr.open_dataset(file_path)
            opened_datasets["wglc"] = wglc
            print("loaded")

            # List of available simulations
        list_of_sim = [
            "lightning_model",
            "lightning_nudged",
            "model_anthro",
            "nudged_anthro",
        ]

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

        diag_input = input(
            "Specify diagnostics from {f_ignCG for modelVs nudged and f_ignHUMAN for anthro}"
        )
        diagnostics_list = diag_input.split(",")

        # Trim and convert to lowercase for consistency
        diagnostics_list = [diag.strip() for diag in diagnostics_list]
        output_path = output_path + "Lightning/"
        Data_TS_Season_Regional(
            opened_datasets, diagnostics_list, val, unit, output_path
        )
        Data_TS_Season_model_gfed(
            opened_datasets, diagnostics_list, val, unit, output_path
        )

    elif val == "Precip":
        unit = "[mm/day]"
        opened_datasets = {}
        list_of_sim = ["precip_model", "precip_nudged"]
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

        diag_input = input("Specify diagnostics from {FLAMM_prec}")
        diagnostics_list = diag_input.split(",")

        # Trim and convert to lowercase for consistency
        diagnostics_list = [diag.strip() for diag in diagnostics_list]
        output_path = output_path + "Precip/"
        Data_TS_Season_Regional(
            opened_datasets, diagnostics_list, val, unit, output_path
        )
        Data_TS_Season_model_gfed(
            opened_datasets, diagnostics_list, val, unit, output_path
        )
    else:
        print("Invalid input for 'BA' or 'Lightning' or 'Precip'")


######################################################
#                    MAPS                            #
######################################################
def maps_data_grid_stat(
    target_data,
    months_of_interest=[[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
    months_name=[["DJF"], ["MAM"], ["JJA"], ["SON"]],
    variable_name_list=["Total", "Crop", "Defo", "Peat"],
    save_path="/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/Model_meanBA_1997_2019_nudged.nc",
):
    # Create an empty xarray dataset to store the monthly means
    monthly_means_data = xr.Dataset()

    # Loop through each variable
    for variable in variable_name_list:
        # Loop through each month set of interest
        for i, months in enumerate(months_of_interest):
            # Select the burned area values for the current month set and variable
            burned_area = target_data[variable].sel(
                time=target_data["time.month"].isin(months)
            )
            print(burned_area)
            # Calculate the mean burned area for each grid cell
            mean_burned_area = burned_area.mean(dim="time")
            # <CROP/PEAT/DEFO/NAT/TOTAL>_meanBA_DJF,
            # Add the mean burned area as a new variable to the monthly_means_data dataset
            variable_name = f"{variable}_meanBA_{months_name[i][0]}"
            monthly_means_data[variable_name] = mean_burned_area
        annual_name = f"{variable}_meanBA_ANN"
        monthly_means_data[annual_name] = target_data[variable].mean(dim="time")

    # Add the latitude and longitude coordinates to the monthly_means_data dataset
    monthly_means_data["lat"] = target_data["lat"]
    monthly_means_data["lon"] = target_data["lon"]

    monthly_means_data.to_netcdf(save_path)


def maps_max_burn_area(
    target_data,
    latitude_coords,
    longitude_coords,
    array_shape=(720, 1440, 12),
    month_mean_array_shape=(720, 1440, 2),
    save_path="/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/GFED5_monthmaxBA_2001_2020.nc",
    variable_name_list=["Total", "Crop", "Peat", "Defo"],
    list_label=["mean", "month"],
):
    print("MAX BURN AREA FOR EACH MONTH")

    # latitude_coords = np.linspace(latitude_shape[0], latitude_shape[1], num=latitude_num)

    # # Create coordinates for the second dimension (1400 points) from -179.9 to 179.9
    # longitude_coords = np.linspace(longitude_shape[0], longitude_shape[1], num=longitude_num)

    # Use numpy.meshgrid to create 2D coordinate grids for latitude and longitude
    longitude_grid, latitude_grid = np.meshgrid(longitude_coords, latitude_coords)

    monthly_means_data = xr.Dataset()

    # Assuming you have the 'data' array containing the values for 'total'
    total = xr.DataArray(
        coords=[latitude_coords, longitude_coords], dims=["lat", "lon"]
    )
    for variable in variable_name_list:
        my_array = np.zeros(array_shape)
        mm_array = np.zeros(month_mean_array_shape)
        data_array = xr.DataArray(
            my_array,
            coords=[
                ("latitude", latitude_coords),
                ("longitude", longitude_coords),
                ("months", MONTHS_NUM),
            ],
        )
        month_mean = xr.DataArray(
            mm_array,
            coords=[
                ("latitude", latitude_coords),
                ("longitude", longitude_coords),
                ("month_mean", list_label),
            ],
        )
        variable_data = target_data[variable]
        for i, month in enumerate(MONTHS_NUM):
            print(month)

            # get data for months across the years
            burned_area = variable_data.sel(time=target_data["time.month"].isin(month))

            mean_burned_area = burned_area.mean(dim="time")

            # store the mean value for each month in each grid cell in the list variable corresponding to the month
            data_array[:, :, i] = mean_burned_area
            # print("total")

        print(data_array[393, 719])
        # print( np.argmax(data_array[:,:,:], axis=2, keepdims=True))

        # get indices of maximum mean val for each grid cell
        # change index to +1
        # argmax_indices = argmax_indices.reshape(argmax_indices.shape + (1,))
        argmax_indices = np.expand_dims(np.argmax(data_array[:, :, :], axis=2), 2) + 1
        max_values = np.expand_dims(np.max(data_array[:, :, :], axis=2), 2)
        # max_values = max_values.reshape(max_values.shape + (1,))

        # Assign the results to month_mean

        # separate the list into 2 variables
        month_mean[:, :, 1] = (
            argmax_indices.squeeze()
        )  # Remove the singleton dimension to match month_mean shape
        month_mean[:, :, 0] = max_values.squeeze()
        # <CROP/PEAT/DEFO/NAT/TOTAL>_monthmaxBA,

        monthly_means_data[f"{variable}_monthmaxBA"] = month_mean

    monthly_means_data.to_netcdf(save_path)


def mapCombineDataset(
    comb_dir_path="/discover/nobackup/kmezuman/E6TpyrEPDnu",
    file_pattern_extension=".aijE6TpyrEPDnu.nc",
    variables_to_extract=["BA_tree", "BA_shrub", "BA_grass"],
    output_path="/discover/nobackup/projects/giss_ana/users/kmezuamn/GFED5/combined_monthly_data.nc",
):
    warnings.filterwarnings("ignore")

    os.chdir(comb_dir_path)

    # Specify the path to store the PNG files
    fnms = []

    # # Open the files using xr.open_mfdataset()
    # target_data = xr.open_mfdataset(fnms)

    years = range(1997, 2020)
    # Open each file and load them into separate Datasets
    datasets = []

    for year in years:
        for month in MONTHLIST:
            file_pattern = f"{month}{year}{file_pattern_extension}"
            file_paths = [f for f in os.listdir(".") if f.startswith(file_pattern)]

            for file_path in file_paths:
                dataset = xr.open_dataset(file_path)
                extracted_dataset = dataset.drop_vars(
                    [
                        var
                        for var in dataset.variables
                        if var not in variables_to_extract
                    ]
                )
                time_stamp = f"{month}{year}"  # Create a time stamp like 'JAN2013'
                extracted_dataset = extracted_dataset.expand_dims(
                    time=[time_stamp]
                )  # Add time as a new dimension
                datasets.append(extracted_dataset)

    # Combine all extracted datasets into a single dataset along the time dimension
    combined_dataset = xr.concat(datasets, dim="time")

    # Save the combined dataset to a NetCDF file
    combined_dataset.to_netcdf(output_path)

    # Close the datasets
    for dataset in datasets:
        dataset.close()
        # maps_max_burn_area(target_data=target_data,)


def getTargetData(file_path):
    target_data = xr.open_dataset(file_path, engine="netcdf4")
    target_data["BA"] = (
        target_data["BA_shrub"] + target_data["BA_tree"] + target_data["BA_grass"]
    )
    time_coords_str = target_data["time"].values
    time_coords = pd.to_datetime(
        time_coords_str, format="%b%Y"
    )  # Assuming 'JAN1997' format
    # Update 'time' coordinate in the target_data dataset
    target_data = target_data.assign_coords(time=time_coords)
    return target_data


def getMultiYearTarget(dir_path):
    fnms = []
    for year in range(2001, 2021):
        pattern = f"BA{year}??.nc"
        file_paths = glob(os.path.join(dir_path, pattern))
        fnms.extend(file_paths)

    # Open the files using xr.open_mfdataset()
    target_data = xr.open_mfdataset(fnms)
    return target_data


def mapRunner(
    target_data_list,
    fnms_input_folder_path="/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/BA",
    maps_data_grid_stat_fnms_save_path="/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/Model_meanBA_1997_2019_nudged.nc",
    maps_max_burn_area_fnms_save_path="/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/GFED5_monthmaxBA_2001_2020.nc",
    target_data_path="/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/combined_monthly_data.nc",
):
    target_data = getMultiYearTarget(fnms_input_folder_path)

    maps_data_grid_stat(
        target_data=target_data,
        months_of_interest=[[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        months_name=[["DJF"], ["MAM"], ["JJA"], ["SON"]],
        variable_name_list=["Total", "Crop", "Defo", "Peat"],
        save_path=maps_data_grid_stat_fnms_save_path,
    )
    maps_max_burn_area(
        target_data,
        latitude_coords=(np.linspace(-89.88, 89.88, num=720)),
        longitude_coords=(np.linspace(-179.9, 179.9, num=1440)),
        array_shape=(720, 1440, 12),
        month_mean_array_shape=(720, 1440, 2),
        save_path=maps_max_burn_area_fnms_save_path,
        variable_name_list=["Total", "Crop", "Peat", "Defo"],
        list_label=["mean", "month"],
    )

    for target_env_value in target_data_list:
        mapCombineDataset(
            comb_dir_path=target_env_value["comb_dir_path"],
            file_pattern_extension=target_env_value["file_pattern_extension"],
            variables_to_extract=target_env_value["variables_to_extract"],
            output_path=target_env_value["output_path"],
        )

        # Update 'time' coordinate in the target_data dataset
        target_data = getTargetData(file_path=target_data_path)
        maps_data_grid_stat(
            target_data=target_data,
            months_of_interest=target_env_value["months_of_interest"],
            months_name=target_env_value["months_name"],
            variable_name_list=target_env_value["variable_name_list"],
            save_path=target_env_value["maps_data_grid_stat_save_path"],
        )
        maps_max_burn_area(
            target_data=target_data,
            latitude_coords=(
                np.linspace(-89.88, 89.88, num=target_env_value["array_shape"][0])
            ),
            longitude_coords=(
                np.linspace(-179.9, 179.9, num=target_env_value["array_shape"][1])
            ),
            array_shape=target_env_value["array_shape"],
            month_mean_array_shape=target_env_value["month_mean_array_shape"],
            save_path=target_env_value["maps_max_burn_area_save_path"],
            variable_name_list=target_env_value["variable_name_list"],
            list_label=target_env_value["list_label"],
        )


######################################################
#             /data2netcdf/PERCIP                    #
######################################################
def percipNudgeFunction(
    model_path,
    file_pattern_end=".aijE6TpyrEPDnu.nc",
    variables_to_extract=["FLAMM_prec"],
    year_range=(1997, 2020),
):
    ######################################################
    #                  NUDGED                            #
    ######################################################

    # Set the directory where your netCDF files are located

    os.chdir(model_path)  # path to model op>
    # List of months and years to consider
    years = range(year_range)  # Update this range with the years you want
    # variables_to_extract = ['fireCount', 'BA_tree', 'BA_shrub', 'BA_grass', 'FLAMM', 'FLAMM_prec', 'f_ignCG', 'f_ignHUMAN']

    # Open each file and load them into separate Datasets
    datasets = []

    for year in years:
        for month in MONTHLIST:
            file_pattern = f"{month}{year}{file_pattern_end}"
            file_paths = [f for f in os.listdir(".") if f.startswith(file_pattern)]

            for file_path in file_paths:
                dataset = xr.open_dataset(file_path)
                extracted_dataset = dataset.drop_vars(
                    [
                        var
                        for var in dataset.variables
                        if var not in variables_to_extract
                    ]
                )
                time_stamp = f"{month}{year}"  # Create a time stamp like 'JAN2013'
                extracted_dataset = extracted_dataset.expand_dims(
                    time=[time_stamp]
                )  # Add time as a new dimension
                datasets.append(extracted_dataset)

    # Access and work with individual Datasets
    for i, dataset in enumerate(datasets):
        print(f"Dataset {i+1}:")
        print(dataset)

    return datasets


def perciepApplyMask(
    datasets,
    regrid_mask_dataset,
    variables_list=["FLAMM_prec"],
    mask_value=(0, True, False),
):
    ##########################################
    #              APPLY MASK                #
    ##########################################
    time_values = []
    total_dest = np.zeros((len(datasets), len(MASK_LIST)))
    # conversion_factor = 86400/1000000
    # conversion_factor = 1/864000*1000000
    # conversion_factor = 1
    for t, data in enumerate(datasets):

        print(data.time)
        time_values.append(data.coords["time"].values[0])

        for var_idx, i in enumerate(variables_list):

            total_model_arr = data[i]

            for mask in MASK_LIST:

                masked_data_array = np.ma.masked_array(
                    total_model_arr,
                    mask=np.where(regrid_mask_dataset[mask] == mask_value),
                )

                print("nonnan count")

                print(np.count_nonzero(~np.isnan(masked_data_array)))
                print("nan count")
                print(np.count_nonzero(np.isnan(masked_data_array)))

                region_total = masked_data_array
                total_dest[t, mask] = np.nansum(region_total)

    xDataArray = xr.DataArray(
        total_dest,
        dims=["time", "mask"],
        coords={
            "time": time_values,
            "mask": MASK_LIST,
        },
        attrs={"units": "mm/day"},
    )
    return xDataArray


def createPrecipNetcdf(
    regrid_mask_path,
    model_output=[
        "/discover/nobackup/projects/giss_ana/users/kmezuman/Precip/model_precipitation.nc",
        "/discover/nobackup/projects/giss_ana/users/kmezuman/Precip/nudged_precipitation.nc",
    ],
):
    regrid_mask = xr.open_dataset(regrid_mask_path)
    regrid_mask = regrid_mask.to_array().values

    model_datasets = percipNudgeFunction(
        model_path="/discover/nobackup/kmezuman/E6TpyrEPDnu",
        file_pattern_end=".aijE6TpyrEPDnu.nc",
        variables_to_extract=["FLAMM_prec"],
        year_range=(1997, 2020),
    )
    model_x_data_array = perciepApplyMask(
        datasets=model_datasets,
        regrid_mask_dataset=regrid_mask,
        variables_list=["FLAMM_prec"],
        mask_value=(0, True, False),
    )

    nudge_datasets = percipNudgeFunction(
        model_path="/discover/nobackup/kmezuman/E6TpyrEPDnu",
        file_pattern_end=".aijE6TpyrEPDnu.nc",
        variables_to_extract=["FLAMM_prec"],
        year_range=(1997, 2020),
    )
    nudge_x_data_array = perciepApplyMask(
        datasets=nudge_datasets,
        regrid_mask_dataset=regrid_mask,
        variables_list=["FLAMM_prec"],
        mask_value=(0, True, False),
    )

    # Create xarray Datasets for model and nudged results
    model_dataset = xr.Dataset({"FLAMM_prec": model_x_data_array})
    nudged_dataset = xr.Dataset({"FLAMM_prec": nudge_x_data_array})

    # Save model and nudged datasets to separate netCDF files
    dataset_list = [model_dataset, nudged_dataset]
    if len(model_output) == len(dataset_list):
        percipNetcdfConversion(model_output, dataset_list)


def percipNetcdfConversion(model_output, datasets):
    try:
        if len(model_output) >= len(datasets):
            for path, index in enumerate(model_output):
                datasets[index].to_netcdf(path)
    except:
        print("[-] Unable to preform file conversion on the datasets")


######################################################
#            /data2netcdf/gfed_15th_region           #
######################################################
def gfed15thRegion(
    dataset_path="/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/gfed_burn_area.nc",
    variable_name_list=["Total", "Crop", "Peat", "Defo", "Regional"],
    mask_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    output_path="/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/15th_region.nc",
):
    BA_sorted = xr.open_dataset(dataset_path)
    dest_dataset = xr.Dataset()
    time = BA_sorted["time"].values

    for variable_name in variable_name_list:
        var_data = BA_sorted[variable_name]
        total_var_list = (
            np.zeros((len(time), 20))
            if variable_name == "Regional"
            else np.zeros((len(time)))
        )
        for mask in mask_list:
            total_var_list += (
                var_data[:, mask, :]
                if variable_name == "Regional"
                else var_data[:, mask]
            )

        # Add total_var_list as a data variable
        if variable_name == "Regional":
            land_cover_types = var_data.ilct.values
            unique_land_cover_types = list(set(land_cover_types))
            dest_dataset[variable_name] = xr.DataArray(
                total_var_list,
                dims=("time", "ilct"),
                coords={"time": time, "ilct": unique_land_cover_types},
            )
            dest_dataset[variable_name].attrs["units"] = "km^2"
        else:
            dest_dataset[variable_name] = xr.DataArray(
                total_var_list, dims=("time"), coords={"time": time}
            )
            dest_dataset[variable_name].attrs["units"] = "km^2"

    # multiply by 10^-4  km^2 to mha
    for var_name in dest_dataset.data_vars:
        var = dest_dataset[var_name]

        # var_sqha = var * conversion_factor_sqm_to_sqha
        var_Mha = var * 1e-4
        dest_dataset[var_name + "_Mha"] = var_Mha
        dest_dataset[var_name + "_Mha"].attrs["units"] = "Mha"
        dest_dataset = dest_dataset.drop_vars(var_name)  # Drop the original variable

    # Save the modified Dataset to a new NetCDF file
    dest_dataset.to_netcdf(output_path, format="netcdf4")

    # Save the Dataset as a NetCDF file
    # ds.to_netcdf(output_path, format='netcdf4')


######################################################
#            /data2netcdf/gfed_mha_conv           #
######################################################
def gfedMhaConvRunner(
    file_path="/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/gfed_burn_area.nc",
    output_file="/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/GFED_Mha.nc",
    units="Mha",
    var_name_extension="_Mha",
):
    warnings.filterwarnings("ignore")
    BA = xr.open_dataset(file_path)
    # BA_sorted = BA.sortby("time")
    ds = BA.sortby("time")
    for var_name in ds.data_vars:
        var = ds[var_name]

        # var_sqha = var * conversion_factor_sqm_to_sqha
        var_Mha = var * SQM_TO_SQHA
        ds[var_name + var_name_extension] = var_Mha
        ds[var_name + var_name_extension].attrs["units"] = units
        ds = ds.drop_vars(var_name)  # Drop the original variable

        # Save the modified Dataset to a new NetCDF file

        ds.to_netcdf(output_file, format="netcdf4")
