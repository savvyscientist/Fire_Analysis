from os import listdir
import os
from os.path import join, isfile
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import json
import matplotlib.dates as mdates
import xarray as xr

SCRIPTS_ENV_VARIABLES = "fire_analysis_env_variables.json"
MONTHLIST = [
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
]
DISTINCT_COLORS = [
    "#FF0000",
    "#00FF00",
    "#0000FF",
    "#FFFF00",
    "#FF00FF",
    "#00FFFF",
    "#FFA500",
    "#008000",
    "#800080",
    "#008080",
    "#800000",
    "#000080",
    "#808000",
    "#800080",
    "#FF6347",
    "#00CED1",
    "#FF4500",
    "#DA70D6",
    "#32CD32",
    "#FF69B4",
    "#8B008B",
    "#7FFF00",
    "#FFD700",
    "#20B2AA",
    "#B22222",
    "#FF7F50",
    "#00FA9A",
    "#4B0082",
    "#ADFF2F",
    "#F08080",
]

MASK_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
MONTHS_NUM = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

GFED_COVER_LABELS = {
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
    14: "AUST",
    15: "Total",
}


LAND_COVER_LABELS = {
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
    19: "",
}


NUM_MONTHS = len(MONTHLIST)
MARKER = "o"
SECONDS_IN_A_YEAR = 60.0 * 60.0 * 24.0 * 365.0
KILOGRAMS_TO_GRAMS = 10.0**3
COLOR_MAP = plt.get_cmap("tab20")


def getEnvironmentVariables():
    return json.load(open(SCRIPTS_ENV_VARIABLES, "r"))


def plotTimeAnalysis(
    data_set,
    directory_path,
    year_start,
    year_end,
    species,
    legend_array,
    color_array,
    figure_size,
):
    # iterate over species in the species list
    for species_element in species:
        # create the matplotlib figure
        plt.figure(figsize=figure_size)
        # plot values
        for legend_index, legend in enumerate(legend_array):
            plt.plot(
                MONTHLIST,
                data_set[legend_index, :],
                label=legend,
                marker=MARKER,
                color=color_array[legend_index],
            )
        # include various metadata for the created plot
        plt.title(f"{species_element} Emissions by Sector ({year_start} - {year_end})")
        plt.xlabel("Month")
        plt.ylabel(f"{species_element} Emissions [Pg]")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f"{directory_path}/plots/fire_repository/Develpment/{species_element}_emissions_by_sector.eps"
        )


def executeTimeAnalysis(
    file_path,
    species,
    sectors,
    simulations,
    directory_path,
    area_variable_name,
):
    # obtain the dataset earth surface area data
    time_netcdf_dataset = nc.Dataset(file_path, "r")
    dataset_area = time_netcdf_dataset.variables[area_variable_name]
    dataset_earth_surface_area = np.sum(dataset_area)
    # This script plots a time series of one year of data (seasonality)
    # of specified emissions sources from two simulations and also calculates
    # the difference between the two
    dest_data = np.zeros((len(sectors) * len(simulations), NUM_MONTHS))
    for species_element in species:
        for month_index, month in enumerate(MONTHLIST):
            row_index = 0
            for simulation_element in simulations:
                # Construct file name
                filename = f"{directory_path}/{species_element}/{month}_1996.taij{simulation_element}.nc"
                try:
                    parseDataset = nc.Dataset(filename, "r")
                    for sector in enumerate(sectors):
                        if (
                            sector != "pyrE_src_hemis"
                            and simulation_element != "E6TomaF40intpyrEtest2"
                        ):
                            var_name = f"{species_element}_{sector}"
                            hemisphere_value = parseDataset.variables[var_name]
                            global_val = hemisphere_value[2,]
                            dest_data[row_index, month_index] = (
                                global_val
                                * dataset_earth_surface_area
                                * SECONDS_IN_A_YEAR
                                * KILOGRAMS_TO_GRAMS
                            )
                            row_index += 1
                    parseDataset.close()
                except FileNotFoundError:
                    print(f"File {filename} not found.")
                except Exception as e:
                    print(f"Error reading from {filename}: {str(e)}")
            dest_data[-1, month_index] = (
                dest_data[0, month_index] + dest_data[1, month_index]
            )
    return dest_data


def plotPanel(output_directory):
    plt.tight_layout()
    plt.savefig(
        join(output_directory, f"Model_Combined_Season.png"), dpi=600
    )  # specify fle name
    plt.close()


def createPanel(rows, columns, total_plots, plot_figure_size, output_directory):
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


def obtain_carbon_budget_variables(input_files, output_file, df_variable_names) -> None:
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














# xlabel_name = Time or Month
def line_plot_mask(
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


def wglc_line_plots(dataset, axis, mask_index, dataset_key_index, monthly=True):
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

    line_plot_mask(
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
    line_plot_mask(
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
                wglc_line_plots(
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

        line_plot_mask(
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
        fig, ax = plt.subplots(figsize=figure_shape, tight_layout=True)

        color_map = plt.get_cmap("tab20")

        color_iter = iter(DISTINCT_COLORS)

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
                wglc_line_plots(
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
    line_plot_mask(
        axis=ax,
        output_path=output_path,
        xlabel_name="Time",
        ylabel_name=f"Total {val} {unit}",
        title_name=f"Total {val} for Mask Value {GFED_COVER_LABELS[idx]} ",
        output_file_name=f"{val}_{idx}_TS_{GFED_COVER_LABELS[i]}_1997_2020.png",
        rotation_type="Regional",
    )


def handleLinePlotRun(netcdf_paths, output_path, val, unit):
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






def percipNudgeFunction(model_path, file_pattern_end='.aijE6TpyrEPDnu.nc', variables_to_extract=['FLAMM_prec'], year_range=(1997, 2020)):
    ######################################################
    #                  NUDGED                            #
    ######################################################

    # Set the directory where your netCDF files are located

    os.chdir(model_path) #path to model op>
    # List of months and years to consider
    years = range(year_range)  # Update this range with the years you want
    #variables_to_extract = ['fireCount', 'BA_tree', 'BA_shrub', 'BA_grass', 'FLAMM', 'FLAMM_prec', 'f_ignCG', 'f_ignHUMAN']

    # Open each file and load them into separate Datasets
    datasets = []

    for year in years:
        for month in MONTHLIST:
            file_pattern = f'{month}{year}{file_pattern_end}'
            file_paths = [f for f in os.listdir('.') if f.startswith(file_pattern)]
            
            for file_path in file_paths:
                dataset = xr.open_dataset(file_path)
                extracted_dataset = dataset.drop_vars([var for var in dataset.variables if var not in variables_to_extract])
                time_stamp = f'{month}{year}'  # Create a time stamp like 'JAN2013'
                extracted_dataset = extracted_dataset.expand_dims(time=[time_stamp])  # Add time as a new dimension
                datasets.append(extracted_dataset)

    # Access and work with individual Datasets
    for i, dataset in enumerate(datasets):
        print(f"Dataset {i+1}:")
        print(dataset)
        
    return datasets


def perciepApplyMask(datasets, regrid_mask_dataset,variables_list=['FLAMM_prec'], mask_value=(0,True,False)):
    ##########################################
    #              APPLY MASK                #
    ##########################################
    time_values = []
    total_dest = np.zeros((len(datasets),len(MASK_LIST)))
    #conversion_factor = 86400/1000000
    #conversion_factor = 1/864000*1000000
    #conversion_factor = 1
    for t,data in enumerate(datasets):
            
        print(data.time)
        time_values.append(data.coords['time'].values[0])

        for var_idx,i in enumerate(variables_list):
            
            total_model_arr = data[i]
            
            for mask in MASK_LIST:
            
                masked_data_array = np.ma.masked_array(total_model_arr, mask=np.where(regrid_mask_dataset[mask] == mask_value))
                    
                print("nonnan count")
                
                print(np.count_nonzero(~np.isnan(masked_data_array)))
                print("nan count")
                print(np.count_nonzero(np.isnan(masked_data_array)))
                
                region_total = masked_data_array
                total_dest[t,mask] = np.nansum(region_total)
    
    xDataArray = xr.DataArray(
        total_dest,
        dims=["time", "mask"],
        coords={
            "time": time_values,
            "mask": MASK_LIST,
        },
        attrs={'units': 'mm/day'}
    ) 
    return xDataArray

def createPrecipNetcdf(regrid_mask_path, model_output=["/discover/nobackup/projects/giss_ana/users/kmezuman/Precip/model_precipitation.nc", "/discover/nobackup/projects/giss_ana/users/kmezuman/Precip/nudged_precipitation.nc"]):
    regrid_mask = xr.open_dataset(regrid_mask_path)
    regrid_mask = regrid_mask.to_array().values
    
    model_datasets = percipNudgeFunction(model_path='/discover/nobackup/kmezuman/E6TpyrEPDnu', file_pattern_end='.aijE6TpyrEPDnu.nc', variables_to_extract=['FLAMM_prec'], year_range=(1997, 2020))
    model_x_data_array = perciepApplyMask(datasets=model_datasets, regrid_mask_dataset=regrid_mask,variables_list=['FLAMM_prec'], mask_value=(0,True,False))

    nudge_datasets = percipNudgeFunction(model_path='/discover/nobackup/kmezuman/E6TpyrEPDnu', file_pattern_end='.aijE6TpyrEPDnu.nc', variables_to_extract=['FLAMM_prec'], year_range=(1997, 2020))
    nudge_x_data_array = perciepApplyMask(datasets=nudge_datasets, regrid_mask_dataset=regrid_mask,variables_list=['FLAMM_prec'], mask_value=(0,True,False))

    # Create xarray Datasets for model and nudged results
    model_dataset = xr.Dataset({"FLAMM_prec": model_x_data_array})
    nudged_dataset = xr.Dataset({"FLAMM_prec": nudge_x_data_array})
    
    # Save model and nudged datasets to separate netCDF files
    dataset_list = [model_dataset, nudged_dataset]
    percipNetcdfConversion(model_output, dataset_list)


def percipNetcdfConversion(model_output, datasets):
    try:
        if len(model_output) >= len(datasets):
            for path, index in enumerate(model_output):  
                datasets[index].to_netcdf(path)
    except:
        print("[-] Unable to preform file conversion on the datasets")
        
    
    


def utilityRunner():
    env_json = getEnvironmentVariables()
    for script in env_json["selected_script"]:
        if script == "time_analysis":
            script_env_data = env_json[script]
            data_set = executeTimeAnalysis(
                file_path=script_env_data["input_file_path"],
                species=script_env_data["species"],
                sectors=script_env_data["sectors"],
                simulations=script_env_data["simulations"],
                directory_path=script_env_data["input_directory_path"],
                area_variable_name=script_env_data["area_variable_name"],
            )
            plotTimeAnalysis(
                data_set,
                directory_path=script_env_data["directory_path"],
                year_start=script_env_data["year_start"],
                year_end=script_env_data["year_end"],
                species=script_env_data["species"],
                legend_array=script_env_data["legend_array"],
                color_array=script_env_data["color_array"],
                figure_size=script_env_data["figure_size"],
            )

        elif script == "panels":
            script_env_data = env_json[script]
            createPanel(
                rows=script_env_data["rows"],
                columns=script_env_data["columns"],
                total_plots=script_env_data["total_plots"],
                plot_figure_size=script_env_data["plot_figure_size"],
                output_directory=script_env_data["output_directory"],
            )

        elif script == "carbon_budget":
            script_env_data = env_json[script]
            try:
                (
                    destination_variable_names,
                    destination_units,
                    destination_CO2,
                    total_carbon,
                ) = obtain_carbon_budget_variables(
                    input_files=env_json["input_files"],
                    output_file=env_json["output_file"],
                    pd_variables=env_json["df_variable_names"],
                )
                createDataframe(
                    destination_variable_names,
                    destination_units,
                    destination_CO2,
                    total_carbon,
                )
            except:
                print("[-] Failed to run script")
        elif script == "line_plots":
            script_env_data = env_json[script]
            handleLinePlotRun(
                netcdf_paths=script_env_data["netcdf_paths"],
                output_path=script_env_data["output_path"],
                val=script_env_data["val"],
                unit=script_env_data["unit"],
            )


def main():
    utilityRunner()
    pass


if __name__ == "__main__":
    main()
