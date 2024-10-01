from os import listdir
from os.path import join, isfile
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import json

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
NUM_MONTHS = len(MONTHLIST)
MARKER = "o"
SECONDS_IN_A_YEAR = 60.0 * 60.0 * 24.0 * 365.0
KILOGRAMS_TO_GRAMS = 10.0**3


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


def main():
    utilityRunner()
    pass


if __name__ == "__main__":
    main()
