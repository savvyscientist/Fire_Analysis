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


class TimeAnalysis:

    def __init__(
        self,
        file_path,
        directory_path,
        year_start,
        year_end,
        area_variable_name,
        species,
        sectors,
        simulations,
        legend_array,
        color_array,
    ) -> None:
        self.file_path = file_path
        self.directory_path = directory_path

        self.year_start = year_start
        self.year_end = year_end
        self.years_arr = np.arange(year_start, year_end + 1)

        self.time_netcdf_dataset = nc.Dataset(self.filename, "r")
        self.dataset_area = self.time_netcdf_dataset.variables[area_variable_name]
        self.dataset_earth_surface_area = np.sum(self.dataset_area)
        self.species = species
        self.sectors = sectors
        self.simulations = simulations
        self.legend_array = legend_array
        self.color_array = color_array

    def plotTimeAnalysis(self, figure_size):
        # iterate over species in the species list
        for species_element in self.species:
            # create the matplotlib figure
            plt.figure(figsize=figure_size)
            # plot values
            for legend_index, legend in enumerate(self.legend_array):
                plt.plot(
                    MONTHLIST,
                    self.dest_data[legend_index, :],
                    label=legend,
                    marker=MARKER,
                    color=self.color_array[legend_index],
                )
            # include various metadata for the created plot
            plt.title(
                f"{species_element} Emissions by Sector ({self.year_start} - {self.year_end})"
            )
            plt.xlabel("Month")
            plt.ylabel(f"{species_element} Emissions [Pg]")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                f"{self.directory_path}/plots/fire_repository/Develpment/{species_element}_emissions_by_sector.eps"
            )

    def executeTimeAnalysis(self):
        # This script plots a time series of one year of data (seasonality)
        # of specified emissions sources from two simulations and also calculates
        # the difference between the two
        dest_data = np.zeros((len(self.sectors) * len(self.simulation), NUM_MONTHS))
        for species_element in self.species:
            for month_index, month in enumerate(MONTHLIST):
                row_index = 0
                for simulation_element in self.simulation:
                    # Construct file name
                    filename = f"{self.directory_path}/{species_element}/{month}_1996.taij{simulation_element}.nc"
                    try:
                        parseDataset = nc.Dataset(filename, "r")
                        for sector in enumerate(self.sectors):
                            if (
                                sector != "pyrE_src_hemis"
                                and simulation_element != "E6TomaF40intpyrEtest2"
                            ):
                                var_name = f"{species_element}_{sector}"
                                hemisphere_value = parseDataset.variables[var_name]
                                global_val = hemisphere_value[2,]
                                dest_data[row_index, month_index] = (
                                    global_val
                                    * self.dataset_earth_surface_area
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


class Panels:
    def __init__(
        self, output_directory, total_plots, rows, columns, plot_figure_size
    ) -> None:
        # Define the number of plots and rows
        self.img_paths = []
        self.shape = (rows, columns)
        self.total_plots = total_plots  # give own val
        self.output_dir = output_directory
        self.plot_figure_size = plot_figure_size

    def createPanel(self):
        rows, columns = self.shape[0], self.shape[1]
        fig, axs = plt.subplots(rows, columns, figsize=self.plot_figure_size)
        for row_index in range(rows):
            for column_index in range(columns):
                index = row_index * columns + column_index

                if index >= self.total_plots:
                    break  # Stop if all plots are processed

                img_path = self.img_paths[index]
                img = mpimg.imread(img_path)
                axs[row_index, column_index].imshow(img)
                axs[row_index, column_index].axis("off")
                # Add any additional customizations to the subplots if needed

        # Hide any empty subplots
        for i in range(self.total_plots, rows * columns):
            axs.flatten()[i].axis("off")

        self.plotPanel()

    def plotPanel(self):
        plt.tight_layout()
        plt.savefig(
            join(self.output_dir, f"Model_Combined_Season.png"), dpi=600
        )  # specify fle name
        plt.close()


class CarbonBudget:
    def __init__(self, input_files, output_file, df_variable_names) -> None:
        try:
            self.dataset_one = nc.Dataset(input_files[0], "r")
            self.dataset_two = nc.Dataset(input_files[1], "r")
            self.dataset_three = nc.Dataset(input_files[2], "r")

            self.output_file = output_file

            self.CO2n_pyrE_src_hemis = self.dataset_three.variables[
                "CO2n_pyrE_src_hemis"
            ]
            self.CO2n_pyrE_src = self.dataset_three.variables["CO2n_pyrE_src"]
            self.CO2n_pyrE_src_units = self.CO2n_pyrE_src.units

            self.CO2n_Total_Mass_hemis = self.dataset_one.variables[
                "CO2n_Total_Mass_hemis"
            ]
            self.CO2n_Total_Mass = self.dataset_one.variables["CO2n_Total_Mass"]
            self.CO2n_Total_Mass_units = self.CO2n_Total_Mass.units

            self.C_lab_hemis = self.dataset_two.variables["C_lab_hemis"]
            self.C_lab = self.dataset_two.variables["C_lab"]
            self.C_lab_units = self.C_lab.units

            self.soilCpool_hemis = self.dataset_two.variables["soilCpool_hemis"]
            self.soilCpool = self.dataset_two.variables["soilCpool"]
            self.soilCpool_units = self.soilCpool.units

            self.gpp_hemis = self.dataset_two.variables["gpp_hemis"]
            self.gpp = self.dataset_two.variables["gpp"]
            self.gpp_units = self.gpp.units

            self.rauto_hemis = self.dataset_two.variables["rauto_hemis"]
            self.rauto = self.dataset_two.variables["rauto"]
            self.rauto_units = self.rauto.units

            self.soilresp_hemis = self.dataset_two.variables["soilresp_hemis"]
            self.soilresp = self.dataset_two.variables["soilresp"]
            self.soilresp_units = self.soilresp.units

            self.ecvf_hemis = self.dataset_two.variables["ecvf_hemis"]
            self.ecvf = self.dataset_two.variables["ecvf"]
            self.ecvf_units = self.ecvf.units

            self.destination_variable = df_variable_names
            self.destination_units = [
                (self.CO2n_pyrE_src_units),
                (self.CO2n_Total_Mass_units),
                (self.C_lab_units),
                (self.soilCpool_units),
                (self.gpp_units),
                (self.rauto_units),
                (self.soilresp_units),
                (self.ecvf_units),
                (self.ecvf_units),
                "-",
            ]
            self.destination_CO2 = [
                (self.CO2n_pyrE_src_hemis[2,]),
                (self.CO2n_Total_Mass_hemis[2,]),
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
            ]
            self.total_carbon = [
                "-",
                "-",
                (self.C_lab_hemis[2,]),
                (self.soilCpool_hemis[2,]),
                (self.gpp_hemis[2,]),
                (self.rauto_hemis[2,]),
                (self.soilresp_hemis[2,]),
                (self.ecvf_hemis[2,]),
                (
                    self.gpp_hemis[2,]
                    - self.rauto_hemis[2,]
                    - self.soilresp_hemis[2,]
                    - self.ecvf_hemis[2,]
                ),
                "-",
            ]

        except Exception as error:
            print("[-] User must input three files for this script", error)

    def createDataframe(self):
        df = pd.DataFrame(
            {
                "Varaiable": self.destination_variable_names,
                # 'CO': [(CO_biomass_src_hemis[2,]),(CO_Total_Mass_hemis[2,]), '-','-', '-'],
                "Units": self.destination_units,
                "CO2": self.destination_CO2,
                "Total Carbon": self.total_carbon,
            }
        )
        self.saveDataframe(df.to_string(index=False))

    def saveDataframe(self, formatted_table):
        with open(self.output_file, "w") as file:
            file.write(formatted_table)


def utilityRunner():
    env_json = getEnvironmentVariables()
    for script in env_json["selected_script"]:
        if script == "time_analysis":
            script_env_data = env_json[script]
            timeAnalysisInstance = TimeAnalysis(
                directory_path=script_env_data["input_directory_path"],
                file_path=script_env_data["input_file_path"],
                year_start=script_env_data["year_start"],
                year_end=script_env_data["year_end"],
                area_variable_name=script_env_data["area_variable_name"],
                species=script_env_data["species"],
                sectors=script_env_data["sectors"],
                simulations=script_env_data["simulations"],
                legend_array=script_env_data["legend_array"],
                color_array=script_env_data["color_array"],
            )
            timeAnalysisInstance.executeTimeAnalysis()
            timeAnalysisInstance.plotTimeAnalysis(
                figure_size=script_env_data["figure_size"]
            )

        elif script == "panels":
            script_env_data = env_json[script]
            panelInstance = Panels(
                rows=script_env_data["rows"],
                columns=script_env_data["columns"],
                total_plots=script_env_data["total_plots"],
                plot_figure_size=script_env_data["plot_figure_size"],
                output_directory=script_env_data["output_directory"],
            )
            panelInstance.createPanel()

        elif script == "carbon_budget":
            script_env_data = env_json[script]
            carbonBudgetInstance = CarbonBudget(
                input_files=env_json["input_files"],
                output_file=env_json["output_file"],
                pd_variables=env_json["df_variable_names"],
            )
            carbonBudgetInstance.createDataframe()


def main():
    utilityRunner()
    pass


if __name__ == "__main__":
    main()
