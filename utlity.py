from os import listdir
from os.path import join, isfile
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

FIRE_SCRIPTS_PATH = "./"
CONVERSION_SCRIPTS_PATH = "./data2netcdf"

VALID_FIRE_SCRIPTS = [
    join(FIRE_SCRIPTS_PATH, file)
    for file in listdir(FIRE_SCRIPTS_PATH)
    if isfile(join(FIRE_SCRIPTS_PATH, file))
    and file != "scriptModule.py"
    and file.split(".")[-1] == "py"
]

VALID_CONVERSION_SCRIPTS = [
    join(CONVERSION_SCRIPTS_PATH, file)
    for file in listdir(CONVERSION_SCRIPTS_PATH)
    if isfile(join(CONVERSION_SCRIPTS_PATH, file))
    and file != "scriptModule.py"
    and file.split(".")[-1] == "py"
]

VALID_SCRIPTS = VALID_FIRE_SCRIPTS + VALID_CONVERSION_SCRIPTS

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


class FireAnalysisUtility:

    def __init__(self, input_filenames, input_directories, output_filepath="") -> None:
        self.input_filenames = input_filenames
        self.input_directories = input_directories
        self.output_filepath = output_filepath

    def extractVariableDataUnits(self, netCDFDataset, variables):
        units_list = []
        for variable in variables:
            units_list.append(netCDFDataset.variables[variable].units)
        return units_list

    class TimeAnalysis:

        def __init__(
            self,
            year_start,
            year_end,
            area_variable_name,
            species,
            sectors,
            simulations,
            legend_array,
            color_array,
        ) -> None:
            self.filename = self.input_filenames[0] if self.input_filenames else ""
            self.directory = self.input_directories[0] if self.input_directories else ""

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
                    f"{self.directory}/plots/fire_repository/Develpment/{species_element}_emissions_by_sector.eps"
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
                        filename = f"{self.directory}/{species_element}/{month}_1996.taij{simulation_element}.nc"
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

            self.plotTimeAnalysis(figure_size=(12, 8))

    class Panels:
        def __init__(self, total_plots, rows, columns) -> None:
            # Define the number of plots and rows
            self.total_plots = total_plots  # give own val
            self.shape = (rows, columns)
            self.img_paths = []
            self.plot_figure_size = (20, 15)
            self.output_dir = ""

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

            plt.tight_layout()
            plt.savefig(
                join(self.output_dir, f"Model_Combined_Season.png"), dpi=600
            )  # specify fle name
            plt.close()

    class CarbonBudget:
        def __init__(
            self, file_one_variables, file_two_variables, file_three_variables
        ) -> None:
            self.file_one_variables = file_one_variables
            self.file_two_variables = file_two_variables
            self.file_three_variables = file_three_variables
            self.new_variable_names = [
                "Biomass emissions",
                "Atmospheric column load",
                "Vegatation Carbon",
                "Soil Carbon",
                "GPP",
                "Autotrophic Respiration",
                "Soil Respiration",
                "EXCESS C FLUX DUE TO VEG FRACTIONS CHANGE",
                "   Net Flux = GPP - rauto - soilresp -ecvf",
                "Ocean Carbon",
            ]


def main():
    # create FireAnalysis class object
    timeAnalysisInstance = FireAnalysisUtility(
        input_filename=[
            "/discover/nobackup/kmezuman/E6TomaF40intpyrEtest/JAN1996.taijE6TomaF40intpyrEtest.nc"
        ],
        input_directory=["/discover/nobackup/kmezuman"],
    ).TimeAnalysis(
        year_start=1996,
        year_end=1996,
        area_variable_name="axyp",
        species=[
            "NOx",
            "OCB",
            "BCB",
            "NH3",
            "SO2",
            "Alkenes",
            "Paraffin",
            "CO",
        ],
        sectors=["pyrE_src_hemis", "biomass_src_hemis"],
        simulations=["E6TomaF40intpyrEtest", "E6TomaF40intpyrEtest2"],
        legend_array=["pyrE", "defo", "biomass", "pyrE+defo"],
        color_array=["black", "blue", "red", "magenta", "orange", "green"],
    )
    timeAnalysisInstance.executeTimeAnalysis()
    pass


if __name__ == "__main__":
    main()
