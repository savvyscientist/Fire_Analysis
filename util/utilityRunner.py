import json
from utilityFunc import (
    getEnvironmentVariables,
    run_time_series_analysis,
)


def utilityRunner():
    env_json = getEnvironmentVariables()
    for script in env_json["selected_script"]:
        if script == "time_analysis_version_two":
            script_env_data = env_json[script]
            run_time_series_analysis(
                folder_data_list=script_env_data["folders"],
                time_analysis_figure_data=script_env_data["time_analysis_figure_data"],
            )

        # elif script == "time_analysis":
        #     script_env_data = env_json[script]
        #     timeAnalysisRunner(
        #         file_path=script_env_data["input_file_path"],
        #         species=script_env_data["species"],
        #         sectors=script_env_data["sectors"],
        #         simulations=script_env_data["simulations"],
        #         directory_path=script_env_data["input_directory_path"],
        #         area_variable_name=script_env_data["area_variable_name"],
        #         year_start=script_env_data["year_start"],
        #         year_end=script_env_data["year_end"],
        #         legend_array=script_env_data["legend_array"],
        #         color_array=script_env_data["color_array"],
        #         figure_size=script_env_data["figure_size"],
        #     )
        # elif script == "panels":
        #     script_env_data = env_json[script]
        #     panelRunner(
        #         rows=script_env_data["rows"],
        #         columns=script_env_data["columns"],
        #         total_plots=script_env_data["total_plots"],
        #         plot_figure_size=script_env_data["plot_figure_size"],
        #         output_directory=script_env_data["output_directory"],
        #     )
        # elif script == "carbon_budget":
        #     script_env_data = env_json[script]
        #     carbonBudgetRunner(
        #         script_env_data["input_files"],
        #         script_env_data["output_file"],
        #         script_env_data["df_variable_names"],
        #     )
        # elif script == "line_plots":
        #     script_env_data = env_json[script]
        #     linePlotsRunner(
        #         netcdf_paths=script_env_data["netcdf_paths"],
        #         output_path=script_env_data["output_path"],
        #         val=script_env_data["val"],
        #         unit=script_env_data["unit"],
        #     )
        # elif script == "maps":
        #     script_env_data = env_json[script]
        #     mapRunner(
        #         target_data_list=script_env_data["target_data_list"],
        #         fnms_input_folder_path=script_env_data["fnms_input_folder_path"],
        #         maps_data_grid_stat_fnms_save_path=script_env_data[
        #             "maps_data_grid_stat_fnms_save_path"
        #         ],
        #         maps_max_burn_area_fnms_save_path=script_env_data[
        #             "maps_max_burn_area_fnms_save_path"
        #         ],
        #         target_data_path=script_env_data["target_data_path"],
        #     )
        else:
            print("No Script Found")


def main():
    utilityRunner()


if __name__ == "__main__":
    main()
