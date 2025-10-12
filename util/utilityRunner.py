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
            
            # Check if annual is specified 
            annual = script_env_data.get("annual", None)
            
            # If not found at root, look in time_analysis_figure_data
            if annual is None and "time_analysis_figure_data" in script_env_data:
                annual = script_env_data["time_analysis_figure_data"].get("annual", False)
            else:
                # Default to False (monthly data) if not found anywhere
                annual = False

            # Get save_netcdf flag (default to False if not specified)
            save_netcdf = script_env_data.get("save_netcdf", False)
                
            print(f"Annual aggregation setting: {annual}") 
            print(f"Save NetCDF setting: {save_netcdf}")

            
            run_time_series_analysis(
                folder_data_list=script_env_data["folders"],
                time_analysis_figure_data=script_env_data["time_analysis_figure_data"],
                annual=annual,  # Explicitly pass the annual parameter
                save_netcdf=save_netcdf  # Add this parameter
            )

        else:
            print("No Script Found")


def main():
    utilityRunner()


if __name__ == "__main__":
    main()
