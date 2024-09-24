from os import listdir
from os.path import join, isfile
from subprocess import call

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


def displayScripts():
    if len(VALID_SCRIPTS) == 0:
        print("[-] No Valid Scripts Found")
    else:
        print("[o] Please select a script you would like to run: ")
        # loops through each script and prints the name of the script and its index
        for index in range(len(VALID_SCRIPTS)):
            script = VALID_SCRIPTS[index]
            file_name = script.split("/")[-1]
            print(f"\t{index}.) {file_name}")
        print()


def handleScriptSelection():
    try:
        # display the current python scripts
        displayScripts()
        # asks the user to select a script based on the number it is assigned
        script_num = int(
            input("Please enter the number for the script you would like to run: ")
        )
        # keep prompting the user until they select a valid script from the shown scripts
        while script_num < -1 or script_num > len(VALID_SCRIPTS) - 1:
            print("[-] Invalid Script number selected, Please try again")
            print("\t[o] Enter (-1) to end the program")
            script_num = int(
                input("Please enter the number for the script you would like to run: ")
            )
        return script_num
    except:
        return -1


def handleScriptExecution(
    script,
):
    # executes remote python script
    call(["python", script])


def main():
    # program continues executing until user is finished (inputs negative one)
    running = True
    while running:

        # obtain the index of the script
        selected_script = handleScriptSelection()
        # pass in the script path to the function
        handleScriptExecution(VALID_SCRIPTS[selected_script])
        if selected_script == -1:
            running = False


if __name__ == "__main__":
    main()
