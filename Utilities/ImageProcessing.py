import Utilities as utils
import os

if __name__ == "__main__":
    input_directory = ".\\Utilities\\diagrams\\Letters"
    utils.module_wd(working_directory=input_directory)
    output_directory = os.getcwd() + "\\Sharpen-Letters"
    for file in os.listdir():
        file_path = os.path.join(os.getcwd(), file)
        if os.path.isfile(file):
            print(f"Sharpening file: {file}")
            utils.sharpen_image(file_path, output_directory)
            print(f"Finished Sharpening file: {file}")
