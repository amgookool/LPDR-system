import os 

project_folder = "LPDR-system"
index_project_folder = str(os.getcwd()).find(project_folder) + len(project_folder)
project_directory = str(os.getcwd())[0:index_project_folder]
os.chdir(project_directory + "\Datasets")


if __name__ == "__main__":
    testing_directory = ".\Testing"
    training_directory = ".\Training"

    