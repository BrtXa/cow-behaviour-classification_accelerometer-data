import os

notebook_mode: int = int(
    input(
        """
    Select notebook mode: 
    1. Google Colab  2. Local
    """
    )
)

if notebook_mode == 1:
    # Run on Colab.
    INPUT_PATH: str = "/content/drive/MyDrive/Ellinbank/video_observation/data/"
    SCRIPT_PATH: str = "/content/drive/MyDrive/Ellinbank/video_observation/training_testing/data_labelling/"
    OUTPUT_PATH: str = "/content/drive/MyDrive/Ellinbank/video_observation/output/"
    os.system(command="cp {}custom_model.py .".format(SCRIPT_PATH))
    os.system(command="cp {}inference.py .".format(SCRIPT_PATH))
    os.system(command="cp {}utils.py .".format(SCRIPT_PATH))
    os.system(command="cp {}operation.py .".format(SCRIPT_PATH))
elif notebook_mode == 2:
    INPUT_PATH: str = "../../../../data/"
    SCRIPT_PATH: str = "./"
    OUTPUT_PATH: str = "./out/"
