# Project setup
1. Clone the project

    ```Shell
    git clone <git link>
    cd Ultra-Fast-Lane-Detection
    ```

2. Next step is to setup the interpreter environment. This does not have to be a conda env.  
   Create a conda virtual environment and activate it

    ```Shell
    conda create -n lane-det python=3.7 -y
    conda activate lane-det
    ```

3. Install dependencies
   torchvision is not compatible with every `cudatoolkit` version. See pytorch documentation to see supported versions.

    ```Shell
    # If you dont have pytorch
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch 

    pip install -r requirements.txt
    ```
