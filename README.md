<h1>CountingStars</h1>

# :books: Project Introduction
This is a repository of the simulator used in the paper "CountingStars: Low-overhead Network-wide Measurement in LEO Mega-constellation Networks".  The repository contains the simulator as well as the scripts used to create the graphs displayed in the evaluation section.

# :rocket: Getting Started
This simulator is developed using the Python programming language and is specifically designed for the Windows operating system.
## Environment requirements
* Python (3.7 or higher)
* Pip package manager

## Installing Project Dependencies
Run the following command in the root directory to configure the third-party libraries（It is recommended to operate in a virtual environment）:

```
pip install -r requirements.txt
```

## Data Preparation
We have securely hosted the complete dataset on the **Zenodo** platform, and we will continue to update the dataset in the future.

### How to obtain data
1.  Visit our Zenodo dataset page: [https://doi.org/10.5281/zenodo.16536868](https://doi.org/10.5281/zenodo.16536868) 
2. Please place the downloaded data files in the ``Simulation Platform`` directory of the project.

# :satellite: Experimental Replication
## Run a single experiment
Under the ``Simulation Platform`` directory, specify the name of the experiment to be run using the ``-e`` or ``--experiment`` parameter.

Experiments that currently support replication:

* Iridium constellation under a 0.1 load: ``Iridium_load_0_1``
* Iridium constellation under a 0.5 load: ``Iridium_load_0_5``
* Iridium constellation under a 0.9 load: ``Iridium_load_0_9``
* Starlink constellation under a 0.1 load: ``Starlink_load_0_1``
* Starlink constellation under a 0.5 load: ``Starlink_load_0_5``
* Starlink constellation under a 0.9 load: ``Starlink_load_0_9``

For the experiments related to the Starlink constellation, we recommend running them on a computer with at least ``32GB`` of memory.

### e.g. Run the experiment "Iridium constellation under a 0.1 load"

```
python run_experiments.py -e Iridium_load_0_1
```

## Adjust the memory size
One of the key comparison dimensions in the paper is the performance of each algorithm under different memory sizes. You can override the memory settings in the configuration file through the --kb parameter in the command line.

### e.g. In the Starlink_load_0_9 experiment, each node was allocated a total memory capacity of 256 KB.

```
python run_experiments.py -e Starlink_load_0_9 --kb 256
```

# :bar_chart: Output Description
## Terminal output
During the simulation process, a progress bar will be displayed. After the simulation is completed, overall statistics and performance evaluation results (such as ARE, WMRE, RE, etc.) for each algorithm will be printed.

## To recreate Figures of section
Under the ``pic`` directory, run the following command:
```
python plot_generator.py
```
This command will utilize our test results to create a ``.png`` file for each set of experiments.
