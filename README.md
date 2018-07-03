# Visual Question Answering through Modal Dialogue

## Pre-requisites

The following are a couple of instructions that must be gone through in order to execute different (or all) sections of this project.

1. Clone the project, replacing ``VQAMD`` with the name of the directory you are creating:

        $ git clone https://github.com/sominwadhwa/VQAMD.git VQAMD
        $ cd VQAMD

2. Make sure you have ``python 3.4.x`` running on your local system. If you do, skip this step. In case you don't, head
head [here](https://www.python.org/downloads/).

3. ``virtualenv`` is a tool used for creating isolated 'virtual' python environments. It is advisable to create one here as well (to avoid installing the pre-requisites into the system-root). Do the following within the project directory:

        $ [sudo] pip install virtualenv
        $ virtualenv --system-site-packages VQAMD
        $ source VQAMD/bin/activate

To deactivate later, once you're done with the project, just type ``deactivate``.

4. Install the pre-requisites from ``requirements.txt`` & run ``tests/init.py`` to check if all the required packages were correctly installed:

        $ pip install -r requirements.txt
        $ python test/init.py

You should see an output - ``Imports successful. Good to go!``

## Directory Structure

#### Top-Level Structure:

    .
    .
    ├── data                     # Data used and/or generated
    │   ├── preprocessed
    ├── src                    # Source Files
    │   ├── trainMLP.py
    │   ├── utils.py
    │   ├── requirements.txt
    │   ├── evaluateMLP.py
    ├── test                    # Testing modules (including those for random-control experiments)
    │   ├── init.py              
    ├── LICENSE
    └── README.md
    .
    .


#### Files' Description

## Running Tests

## Contributing to VQAMD

We welcome contributions to our little project.

### Issues

Feel free to submit issues and enhancement requests.

### Contributing

Please refer to each project's style guidelines and guidelines for submitting patches and additions. In general, we follow the "fork-and-pull" Git workflow.

 1. **Fork** the repo VQAMD on GitHub
 2. **Clone** the project to your own machine
 3. **Commit** changes to your own branch
 4. **Push** your work back up to your fork
 5. Submit a **Pull request** so that we can review your changes

NOTE: Be sure to merge the latest from "upstream" before making a pull request!


## Acknowledgements
