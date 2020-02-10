# mbrl-traffic
Macroscopic model-based reinforcement learning framework for control of mixed-autonomy traffic

## Contents

* [Setup Instructions](#setup-instructions)


## Setup Instructions

This repository is meant to be installed alongside the 
[Flow](https://flow-project.github.io/) computational framework mixed-autonomy 
traffic control. If you have already installed Flow and created a 
[conda](https://www.anaconda.com/distribution/) environment during that 
procedure, we recommend you perform the specified setup instruction within 
Flow's environment. In order to do so, run the following command:

```
source activate flow
```

If you have not already installed Flow or would like to create a new 
environment for this repository, that can be done by running the following 
command:

```
conda env create -f environment.yml
```

The new environment can then be activated using the command:

```
source activate mbrl-traffic
```

Finally, install the repository within your environment by running:

```
pip install -e .
```

You can verify that your installation was successful by running the command:

```
python experiments/simulate.py "ring"
```

If your installation was successful, this should generate a simulation of a 
4-lane ring road of length 1000 m.
