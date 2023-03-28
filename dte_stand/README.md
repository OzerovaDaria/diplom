# Input data format explanation

### topology

Topology file is in gml format. Current example is taken from topology zoo. Link parameter "bandwidth" and "current_bandwidth" were added by hand.

### topology_changes

This file holds how topology changes over time in json format.  
Dict keys are moments of time in milliseconds when the change is detected. This means that if, for example, there are dict keys "10000" and "50000", then at any moment of time between 10000ms and 50000ms the topology has changes described by "10000" dict.  
Inside the change dict are two lists. They describe which nodes and links are missing. Missing node automatically means that all its links are also missing. If both lists are empty, it means there are no changes to topology.

### flows

This json is a list of all flows present throughout the experiment. Each flow has start and end time in milliseconds. Flow bandwidth is represented as a dict similar to topology changes: dict keys are moments of time when flow bandwidth changes

### config.yaml

This file holds config for the experiment input data located in the folder. Values explanation:  
**lsdb_period** - model time interval between algorithm iterations. Note: if topology change was detected, time between iterations may be shorter than period, but never longer  
**iterations** - amount of algorithm iterations  
**log_path** - path to output file with log  
**log_level** - logging level  
**hash_function, algorithm, path_calculator** - python import paths to implementation of the related component. These paths must be importable from main.py.  

### logging.yaml

This file contains logging config. Refer to standard python module "logging", function "dictConfig" for format reference

# Extension guide

There are 3 replaceable components in the stand: hash function, hash-weights calculation algorithm and path calculator. They are located in corresponding folders inside the dte_stand folder.    
Every folder has a base.py file which contains a base class for a component, and a dummy.py file which contains a simple example of component class's interfaces.  
In order to create a new component implementation, you need to inherit from the base class and implement the one abstract function that exists in the base class.

Note: interfaces are not final! any suggestions and fixes are welcome

Current interfaces are described below.

### Algorithm

Algorithm has a single abstract function "step" which is expected to perform one iteration of the algorithm. This function gets topology as an input parameter and must return a HashWeights object. Its structure and methods can be found in data_structures/hash_weights.py

### Hash function

Hash function has one function to implement: _choose_nexthop which must choose one nexthop. As parameters it receives a list of bucket objects (located in data_structures/hash_weights.py). Bucket represents an edge in the graph and its hash-weight. Only buckets that are allowed to be chosen (according to the paths that were build by path calculator) are present in the list. Edge in the graph is represented as GraphPathElement object. Hash function must return a GraphPathElement object from the chosen bucket.

### Path calculator

This component calculates the list of available paths in the graph. Parameters: topology, source node, destination node. Return value: list of paths, where one path is a list of GraphPathElement objects (GraphPathElement object corresponds to an edge in the graph)

# How to use generator

Folder 'generator' contains classes that generate the list of flows from traffic matrices. Currently only one generator is implemented: uniform.
Folder 'parsers' contains traffic matrix parsers for different datasets. Currently only one dataset is implemeted: sndlib's brain dataset

file generate.py can be used to run the generator. To do so, dataset files should be unpacked into a separate folder. Any number of files from the dataset can be taken. Each file contains a single traffic matrix, so the more files are present, the more data points is available to generator, so generated input data will cover a longer experiment. Files taken from the dataset must be sequential.
The following parameters can be set in generate.py:
 - generator parameters - refer to chosen generator's documentation in the code
 - input folder where dataset files are located
 - period between data points in the dataset. Refer to dataset description to get this value. Although from the generator's standpoint it is not required to set period according to dataset. If the dataset specifies 1 minute interval between data points, in code it is allowed to set it to any other value you want. The data will be interpreted according to the period you specified.
 - output file path

Resulting flow bandwidth is just a number generated according to what is given in the dataset. For example for sndlib brain dataset it is bits per second. Other dataset may use different scale. Link bandwidth in topology should be given according to the scale used in the dataset.

# How to use converter

Converter does two things:
 - duplicates every link and orients them in different directions (A-B becomes A->B, B->A)
 - adds parameters "bandwidth" and "current_bandwidth" to links. current_bandwidth should always be 0 in topology file, bandwidth can be set later as you need. If link has "id" parameter (topology zoo topologies do), then one of the links will have "_r" suffix added.

converter accept 2 parameters - path to original file and path where to save the result