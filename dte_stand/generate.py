from generator.generator import UniformFlowGenerator
from generator.parsers import sndlib_brain as sndlib_brain_parser
import json

# create generator and set its parameters
generator = UniformFlowGenerator(5, 10000, 20000)

# parse folder that contains dataset files
matrices = sndlib_brain_parser.parse_all('generator/dataset/')

# run the generator
result = generator.generate(matrices, 20000)

# convert results to a string with optional pretty print - can be removed, only result.json() is needed
# But results will be unreadable for a human is pretty print is removed
pretty_res = json.dumps(json.loads(result.json()), indent=4)

# open result file and write the flow data
with open('flows.log', 'w') as f:
    f.write(pretty_res)
