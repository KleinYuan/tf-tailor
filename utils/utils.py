import yaml
from box import Box


def load_config(fp):
    return Box(yaml.load(open(fp, 'r').read()))