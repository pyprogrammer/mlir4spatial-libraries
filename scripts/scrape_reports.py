import glob
import argparse
import re
import collections
import itertools
import pandas as pd
import pathlib

parser = argparse.ArgumentParser()

REGEX_DICT = {
    "CLB": "\| CLB LUTs\*\s+\|.*?\|.*?\|.*?\|\s+(?P<CLB>.*?)\s+\|",
    "DSP": "\| DSPs\s+\|.*?\|.*?\|.*?\|\s+(?P<DSP>.*?)\s+\|",
    "CYC": "Design ran for (?P<CYC>\d+) cycles"
}

parser.add_argument("files", type=str)
parser.add_argument("-fields", type=str, nargs="+", action="append")
parser.add_argument("-output", type=pathlib.Path)
parser.add_argument("-lstrip_parts", type=int, default=0)
parser.add_argument("-rstrip_parts", type=int, default=0)
args = parser.parse_args()

fields = list(itertools.chain(*args.fields))
regexes = [re.compile(REGEX_DICT[k]) for k in fields]

results = collections.defaultdict(dict)

for filename in glob.glob(args.files):
    with open(filename) as f:
        for line in f:
           matches = [regex.match(line) for regex in regexes]
           for match in filter(bool, matches):
               results[filename].update(match.groupdict())

# currently results is in fname: {data} form.

transposed = collections.defaultdict(dict)
for fname, data in results.items():
    parts = pathlib.Path(fname).parts
    if args.lstrip_parts:
        parts = parts[args.lstrip_parts:]
    if args.rstrip_parts:
        parts = parts[:-args.rstrip_parts]
    cleaned_fname = str(pathlib.Path(*parts))
    for key, value in data.items():
        transposed[key][cleaned_fname] = value

if args.output.exists():
    old_df = pd.read_csv(args.output)
    new_df = pd.DataFrame(transposed)
    new_df.reset_index(inplace=True)
    cols_to_use = list(set(new_df.columns.difference(old_df.columns).to_list() + ["index"]))
    df = pd.merge(old_df, new_df[cols_to_use], on="index")

else:
    df = pd.DataFrame(transposed)
    df.reset_index(inplace=True)

df.to_csv(args.output, index=False)

