from collections import Counter
from insar import parsers
import glob

print("Count of extents of Sentinel bounds, rounded to .1")
print(
    Counter((tuple(map(lambda fl: round(fl, 1),
                       parsers.Sentinel(f).extent)) for f in glob.glob("*.SAFE"))))
