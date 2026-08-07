"""Remove the chunk_rescorer stage from a loader config, via the proto API."""

import sys

from google.protobuf import text_format

from proto.root_config_pb2 import RootConfig

src, dst = sys.argv[1], sys.argv[2]

config = RootConfig()
with open(src) as handle:
    text_format.Parse(handle.read(), config)

stages = list(config.data_loader.stage)
victim = next(s for s in stages if s.name == "chunk_rescorer")
upstream = victim.input[0]
print(f"removing '{victim.name}', whose input was '{upstream}'")

kept = [s for s in stages if s.name != "chunk_rescorer"]
for stage in kept:
    for index, name in enumerate(stage.input):
        if name == "chunk_rescorer":
            print(f"  rewiring '{stage.name}'.input -> '{upstream}'")
            stage.input[index] = upstream

del config.data_loader.stage[:]
config.data_loader.stage.extend(kept)

with open(dst, "w") as handle:
    handle.write(text_format.MessageToString(config))
print("wrote", dst)
print("pipeline:", " -> ".join(s.name for s in config.data_loader.stage))
