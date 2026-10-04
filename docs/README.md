# Architecture

The full architecture write-up, diagrams, and design rationale live on the
project site, not duplicated here: [sencelium.com/architecture](https://sencelium.com/architecture).

For the code itself, start at `scripts/train_sencelium.py` -- the `Config`
dataclass and the model classes (`SenceliumModel`, `MemoryPool`,
`InnerVoiceLoop`, `ComboBlock`) are the entry points.
