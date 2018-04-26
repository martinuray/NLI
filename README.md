# NLI

## Setup
All dependencies are defined in pip file. To install, simply execute `pip install -r requirements.txt`.

## Execution
To execute, simply run `export PYTHONPATH=$PYTHONPATH:. && python models/main.py`.

the different steps of the training process are defined in models/main.py and must be set respectively.

## Known Issues
- if an error like that arrises
```bash
2018-04-26 18:59:47.229177: F ./tensorflow/stream_executor/lib/statusor.h:212] Non-OK-status: status_ status: Failed precondition: could not dlopen DSO: libcupti.so.8.0; dlerror: libcupti.so.8.0: cannot open shared object file: No such file or directory
Aborted
```
you need to export the path to the cuda library:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```
