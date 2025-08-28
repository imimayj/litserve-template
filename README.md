# litserve-template
Template for using Lightning's LitServe framework to deploy text classification models

## Main Functionality

The main LitServe functionality is encapsulated in the litserve folder. To run this service locally (i.e. outside of a Docker build) you can use ``` python -m litserve.entrypoint ```

The __init__.py file encapsulates the LitServe service (decoding requests, loading models, making predictions, and encoding responses/returning them), while the entrypoint file handles starting up the server, identifying the optimal deployment variables based on the given hardware (you may have to tinker around with env vars to get this right depending on where you are serving), mounting the Prometheus logger, and setting up the LitAPI.

| Variable Name     | Description   |
| -------------     | ------------- |
| LOG_REQUEST_IDS = true/false   | If set to 'True' produces extra logs at runtime showing record ID before and after processing |
| MLFLOW_MODEL_PATH = str | This is automatically set from the build-arg MLFLOW_MODEL_URI which is the path pointing to where the model is stored on MLFLOW. This is not essential but is good practice. |
| LITSERVE_FAST_STREAM | use litserve's built-in zmq proxy function |
| LITSERVE_MAX_BATCH_SIZE | max batching for requests to server |
| LITSERVE_WORKERS_PER_DEVICE | number of workers for processing per device |
| PROMETHEUS_MULTIPROC_DIR | temporary directory to mount the prometheus logs |
| LITSERVE_DEVICE_IDS | which device to use for computation |
| LITSERVE_HARDWARE | which hardware the server is running on |
| TORCH_NUM_THREADS | number of threads you want to parallelise model operations on |
| CPU_WORKERS | number of cpu workers to pass in to litservice |

Endpoint Table
| Endpoint Name     | Description   |
| -------------     | ------------- |
| / | Endpoint to call model to classify samples | 
| /health | Healthcheck endpoint for the service |
| /metrics | Prometheus client endpoint for model-specific metrics |
| /info | Information regarding the model and its service |

_ n.b. I would recommend on a linux machine installing only the cpu versions of these models (unless you are deploying to GPU (absolute baller)) _

- torch @ https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp312-cp312-linux_x86_64.whl
- torchvision @ https://download.pytorch.org/whl/cpu/torchvision-0.21.0%2Bcpu-cp312-cp312-linux_x86_64.whl
- torchaudio @ https://download.pytorch.org/whl/cpu/torchaudio-2.6.0%2Bcpu-cp312-cp312-linux_x86_64.whl

Just makes builds slightly less diabolical in terms of time/resource you may not even need :)