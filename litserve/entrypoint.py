import os, dotenv, torch

import litserve as ls
from litserve import LitServer
from prometheus_client import CollectorRegistry, Counter, Histogram, make_asgi_app, multiprocess

from . import LitTCAPI

dotenv.load_dotenv()
port_no = os.getenv("PORT", default=8002)

os.environ['PROMETHEUS_MULTIPROC_DIR'] = "/tmp/prometheus_multiproc_dir"
if not os.path.exists("/tmp/prometheus_multiproc_dir"):
    os.makedirs("/tmp/prometheus_multiproc_dir", exist_ok=True)

registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)

#setting environment variables - for DevOps usage in clusters
zmq = os.getenv("LITSERVE_FAST_STREAM", "false").lower() == "true"
batching = int(os.getenv("LITSERVE_MAX_BATCH_SIZE", 8))
wpd = int(os.getenv("LITSERVE_WORKERS_PER_DEVICE", 2))
device_ids = os.getenv("LITSERVE_DEVICE_IDS", "0")
hardware = os.getenv("LITSERVE_HARDWARE", "auto").lower()
cpu_workers = int(os.getenv("CPU_WORKERS", os.cpu_count()))

class PrometheusLogger(ls.Logger):
    def __init__(self):
        super().__init__()
        self.inference_latency = Histogram(
            "inference_latency_seconds",
            "Time spent performing inference",
            ["model_name"],
            buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5],
            registry=registry
        )

        # Token length tracking
        self.token_length_histogram = Histogram(
            "model_input_token_length",
            "Histogram of token lengths",
            buckets=[64, 128, 256, 384, 512, 1024],
            registry=registry
        )

        # Request count by model
        self.request_counter = Counter(
            "model_requests_total",
            "Total requests received per model",
            ["model_name"],
            registry=registry
        )

        # Response size
        self.response_size_histogram = Histogram(
            "response_size_bytes",
            "Size of model responses in bytes",
            ["model_name"],
            buckets=[256, 512, 1024, 2048, 4096],
            registry=registry
        )
        
    def __getstate__(self):
        return {
            "log_level": self.log_level,
            "log_json": self.log_json
        }  
    
    def process(self, key, value):
        if isinstance(value, dict):
            model_name = value.get("model_name", "ner_model")
            if "latency" in key:
                self.inference_latency.labels(model_name=model_name).observe(value["latency"])
            if "tokens" in value:
                self.token_length_histogram.observe(value["tokens"])
            if "entities" in value:
                self.entity_count_histogram.observe(value["entities"])
            if "label_counts" in value:
                for label, count in value["label_counts"].items():
                    self.label_counter.labels(label=label).inc(count)
            if "response_size" in value:
                self.response_size_histogram.labels(model_name=model_name).observe(value["response_size"])

def get_optimal_config():
    """Determine optimal LitServe config from environment and hardware state."""
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0

    print(f"CUDA available: {cuda_available}, device count: {cuda_count}")

    if hardware == "cuda" or (hardware == "auto" and cuda_available and cuda_count > 0):
        # Parse device list from ENV or default to 0
        try:
            device_list = [int(x.strip()) for x in device_ids.split(",")]
        except ValueError:
            raise ValueError(f"Invalid LITSERVE_DEVICE_IDS: {device_ids}")

        device_list = [d for d in device_list if d < cuda_count]

        if not device_list:
            raise ValueError("No valid CUDA devices specified or available.")

        print(f"Using CUDA devices: {device_list} with {wpd} workers/device")
        return {
            "accelerator": "cuda",
            "devices": device_list,
            "workers_per_device": wpd
        }

    else:
        print(f"Using CPU with {cpu_workers} workers")
        return {
            "accelerator": "cpu",
            "workers_per_device": cpu_workers
        }
        
class RequestTracker(ls.Callback):
    def on_request(self, active_requests: int, **kwargs):
        print(f"Active requests: {active_requests}", flush=True)

if __name__ == "__main__":
    config = get_optimal_config()
    prometheus_logger = PrometheusLogger()
    prometheus_logger.mount(path="/metrics", app=make_asgi_app(registry=registry))
    api = LitTCAPI()
    server = ls.LitServer(
        api, 
        accelerator=config["accelerator"], 
        devices=config.get("devices", None), 
        workers_per_device = config["workers_per_device"],
        api_path='/',
        healthcheck_path='/health',
        info_path='/info',
        loggers=prometheus_logger,
        callbacks=RequestTracker(),
        track_requests=True,
        fast_queue = zmq
        )
    print(f"starting LitServe on port {port_no} with config: {config}")
    server.run(port=port_no)