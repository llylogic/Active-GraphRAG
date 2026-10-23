"""Sample simultaneous allocation instead of adding independent GPU peaks."""
import threading


class GPUMemorySampler:
    def __init__(self, interval=0.005):
        self.interval = interval
        self.peak_bytes = 0
        self.stop = threading.Event()

    def __enter__(self):
        import torch
        self.torch = torch
        self.devices = list(range(torch.cuda.device_count()))
        def sample():
            while not self.stop.is_set():
                self.peak_bytes = max(self.peak_bytes, sum(torch.cuda.memory_allocated(i) for i in self.devices))
                self.stop.wait(self.interval)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        for i in self.devices:
            self.torch.cuda.synchronize(i)
        self.peak_bytes = max(self.peak_bytes, sum(self.torch.cuda.memory_allocated(i) for i in self.devices))
        self.stop.set()
        self.thread.join()
