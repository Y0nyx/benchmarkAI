import importlib.util
import subprocess
import time
import sys


def import_or_install(package):
    package_name = package.split('==')[0] if '==' in package else package
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    globals()[package_name] = __import__(package_name)


def install_torch():
    try:
        command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117"
        process = subprocess.Popen(command, shell=True)
        process.wait()
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])


class BenchmarkInfo:
    time: float
    device: str
    epoch: int
    gpu_name: str
    gpu_vram: float
    ram: float

    def __str__(self):
        return f"{self.time},{self.device},{self.epoch}," \
               f"{self.gpu_name},{self.gpu_vram},{self.ram}"


if __name__ == '__main__':
    install_torch()

    with open('requirements.txt', 'r', encoding='utf-16') as f:
        packages = f.read().splitlines()
        for package in packages:
            try:
                import_or_install(package)
            except Exception as e:
                print(e)

    benchmarkInfo = BenchmarkInfo()

    # GPU Info
    import GPUtil
    GPUs = GPUtil.getGPUs()
    for gpu in GPUs:
        benchmarkInfo.gpu_vram = gpu.memoryTotal
        benchmarkInfo.gpu_name = gpu.name

    # RAM
    import psutil
    benchmarkInfo.ram = format(psutil.virtual_memory().total / (1024.0 ** 2), '.2f')

    # model
    import ultralytics
    model = ultralytics.YOLO("yolov8n.pt")
    benchmarkInfo.epoch = 10

    # change to gpu training
    import torch
    if torch.cuda.is_available():
        benchmarkInfo.device = 'cuda'
        model.to('cuda')
    else:
        benchmarkInfo.device = 'cpu'

    # training
    start_time = time.time()
    model.train(data="coco128.yaml", epochs=benchmarkInfo.epoch)
    benchmarkInfo.time = time.time() - start_time

    # dump benchmark info
    import yaml
    with open('output.yaml', 'w') as file:
        yaml.dump(str(benchmarkInfo), file)
