# Barebones ML Project Template

This repository provides a **barebones template** for ML projects using **PyTorch, Transformers, and Hydra**. It includes definitions for a Docker development container to streamline the environment setup in VS Code.

## 📋 Prerequisites

1. Docker Engine ([Installation Guide](https://docs.docker.com/engine/install/)): Follow installation steps for your Linux distribution, or use [Docker Desktop](https://docs.docker.com/desktop/) for Windows
2. NVIDIA Container Toolkit ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)): Follow steps for *Installation* **and** *Configuration*

<details><summary><b>Move Docker's default data-dir (Only if neeeded)</b><br></summary>

On my system, I have a lot of free space at `/home`, but very little in docker's default directory. Run the following commands to update Docker to store its data in a different directory.

1. Shutdown Docker service

   ```shell
   sudo systemctl stop docker docker.socket
   sudo systemctl status docker
   ```

2. Move data to the new path (if it's not already there)

   ```shell
   sudo mkdir -p /etc/docker
   sudo rsync -avxP /var/lib/docker/ /home/docker/
   echo '{
     "data-root": "/home/docker"
   }' | sudo tee /etc/docker/daemon.json
   ```

3. Restart the Docker services

   ```shell
   sudo systemctl restart docker
   ```

</details>

#### Useful links:

- [Pytorch compatibility matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
- [Official NVIDIA/CUDA docker images](https://hub.docker.com/r/nvidia/cuda/tags)

## 🚀 Installation

```sh
# clone the repository into your new project directory
git clone <repo-url> my_project
cd my_project

# initialize the environment
make initialize-fresh
```

This make command will:

- Build a **Docker container** with all required dependencies
- Remove the existing `.git` history
- Initialize a new Git repository
- Install **pre-commit hooks** for enforcing code quality

After running make initialize-fresh, your environment is ready. Access the development environment in one of two ways:

- **VS Code DevContainer**: Open the project in VS Code, install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), and select "***Dev Containers: Reopen in Container***" from the command palette to work inside the Docker environment.

- **Docker Run Command**: Dispatch scipts to be run on the container using, e.g.:

  ```shell
  docker run --rm -v $(pwd):/workspace $(basename $(pwd)):latest bash -c "./scripts/train.sh"
  ```

## 📂 Project Structure

```
.
├── configs               # Configuration files for Hydra
│   ├── paths             #
│   │   └── default.yaml  # Default paths configuration
│   └── train.yaml        # Training-specific configuration
├── data                  # Data storage directory
├── logs                  # Logs generated from experiments
├── models                # Saved models
├── notebooks             # Jupyter notebooks for research and experimentation
│   └── template.ipynb    # Notebook template
├── scripts               # Shell scripts for automation
│   ├── eval.sh           # Evaluation script
│   └── train.sh          # Training script
├── src                   # Source code for the project
│   └── train.py          # Barebones train script
├── Dockerfile            # Docker environment setup
├── Makefile              # Makefile for automation (build, train, format, etc.)
├── pyproject.toml        # Python project configuration
├── README.md             # Project documentation
├── requirements.txt      # List of required Python packages
└── setup.py              # Python package setup
```

## 🛠 Features

- **Pre-configured PyTorch environment** using Docker
- **Hydra-based configuration** for flexibility in experiment settings
- **Pre-commit hooks** for enforcing code quality
- **Sensible file structure** to facilitate development
- **Automated setup with Makefile**

## 📝 Notes

- **Configurations**: Modify `configs/train.yaml` and `configs/paths/default.yaml` to adjust training settings and paths.
- **Logs & Checkpoints**: Stored in `logs/` and `models/` respectively.
- **Extensibility**: Add new scripts to `scripts/` or modify `Makefile` for custom workflows.
