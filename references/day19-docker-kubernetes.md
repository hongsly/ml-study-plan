# Day 19 Quick Reference: Docker & Kubernetes Basics

**Study Date**: 2025-11-15 (Week 3, Day 5)
**Topics**: Docker fundamentals (images, containers, multi-stage builds, GPU support), Kubernetes basics (Pods, Deployments, Services, resource management, autoscaling)
**Knowledge Check Score**: 89.5% (B+/A-)

---

## Docker Fundamentals

### Images vs Containers

**Image**:
- Blueprint/template containing all files and configuration to run an application
- Built from a Dockerfile
- Immutable (doesn't change once built)
- Stored in registries (Docker Hub, ECR, GCR)

**Container**:
- Running instance of an image
- Isolated process with its own filesystem, network, process namespace
- Ephemeral (can be stopped/deleted, data lost unless volumes used)

**Relationship**: Use an image to start one or many containers

```bash
# Build image from Dockerfile
docker build -t my-ml-app:v1 .

# Run container from image (can run multiple containers from same image)
docker run -d --name container1 my-ml-app:v1
docker run -d --name container2 my-ml-app:v1
```

---

## Multi-stage Builds

### Purpose

Separate BUILD stage from RUNTIME stage to create smaller, leaner images by excluding build dependencies.

### Why It Matters for ML

**Problem**: ML images are huge (5-8GB) because they include:
- CUDA dev tools (nvcc compiler, headers)
- Build tools (gcc, g++, cmake, make)
- Dev dependencies (full PyTorch with all optional features)

**Solution**: Multi-stage builds
- Stage 1 (builder): Install everything needed to compile/build
- Stage 2 (runtime): Copy only runtime artifacts (trained model, inference libs)
- Result: 8GB → 2GB (4× smaller)

**Benefits**:
- Faster deployment (less data to transfer)
- Lower registry storage costs
- Smaller attack surface (fewer packages = fewer vulnerabilities)

### Example: ML Inference Image

```dockerfile
# Stage 1: Build stage (heavy dependencies)
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder
RUN apt-get update && apt-get install -y gcc g++ cmake
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
COPY requirements.txt .
RUN pip install -r requirements.txt

# Stage 2: Runtime stage (lean inference image)
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
# Copy only installed packages, not build tools
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY model.pt /app/
COPY inference.py /app/
CMD ["python", "/app/inference.py"]
```

**Key difference**:
- `cuda:*-devel` image: Includes nvcc compiler, headers (8GB)
- `cuda:*-runtime` image: Only runtime libraries (2GB)

---

## GPU Support in Docker

### Requirements

**On host machine**:
- NVIDIA GPU drivers installed
- **NVIDIA Container Toolkit** installed (enables GPU passthrough)

**In Docker command**:
- `--gpus` flag to specify which GPUs to expose

### Common Patterns

```bash
# All GPUs
docker run --gpus all nvidia/cuda:11.8.0-base nvidia-smi

# Specific GPUs (device IDs)
docker run --gpus '"device=0,1"' my-training-image python train.py

# Limit to N GPUs
docker run --gpus 2 my-image python train.py
```

### Common Base Images

- `nvidia/cuda:11.8.0-runtime-ubuntu22.04` - Runtime only (small)
- `nvidia/cuda:11.8.0-devel-ubuntu22.04` - Dev tools (large)
- `pytorch/pytorch:2.0-cuda11.7-cudnn8-runtime` - PyTorch + CUDA runtime
- `tensorflow/tensorflow:latest-gpu` - TensorFlow + GPU support

---

## Kubernetes Basics

### Core Concepts

**Pod**:
- Smallest deployable unit in Kubernetes
- Contains 1+ containers sharing same network namespace and storage volumes
- Containers in a pod share localhost, can communicate via `127.0.0.1`
- Ephemeral: Pods can die and be recreated with new IPs

**Deployment**:
- Manages **ReplicaSets** (desired state: N replicas of a pod)
- Handles **scaling**: Change replicas from 3 → 10
- Handles **rolling updates**: Update to new image version with zero downtime
- Handles **rollbacks**: Revert to previous version if update fails
- Declarative: You specify desired state, K8s makes it happen

**Service**:
- Provides stable IP address and DNS name for a set of pods
- Load balances requests across pod replicas
- Enables service discovery (pods can find each other by name)
- Types: ClusterIP (internal), NodePort (external on specific port), LoadBalancer (cloud provider LB)

### Relationship

```
Service (stable IP: 10.0.0.1)
  ↓ (load balances to)
Deployment (desired: 3 replicas)
  ↓ (creates)
ReplicaSet (ensures 3 pods running)
  ↓ (manages)
3× Pods (ml-inference-abc, ml-inference-def, ml-inference-ghi)
```

**Why this matters**: Pods die and restart with new IPs. Services provide stable endpoint. Deployments ensure desired number of pods always running.

---

## Resource Management

### CPU and Memory

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-training-pod
spec:
  containers:
  - name: trainer
    image: my-training-image:v1
    resources:
      requests:      # Minimum guaranteed (scheduler uses this)
        cpu: "2"     # 2 CPU cores
        memory: "8Gi"
      limits:        # Maximum allowed (hard cap, OOM kill if exceeded)
        cpu: "4"
        memory: "16Gi"
```

**Key concepts**:
- **requests**: Scheduler ensures node has this much available before placing pod
- **limits**: Container can't exceed this (CPU throttled, memory OOM killed)
- **Overcommitment**: Sum of requests < node capacity, but sum of limits can exceed (assumes not all pods hit limits)

### GPU Resources

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-training-pod
spec:
  containers:
  - name: trainer
    image: my-training-image:v1
    resources:
      limits:
        cpu: "4"
        memory: "16Gi"
        nvidia.com/gpu: 1  # Request 1 GPU (only specified in limits)
```

**Key differences from CPU/memory**:
- **Only specify in `limits`**: K8s automatically sets `requests = limits` for GPUs
- **No fractional GPUs**: You get whole GPUs (1, 2, 4), not 0.5 GPU
- **No overcommitment**: GPUs are exclusive (1 GPU = 1 pod at a time)
- **Requires device plugin**: Cluster must have NVIDIA device plugin installed

**Why no `requests` for GPU**: GPUs are not divisible or shareable, so `requests = limits` always. Kubernetes sets requests automatically when you specify limits.

---

## Autoscaling

### Horizontal Pod Autoscaler (HPA)

**What it does**: Automatically scales number of pod replicas based on observed metrics

**How it works**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70  # Scale up if CPU > 70%
```

**When to use for ML inference**:
- Traffic is variable (high during day, low at night)
- Need to handle spikes (Black Friday, product launches)
- Want cost efficiency (scale down when idle)

**Example**: ML inference API gets 100 QPS normally (2 pods enough), but 1000 QPS during peak (need 10 pods). HPA automatically scales 2 → 10 → 2.

### Vertical Pod Autoscaler (VPA)

**What it does**: Automatically adjusts CPU/memory requests and limits for pods

**When to use**:
- Resource needs change over time (model gets bigger, traffic patterns change)
- Don't know optimal resource settings upfront
- Less common than HPA in ML workloads

**HPA vs VPA**:
- **HPA**: More pods (horizontal scaling) - preferred for stateless inference
- **VPA**: Bigger pods (vertical scaling) - useful for training jobs with fixed parallelism

---

## Common Kubernetes Patterns for ML

### Pattern 1: Stateless Inference Service

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-inference
  template:
    metadata:
      labels:
        app: ml-inference
    spec:
      containers:
      - name: inference
        image: my-model:v1
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: ml-inference-service
spec:
  selector:
    app: ml-inference
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer  # External access
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**What this does**:
- Runs 3 replicas of inference container (can scale 2-10)
- Load balances across replicas via Service
- Exposes to internet via LoadBalancer
- Auto-scales based on CPU utilization

### Pattern 2: GPU Training Job

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: ml-training-job
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: my-training-image:v1
        command: ["python", "train.py", "--epochs", "100"]
        resources:
          limits:
            nvidia.com/gpu: 4  # 4 GPUs
            cpu: "16"
            memory: "64Gi"
      restartPolicy: OnFailure  # Retry if fails
      nodeSelector:
        accelerator: nvidia-tesla-a100  # Only schedule on A100 nodes
```

**What this does**:
- Runs one-time training job (not continuous deployment)
- Requests 4 GPUs + 16 CPUs + 64GB memory
- Only schedules on nodes with A100 GPUs
- Retries if pod fails

---

## Interview Q&A

### Q: "Explain Docker images vs containers with an example"

**A**: "A Docker image is like a blueprint or template—it's an immutable snapshot containing all the files, libraries, and configuration needed to run an application. A container is a running instance of that image—it's an isolated process with its own filesystem and network namespace.

For example, I might have a `pytorch-inference:v1` image that contains PyTorch, my trained model, and inference code. I can run 10 containers from this one image to handle 10 concurrent requests. Each container is isolated, so if one crashes, the others keep running.

Think of it like a class (image) and instances (containers) in OOP."

---

### Q: "Why use multi-stage builds for ML applications?"

**A**: "Multi-stage builds separate the BUILD stage from the RUNTIME stage to create smaller images. For ML inference, we need build tools like gcc, cmake, and CUDA dev libraries to compile dependencies, but we don't need them at runtime.

In stage 1, we install everything and build the application. In stage 2, we start from a lean runtime base image and copy only the built artifacts—the trained model and runtime libraries.

This typically reduces ML images from 8GB to 2GB, which means faster deployment, lower registry storage costs, and a smaller attack surface. For inference services that scale to hundreds of pods, this 4× size reduction is critical."

---

### Q: "How does Kubernetes ensure my ML inference service stays available?"

**A**: "Kubernetes uses Deployments to maintain desired state. If I specify `replicas: 3`, the Deployment creates a ReplicaSet that ensures 3 pods are always running.

If a pod crashes or a node dies, Kubernetes automatically creates replacement pods on healthy nodes. But pods have dynamic IPs, so clients can't directly connect to them.

That's where Services come in—they provide a stable IP address and DNS name that load balances across all healthy pods. Even as pods die and restart with new IPs, the Service endpoint stays constant.

For high availability, I'd also add a HorizontalPodAutoscaler to scale up replicas during traffic spikes and scale down during low traffic, ensuring both availability and cost efficiency."

---

### Q: "How do you request GPUs in Kubernetes? Any gotchas?"

**A**: "You specify GPUs in the `resources.limits` section using the resource name `nvidia.com/gpu`. Unlike CPU and memory, you only specify GPUs in `limits`, not `requests`—Kubernetes automatically sets `requests = limits` for GPUs because they're not fractional or shareable.

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

Key gotchas:
1. The cluster must have NVIDIA device plugin installed
2. You can only request whole GPUs (1, 2, 4), not 0.5 GPU
3. GPUs are exclusive—only one pod can use a GPU at a time (no overcommitment)
4. Use node selectors to request specific GPU types: `nodeSelector: accelerator: nvidia-tesla-a100`

For training jobs, I'd also set CPU and memory requests proportional to GPUs—like 4 CPUs and 16GB RAM per GPU—to ensure the pod gets scheduled on nodes with balanced resources."

---

## Common Pitfalls

### Docker

1. **Large images**: Forgetting to use multi-stage builds → 8GB images slow down deployment
   - Fix: Use multi-stage builds with `-runtime` base images

2. **Missing `--gpus` flag**: Container can't see GPU even though host has one
   - Fix: Install NVIDIA Container Toolkit + use `--gpus all`

3. **Confusing build context**: COPY fails because file not in build context
   - Fix: Ensure Dockerfile and files in same directory, or use `.dockerignore`

### Kubernetes

1. **No resource limits**: Pods use unlimited resources, crash other pods
   - Fix: Always set `resources.limits` for CPU/memory

2. **Forgetting GPU device plugin**: GPU pods stuck in Pending state
   - Fix: Ensure cluster has `nvidia-device-plugin` DaemonSet installed

3. **Deployment without Service**: Can't access pods, IPs keep changing
   - Fix: Create Service with selector matching pod labels

4. **HPA with low limits**: HPA wants to scale up but pods hit CPU limit
   - Fix: Set `limits` higher than `requests` to allow bursting

---

## Resources Studied

### Docker
- Docker Official Docs: https://docs.docker.com/get-started/
  - Docker concepts: Images, containers, multi-stage builds
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

### Kubernetes
- Kubernetes Basics Tutorial: https://kubernetes.io/docs/tutorials/kubernetes-basics/
  - Modules 1-3: Pods, Deployments, Services
- Resource Management: https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/
- GPU Scheduling: https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
- Autoscaling: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
