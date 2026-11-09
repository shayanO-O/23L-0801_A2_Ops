# VisionGuard Engineering Notes and Problem Log

This log documents issues, root causes, and fixes encountered during development, serving as the record for the "what went wrong" sections of the final report.

| Incident # | Phase / Task | Issue Description | Root Cause | Resolution |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Phase 1 / Task 1.1 | Dependency conflict between `tensorflow-cpu==2.21.0` and `mlflow==2.20.2`. | `tensorflow-cpu 2.21.0` requires `protobuf>=6.31.1`, but `mlflow 2.20.2` strictly required `protobuf<6`. | Upgraded pinned MLflow version to `3.18.0` in `requirements-train.txt` and `pins.json`, which natively supports Protobuf 6.x. |
| 2 | Phase 1 / Task 1.4 | DVC crashed with `cannot import name '_DIR_MARK' from 'pathspec.patterns.gitwildmatch'`. | `pathspec` 1.1.1 removed internal attribute `_DIR_MARK` relied upon by DVC 3.59.0. | Pinned `pathspec==0.11.2` in `requirements-train.txt` and `requirements-dev.txt`. |
| 3 | Phase 2 / Task 3.1 | MLflow raised `MlflowException: The filesystem tracking backend (e.g., './mlruns') is in maintenance mode`. | MLflow 3.18.0 deprecates legacy local file storage unless an explicit opt-in environment variable is set. | Set `MLFLOW_ALLOW_FILE_STORE=true` in environment and `src/toy_mlflow.py` to allow local `./mlruns` tracking. |
| 4 | Phase 3 / Task 2.7 | Keras Tuner raised `ModuleNotFoundError: No module named 'tensorboard'` during trial execution. | `keras-tuner` expects `tensorboard.plugins.hparams` to be available. | Installed and pinned `tensorboard==2.21.0` in `requirements-train.txt` matching TensorFlow 2.21.0. |
| 5 | Phase 3 / Task 2.4 | `TypeError: float() argument must be a string or a real number, not 'Functional'` in `experiments.py`. | `train_run` returned `(history, best_epoch, model)` instead of `best_val_loss`. | Updated `train_run` in `src/train.py` to return `float(val_losses[best_epoch - 1])` as the third return value. |

| 6 | Phase 9 / Task 7 | Hugging Face Spaces Docker SDK gated behind paid PRO tier. | Hugging Face policy changed to require paid PRO subscription for all container compute instances. | Transitioned Task 7 to AWS EC2 (Pathway C in brief). Provisioned `t3.small` instance in `ap-southeast-2` running Ubuntu 24.04 with Docker on Port 80, authenticated via `X-API-Key`. |
| 7 | Phase 9 / Task 7 | AWS EC2 `UnauthorizedOperation: ec2:DescribeVpcs explicit deny in service control policy`. | Organization SCP `p-5xlbtkfc` strictly restricts EC2 provisioning to region `ap-southeast-2` (Sydney). | Configured deployment pipeline to target `ap-southeast-2`, where full VPC, Security Group, and EC2 provisioning succeeded. |


---

## Task 7: AWS EC2 Cloud API Deployment Documentation (Pathway C)

### 1. Cloud Architecture & Sizing Rationale
- **Platform**: Amazon Web Services (AWS) Elastic Compute Cloud (EC2).
- **Credit & Tier**: Provisioned under an AWS account with \$100 promotional/student credits.
- **Instance Sizing (`t3.small`)**:
  - `t3.micro` (1 vCPU, 1 GiB RAM) was evaluated but rejected because loading the Champion MobileNetV2 TensorFlow model in memory exceeds 1.2 GiB peak during weights deserialization, reliably triggering Linux Kernel Out-Of-Memory (`OOM-killer`) terminating Python.
  - `t3.small` provides **2 vCPUs and 2 GiB RAM**, offering ample headroom for the Python 3.11 runtime, TensorFlow CPU binary, and concurrent inference requests.
- **Storage**: 20 GiB gp3 General Purpose SSD root volume providing 3,000 IOPS and 125 MB/s baseline throughput.

### 2. Dual Firewall Security Model
1. **Cloud Network Layer (AWS Security Group `23L-0801_A2_Ops`)**:
   - **Port 22 (SSH)**: Strictly restricted to the developer's current public IP (`205.164.155.197/32`). All other internet IPs attempting SSH are dropped at the hypervisor perimeter before reaching the virtual NIC.
   - **Port 80 (HTTP)**: Open globally (`0.0.0.0/0`) to allow stateless client and Postman inference calls.
2. **Host Layer (Ubuntu UFW & Docker iptables)**:
   - Docker daemon manages host iptables forwarding traffic directly from host port 80 to container port 7860.

### 3. Application Authentication & HTTP Boundary
- **HTTP Limitation**: The deployment serves direct IPv4 traffic over HTTP on Port 80. HTTPS/TLS was not configured due to lack of a custom DNS domain and ACME/Let's Encrypt validation constraints for raw dynamic cloud IPs.
- **API Key Protection**: Every `/predict` request must provide the secret `X-API-Key` header. Requests omitting or providing an invalid key receive an immediate `401 Unauthorized` response before touching the TensorFlow inference graph.

### 4. Lifecycle & Credit Management
- **Stopping Saves Compute**: When the instance is in `stopped` state, per-second EC2 compute charges halt, consuming \$0 compute credit.
- **Dynamic IP Warning**: Stopping and starting an EC2 instance without an Elastic IP releases the public IPv4 address back to the AWS pool; upon restart, AWS allocates a new public IP, requiring an update to `CLOUD_API_URL` and client configurations.

### 5. Executed AWS CLI Deployment Commands (Sanitized)

```bash
# 1. Create Key Pair
aws ec2 create-key-pair \
  --key-name "23L-0801_A2_Ops" \
  --key-type rsa \
  --key-format pem \
  --region ap-southeast-2 \
  --query "KeyMaterial" \
  --output text > scratch/23L-0801_A2_Ops.pem

# 2. Create Security Group
aws ec2 create-security-group \
  --group-name "23L-0801_A2_Ops" \
  --description "Security group for 23L-0801 VisionGuard Cloud API" \
  --vpc-id "vpc-0d5231cabcb631152" \
  --region ap-southeast-2

# 3. Authorize Inbound Rules
aws ec2 authorize-security-group-ingress \
  --group-id "sg-0fc0a9d2a99e9d02e" \
  --protocol tcp --port 22 --cidr "205.164.155.197/32" \
  --region ap-southeast-2

aws ec2 authorize-security-group-ingress \
  --group-id "sg-0fc0a9d2a99e9d02e" \
  --protocol tcp --port 80 --cidr "0.0.0.0/0" \
  --region ap-southeast-2

# 4. Launch Ubuntu 24.04 t3.small Instance
aws ec2 run-instances \
  --image-id "ami-01e91e94accc42eec" \
  --instance-type "t3.small" \
  --key-name "23L-0801_A2_Ops" \
  --security-group-ids "sg-0fc0a9d2a99e9d02e" \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":20,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=23L-0801_A2_Ops}]" \
  --region ap-southeast-2

# 5. Remote Docker Setup & Container Launch (Executed via SSH)
ssh -i scratch/23L-0801_A2_Ops.pem ubuntu@3.24.214.144 \
  "sudo apt-get update && sudo apt-get install -y docker.io && sudo systemctl enable --now docker"

# SFTP transfer api/ package (app.py, Dockerfile, requirements.txt, model/champion.keras)
# Build and Run on Port 80
ssh -i scratch/23L-0801_A2_Ops.pem ubuntu@3.24.214.144 \
  "cd api && sudo docker build -t visionguard-cloud:latest . && \
   sudo docker run -d --name vg-api --restart always -p 80:7860 -e API_KEY='[REDACTED]' -e THRESHOLD=0.5252 visionguard-cloud:latest"
```
