# VisionGuard Final Verification & Submission Manifest

## 1. Project Profile
- **Project Name**: `23L-0801_A2_Ops`
- **Student Full Name**: Shayan Ahmed
- **Roll Number**: `23L-0801`
- **GitHub Repository**: [https://github.com/shayanO-O/23L-0801_A2_Ops](https://github.com/shayanO-O/23L-0801_A2_Ops)
- **DagsHub Repository**: [https://dagshub.com/shayanO-O/23L-0801_A2_Ops](https://dagshub.com/shayanO-O/23L-0801_A2_Ops)
- **Live Cloud API URL**: [http://3.24.214.144](http://3.24.214.144)

---

## 2. Cloud Architecture (Task 7 - Pathway C: AWS EC2)
- **Provider**: Amazon Web Services (AWS)
- **Instance ID**: `i-03f435326ddf75d8b`
- **Instance Name**: `23L-0801_A2_Ops`
- **Instance Type**: `t3.small` (2 vCPUs, 2 GiB Memory, x86_64)
- **AMI**: `ami-01e91e94accc42eec` (`ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-20261023`)
- **Storage**: 20 GiB gp3 root volume
- **Region**: `ap-southeast-2` (Sydney - governed by Organization SCP)
- **Public IPv4**: `3.24.214.144`
- **Security Group**: `23L-0801_A2_Ops` (`sg-0fc0a9d2a99e9d02e`)
  - Inbound Port 22 (SSH): Locked strictly to developer public IP (`205.164.155.197/32`)
  - Inbound Port 80 (HTTP): Open to `0.0.0.0/0` for global API inference
- **Container Engine**: Docker Engine 29.1.3 mapping host port `80` to container port `7860` (`vg-api`)
- **Authentication**: `X-API-Key` required on `/predict`
- **Calibration**: Optimal decision threshold $\tau = 0.5252$ (Youden's J)

---

## 3. Important Post-Grading Cleanup Notice
> [!IMPORTANT]
> **Keep this EC2 instance RUNNING until your assignment marks are released.**
> After marks are officially published:
> 1. Terminate EC2 instance `i-03f435326ddf75d8b` in AWS Console (`ap-southeast-2`).
> 2. Delete the Security Group `23L-0801_A2_Ops` and Key Pair `23L-0801_A2_Ops`.
> 3. Deactivate and delete the IAM access key for user `23L-0801_A2_Ops_deployer`.
