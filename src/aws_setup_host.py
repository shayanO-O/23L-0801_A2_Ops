import os
import sys
import time
import socket
from dotenv import load_dotenv
import paramiko

# Ensure Workspace Root In Python Path
sys.path.insert(0, os.path.abspath("."))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv(".env")

PUBLIC_IP = "3.24.214.144"
KEY_PATH = os.path.join("scratch", "23L-0801_A2_Ops.pem")
USERNAME = "ubuntu"


def run_ssh_command(client, command, print_output=True):
    stdin, stdout, stderr = client.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    if print_output and out.strip():
        print(out.strip())
    if exit_status != 0 and err.strip():
        print(f"Error ({exit_status}): {err.strip()}")
    return exit_status, out, err


def upload_directory_sftp(sftp, local_dir, remote_dir):
    try:
        sftp.mkdir(remote_dir)
    except Exception:
        pass

    for item in os.listdir(local_dir):
        l_path = os.path.join(local_dir, item)
        r_path = f"{remote_dir}/{item}"
        if os.path.isdir(l_path):
            upload_directory_sftp(sftp, l_path, r_path)
        else:
            print(f"Uploading {item} ({round(os.path.getsize(l_path)/1024, 1)} KB)...")
            sftp.put(l_path, r_path)


def setup_and_deploy():
    api_key = os.getenv("API_KEY")
    threshold = os.getenv("THRESHOLD", "0.5252")

    print(f"Connecting to EC2 instance {PUBLIC_IP} via SSH...")
    key = paramiko.RSAKey.from_private_key_file(KEY_PATH)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Wait for SSH Daemon
    connected = False
    for attempt in range(1, 30):
        try:
            ssh.connect(hostname=PUBLIC_IP, port=22, username=USERNAME, pkey=key, timeout=10)
            connected = True
            print("SSH connection established successfully.")
            break
        except Exception as e:
            print(f"Waiting for SSH daemon (attempt {attempt}/30)...")
            time.sleep(5)

    if not connected:
        raise TimeoutError("Could not connect to EC2 instance via SSH.")

    # 1. Install Docker
    print("\n--- Installing Docker on Ubuntu 24.04 ---")
    run_ssh_command(ssh, "sudo apt-get update -y && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io")
    run_ssh_command(ssh, "sudo systemctl start docker && sudo systemctl enable docker")
    run_ssh_command(ssh, "sudo usermod -aG docker ubuntu")
    run_ssh_command(ssh, "sudo docker --version")

    # 2. Upload api/ Package via SFTP
    print("\n--- Uploading api/ directory (with champion model) ---")
    sftp = ssh.open_sftp()
    upload_directory_sftp(sftp, "api", "/home/ubuntu/api")
    sftp.close()
    print("Files uploaded successfully.")

    # 3. Docker Build
    print("\n--- Building Docker image on EC2 ---")
    run_ssh_command(ssh, "cd /home/ubuntu/api && sudo docker build -t visionguard-cloud:latest .")

    # 4. Docker Run (Passing API_KEY and THRESHOLD safely)
    print("\n--- Running container on Port 80:7860 ---")
    run_ssh_command(ssh, "sudo docker rm -f vg-api || true", print_output=False)
    
    # Secure execution: API_KEY is passed directly in the SSH command without logging
    run_cmd = f"sudo docker run -d --name vg-api --restart always -p 80:7860 -e API_KEY='{api_key}' -e THRESHOLD='{threshold}' visionguard-cloud:latest"
    stdin, stdout, stderr = ssh.exec_command(run_cmd)
    container_id = stdout.read().decode("utf-8").strip()
    print(f"Container started. ID: {container_id[:12]}")

    # 5. Check container logs and status
    time.sleep(3)
    print("\n--- Verifying Container Status ---")
    run_ssh_command(ssh, "sudo docker ps --filter 'name=vg-api'")
    
    print("\n--- Recent Container Logs ---")
    run_ssh_command(ssh, "sudo docker logs --tail 20 vg-api")

    ssh.close()
    print("\nEC2 host setup and cloud container deployment complete!")


if __name__ == "__main__":
    setup_and_deploy()
