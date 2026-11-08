import os
import sys
import time
import stat
import io
import requests
from dotenv import load_dotenv
import boto3

# Ensure Workspace Root In Python Path
sys.path.insert(0, os.path.abspath("."))
load_dotenv(".env")

RESOURCE_NAME = "23L-0801_A2_Ops"
REGION = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")
KEY_PEM_PATH = os.path.join("scratch", f"{RESOURCE_NAME}.pem")
UBUNTU_2404_AMI = "ami-01e91e94accc42eec"  # Ubuntu 24.04 LTS x86_64 in ap-southeast-2


def get_current_public_ip() -> str:
    # Query Current External Public IPv4 Address
    for url in ["https://checkip.amazonaws.com", "https://api.ipify.org", "https://ifconfig.me"]:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                ip = resp.text.strip()
                if ip:
                    return ip
        except Exception:
            continue
    raise RuntimeError("Unable to determine current public IP address.")


def deploy_aws_infrastructure():
    ak = os.getenv("AWS_ACCESS_KEY_ID")
    sk = os.getenv("AWS_SECRET_ACCESS_KEY")
    api_key = os.getenv("API_KEY")
    threshold = os.getenv("THRESHOLD", "0.5252")

    if not ak or not sk:
        raise ValueError("AWS credentials missing in .env")

    session = boto3.Session(aws_access_key_id=ak, aws_secret_access_key=sk, region_name=REGION)
    ec2 = session.client("ec2")
    current_ip = get_current_public_ip()
    print(f"Current Public IP for SSH whitelist: {current_ip}")
    print(f"Target AWS Region: {REGION}")

    # 1. Manage EC2 Key Pair
    os.makedirs("scratch", exist_ok=True)
    existing_keys = ec2.describe_key_pairs(Filters=[{"Name": "key-name", "Values": [RESOURCE_NAME]}])["KeyPairs"]
    
    if existing_keys and not os.path.exists(KEY_PEM_PATH):
        print(f"Existing key pair found in AWS but {KEY_PEM_PATH} missing locally. Recreating key pair...")
        ec2.delete_key_pair(KeyName=RESOURCE_NAME)
        existing_keys = []

    if not existing_keys:
        print(f"Creating new EC2 Key Pair: {RESOURCE_NAME}...")
        key_resp = ec2.create_key_pair(KeyName=RESOURCE_NAME, KeyType="rsa", KeyFormat="pem")
        with open(KEY_PEM_PATH, "w", encoding="utf-8") as f:
            f.write(key_resp["KeyMaterial"])
        print(f"Saved private key to {KEY_PEM_PATH}")
    else:
        print(f"Using existing EC2 Key Pair: {RESOURCE_NAME}")

    # Enforce Read-Only Permissions On Key
    try:
        os.chmod(KEY_PEM_PATH, stat.S_IRUSR)
    except Exception:
        pass

    # 2. Manage Security Group
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        vpcs = ec2.describe_vpcs()["Vpcs"]
    default_vpc_id = vpcs[0]["VpcId"]
    print(f"VPC ID: {default_vpc_id}")

    existing_sgs = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [RESOURCE_NAME]}, {"Name": "vpc-id", "Values": [default_vpc_id]}]
    )["SecurityGroups"]

    if existing_sgs:
        sg_id = existing_sgs[0]["GroupId"]
        print(f"Using existing Security Group: {RESOURCE_NAME} ({sg_id})")
    else:
        print(f"Creating Security Group: {RESOURCE_NAME}...")
        sg_resp = ec2.create_security_group(
            GroupName=RESOURCE_NAME,
            Description=f"Security group for {RESOURCE_NAME} VisionGuard Cloud API",
            VpcId=default_vpc_id,
            TagSpecifications=[{
                "ResourceType": "security-group",
                "Tags": [{"Key": "Name", "Value": RESOURCE_NAME}]
            }]
        )
        sg_id = sg_resp["GroupId"]
        print(f"Created Security Group ID: {sg_id}")

    # Configure Inbound Rules: SSH (22) from Current IP, HTTP (80) from Anywhere
    rules_to_authorize = [
        {
            "IpProtocol": "tcp",
            "FromPort": 22,
            "ToPort": 22,
            "IpRanges": [{"CidrIp": f"{current_ip}/32", "Description": "SSH administrative access from developer IP"}]
        },
        {
            "IpProtocol": "tcp",
            "FromPort": 80,
            "ToPort": 80,
            "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "HTTP public inference API access"}]
        }
    ]

    for rule in rules_to_authorize:
        try:
            ec2.authorize_security_group_ingress(GroupId=sg_id, IpPermissions=[rule])
            print(f"Authorized ingress: Port {rule['FromPort']} from {rule['IpRanges'][0]['CidrIp']}")
        except ec2.exceptions.ClientError as ce:
            if "InvalidPermission.Duplicate" in str(ce):
                pass
            else:
                print(f"Note on SG rule {rule['FromPort']}: {ce}")

    # 3. Check Or Launch EC2 Instance
    instances = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [RESOURCE_NAME]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}
        ]
    )

    target_instance = None
    for r in instances["Reservations"]:
        for inst in r["Instances"]:
            target_instance = inst
            break

    if target_instance:
        instance_id = target_instance["InstanceId"]
        state = target_instance["State"]["Name"]
        print(f"Found existing instance: {instance_id} (State: {state})")
        if state == "stopped":
            print(f"Starting stopped instance {instance_id}...")
            ec2.start_instances(InstanceIds=[instance_id])
    else:
        print(f"Launching new t3.small EC2 Instance: {RESOURCE_NAME}...")
        run_resp = ec2.run_instances(
            ImageId=UBUNTU_2404_AMI,
            InstanceType="t3.small",
            KeyName=RESOURCE_NAME,
            SecurityGroupIds=[sg_id],
            MinCount=1,
            MaxCount=1,
            BlockDeviceMappings=[
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize": 20,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True
                    }
                }
            ],
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": RESOURCE_NAME}]
            }]
        )
        instance_id = run_resp["Instances"][0]["InstanceId"]
        print(f"Launched Instance ID: {instance_id}")

    # Wait For Instance To Be Running
    print(f"Waiting for instance {instance_id} to reach 'running' state...")
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])

    # Retrieve Public IP
    inst_desc = ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]
    public_ip = inst_desc.get("PublicIpAddress")
    print(f"\n=======================================================")
    print(f" Instance {instance_id} is RUNNING!")
    print(f" Public IPv4: {public_ip}")
    print(f" Cloud API Base URL: http://{public_ip}")
    print(f"=======================================================\n")

    return instance_id, public_ip, sg_id


if __name__ == "__main__":
    deploy_aws_infrastructure()
