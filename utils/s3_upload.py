import boto3
from botocore.exceptions import NoCredentialsError

def upload_file_to_s3(csv_file_path, bucket_name, object_name=None):
    """
    Upload a CSV file to an S3 bucket

    :param csv_file_path: File path to the CSV file to upload
    :param bucket_name: Bucket to upload to
    :param object_name: S3 object name. If not specified, csv_file_path is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use csv_file_path
    if object_name is None:
        object_name = csv_file_path

    # Create an S3 client
    s3_client = boto3.client('s3')

    try:
        # Upload the CSV file
        s3_client.upload_file(csv_file_path, bucket_name, object_name)
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

    print(f"File {csv_file_path} uploaded to {bucket_name}/{object_name}")
    return True

# Example usage
upload_file_to_s3('path/to/your/file.csv', 'your-bucket-name', 'your-object-name.csv')