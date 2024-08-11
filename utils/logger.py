import datetime
import os

def log_run_details_to_file(experiment_name, experiment_id, run_id, execution_time, model_path, log_dir="./mlruns"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    file_path = os.path.join(log_dir, "run_logs.csv")
    log_content = f"{timestamp},{experiment_name},{experiment_id},{run_id},{execution_time},{model_path}\n"

    if os.path.exists(file_path):
        # Read all content, preserve the header and append new log in the second line
        with open(file_path, 'r+') as file:
            lines = file.readlines()
            if len(lines) > 0 and lines[0].strip() == "timestamp,experiment_name,experiment_id,run_id,execution_time,model_path":
                lines.insert(1, log_content)  # Insert the log line after the header
            else:
                # If the header is incorrect/missing, re-write it
                lines = ["timestamp,experiment_name,experiment_id,run_id,execution_time,model_path\n", log_content] + lines
            file.seek(0)
            file.writelines(lines)  # Write all the lines including header and new log line
            file.truncate()  # Truncate any remaining old data that might linger after overwrite
    else:
        # Create the file and write the header followed by the log line
        with open(file_path, 'w') as file:
            file.write("timestamp,experiment_name,experiment_id,run_id,execution_time,model_path\n")
            file.write(log_content)

    print(f"Logged run details to {file_path}")