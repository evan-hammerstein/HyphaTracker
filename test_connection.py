import sys
import os
import csv

def process_file(file_path, x_dimension, y_dimension, background, sensitivity, outputs_dir):
    print(f"Processing {file_path} with parameters:")
    print(f"X-Dimension: {x_dimension}, Y-Dimension: {y_dimension}")
    print(f"Background: {background}, Sensitivity: {sensitivity}")

    # Create output folder inside outputs_dir
    base_name = os.path.basename(file_path).rsplit(".", 1)[0]  # Get file name without extension
    output_folder = os.path.join(outputs_dir, f"{base_name}_processed")
    os.makedirs(output_folder, exist_ok=True)

    # Generate processed images (simulated)
    for i in range(5):  # Simulate 5 processed images
        output_file = os.path.join(output_folder, f"processed_image_{i + 1}.png")
        with open(output_file, "w") as f:
            f.write("This is a processed image simulation.")
        print(f"Saved {output_file}")

    # Create a .csv file with parameters
    csv_file = os.path.join(output_folder, "parameters.csv")
    with open(csv_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Parameter", "Value"])
        csv_writer.writerow(["X-Dimension", x_dimension])
        csv_writer.writerow(["Y-Dimension", y_dimension])
        csv_writer.writerow(["Background", background])
        csv_writer.writerow(["Sensitivity", sensitivity])
    print(f"Saved parameters to {csv_file}")

    return csv_file

def main():
    if len(sys.argv) < 7:  # 6 arguments + script name
        print("Usage: script.py <filePath> <xDimension> <yDimension> <background> <sensitivity> <outputsDir>")
        return

    file_path = sys.argv[1]
    x_dimension = int(sys.argv[2])
    y_dimension = int(sys.argv[3])
    background = sys.argv[4]
    sensitivity = int(sys.argv[5])
    outputs_dir = sys.argv[6]

    csv_file = process_file(file_path, x_dimension, y_dimension, background, sensitivity, outputs_dir)
    print(f"Processing complete. CSV file: {csv_file}")

if __name__ == "__main__":
    main()
