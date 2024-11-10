import PyInstaller.__main__
import os
import sys

def create_build_command():
    # Get the directory containing build_exe.py
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths
    models_dir = os.path.join(base_path, 'models')
    rfb_dir = os.path.join(models_dir, 'RFB-320')
    
    # Verify model files exist
    required_files = [
        (os.path.join(models_dir, 'emotion-ferplus-8.onnx'), 'models'),
        (os.path.join(rfb_dir, 'RFB-320.caffemodel'), 'models/RFB-320'),
        (os.path.join(rfb_dir, 'RFB-320.prototxt'), 'models/RFB-320')
    ]
    
    # Check if all required files exist
    for file_path, _ in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            sys.exit(1)
    
    # Create the PyInstaller command
    command = [
        'main.py',
        '--name=EmotionDetection',
        '--onefile',
        '--windowed',
        '--noconfirm',  # Override existing build files
        '--clean',      # Clean cache before building
    ]
    
    # Add data files
    for file_path, dest_dir in required_files:
        # Use appropriate path separator based on OS
        separator = ';' if sys.platform.startswith('win') else ':'
        command.append(f'--add-data={file_path}{separator}{dest_dir}')
    
    # Add required imports
    command.extend([
        '--hidden-import=cv2',
        '--hidden-import=numpy',
        '--hidden-import=tkinter',
        '--hidden-import=onnx',
    ])
    
    return command

if __name__ == "__main__":
    try:
        print("Starting build process...")
        command = create_build_command()
        print("Running PyInstaller with command:", ' '.join(command))
        PyInstaller.__main__.run(command)
        print("Build completed successfully!")
    except Exception as e:
        print(f"Build failed: {str(e)}")
        sys.exit(1)