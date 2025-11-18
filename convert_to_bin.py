"""
This script loads the safetensor mapping file and converts them into invididual bin files.
"""

import json
import struct
import os
from safetensors import safe_open

def convert_safetensors_to_bin(index_path, output_dir):
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    tensor_mapping = {}
    processed_tensors = set()
    
    weight_map = index_data["weight_map"]
    safetensors_files = set(weight_map.values())
    
    for safetensor_file in safetensors_files:
        safetensor_path = os.path.join(os.path.dirname(index_path), safetensor_file)
        
        print(f"processing {safetensor_file}...")
        
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            tensor_names = f.keys()
            
            for tensor_name in tensor_names:
                if tensor_name in processed_tensors:
                    continue
                    
                tensor = f.get_tensor(tensor_name)
                
                tensor_np = tensor.numpy()
                
                bin_filename = tensor_name.replace('.', '_') + '.bin'
                bin_path = os.path.join(output_dir, bin_filename)
                
                with open(bin_path, 'wb') as bin_file:
                    dtype_map = {
                        'float32': 0,
                        'float16': 1, 
                        'int32': 2,
                        'int64': 3
                    }
                    
                    dtype_code = dtype_map.get(str(tensor_np.dtype), 255)
                    
                    bin_file.write(struct.pack('B', dtype_code))
                    bin_file.write(struct.pack('B', tensor_np.ndim))
                    
                    for dim in tensor_np.shape:
                        bin_file.write(struct.pack('i', dim))
                    
                    bin_file.write(tensor_np.tobytes())
                
                tensor_mapping[tensor_name] = bin_filename
                processed_tensors.add(tensor_name)
    
    metadata = {
        "tensor_mapping": tensor_mapping,
        "total_tensors": len(processed_tensors)
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nconversion complete! processed {len(processed_tensors)} tensors.")
    print(f"output directory: {output_dir}")


if __name__ == "__main__":
    index_path = "model/model.safetensors.index.json"
    output_dir = "bin_files"
    
    convert_safetensors_to_bin(index_path, output_dir)