import os
import torch
import torch.nn as nn
import warnings
import inspect
import gc

def set_torch_hub_dir(custom_path=None):
    """
    Set the directory for storing models downloaded via torch.hub.

    Parameters:
    - custom_path (str, optional): A custom path to set as the torch.hub directory.
                                   If not provided, the function will attempt to use
                                   the path specified by the 'ALL_CCFRWORK' environment variable.
    
    Returns:
    - str or None: The path to the torch.hub directory if set successfully, None otherwise.
    
    This function sets the directory where torch.hub should look for or save models.
    If a custom path is provided, it uses that path; otherwise, it looks for the
    'ALL_CCFRWORK' environment variable to determine the path. If neither is available,
    it prints an error message and returns None.
    """
    # Define the environment variable key
    env_var = 'ALL_CCFRWORK'

    # Determine the base path using either the provided custom path or environment variable
    base_path = custom_path if custom_path else os.getenv(env_var)
    
    if base_path:
        # Join the base path with 'models' subdirectory
        hub_dir = os.path.join(base_path, 'models')
        # Set the torch.hub directory to the determined path
        torch.hub.set_dir(hub_dir)
        # Notify the user of the set directory
        print(f"Model directory set to: {hub_dir}")
        return hub_dir
    else:
        # Build an appropriate error message based on the input provided
        error_message = "Custom path is invalid." if custom_path else f"Environment variable '{env_var}' is not set."
        # Print the error message
        print(error_message)
        return None



def download_and_clear_memory(model_name):
    """
    Downloads a specified DINOv2 model and immediately clears it from memory.

    This function is designed to populate the cache with DINOv2 models without retaining
    them in RAM, which is useful for pre-caching models on systems where memory is limited.

    Parameters:
    - model_name (str): The name of the DINOv2 model to download. This name must match
                        the model identifier used by `torch.hub.load`.

    Returns:
    - None
    """
    # Download the model
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    # Immediately release the model from memory
    del model
    #torch.cuda.empty_cache()  # Clear CUDA cache if GPUs are used
    gc.collect()  # Run garbage collection to free up memory


def check_xformers_status():
    xformers_disabled = os.environ.get("XFORMERS_DISABLED")
    print(f"Environment variable 'XFORMERS_DISABLED': {xformers_disabled}")

    # Determine if xformers should be enabled
    XFORMERS_ENABLED = xformers_disabled is None
    try:
        if XFORMERS_ENABLED:
            from xformers.ops import memory_efficient_attention, unbind

            XFORMERS_AVAILABLE = True
            warnings.warn("xFormers is available (Attention)")
        else:
            warnings.warn("xFormers is disabled (Attention)")
            raise ImportError
    except ImportError:
        XFORMERS_AVAILABLE = False
        warnings.warn("xFormers is not available (Attention)")

    print(f"xFormers Enabled: {XFORMERS_ENABLED}")
    print(f"xFormers Available: {XFORMERS_AVAILABLE}")
    return XFORMERS_ENABLED, XFORMERS_AVAILABLE

# Define a function that will be our hook for inspecting outputs
def inspect_attention_hook(module, input, output):
    # Print output to see its structure and dimensions
    print("Output of attention layer:", output)
    print("Shape of output:", output.shape if torch.is_tensor(output) else [o.shape for o in output])


def inspect_model(model):
    # Print a basic description of the model object
    print("Model Description:")
    print(model)
    
    # List all attributes and methods of the model
    print("\nList of all attributes and methods in the model:")
    print(dir(model))
    
    # Check if the model has a built-in way to return attention maps
    print("\nChecking for self-attention retrieval capability:")
    if hasattr(model, 'get_last_selfattention'):
        print("Model supports getting the last self-attention directly.")
    else:
        print("Model does not support getting the last self-attention directly. We might need to modify the model.")
    
    # Check for existence of a specific attribute for attention maps
    print("\nChecking if model has 'get_attention_map' attribute:")
    print(hasattr(model, 'get_attention_map'))
    
    # Print the current attention layer's setup to understand its configuration
    print("\nCurrent attention layer configuration:")
    print(model.blocks[-1].attn)
    
    # Print the source code of the MemEffAttention class
    print("\nMemEffAttention class signature and source code:")
    attention_class = model.blocks[-1].attn.__class__
    print("Constructor signature:")
    print(inspect.signature(attention_class.__init__))
    print("Source code:")
    print(inspect.getsource(attention_class))


import torch

def test_attention_output(model, device):
    # Define a function that will be our hook for inspecting outputs
    def inspect_attention_hook(module, input, output):
        # Print output to see its structure and dimensions
        print("Output of attention layer:", output)
        print("Shape of output:", output.shape if torch.is_tensor(output) else [o.shape for o in output])
    
    # Attach the hook to the last block's attention layer
    last_block = model.blocks[-1].attn
    hook = last_block.register_forward_hook(inspect_attention_hook)
    
    # Prepare a dummy input and run a forward pass
    dummy_input = torch.randn(1, 3, 518, 518).to(device)
    try:
        _ = model(dummy_input)
    finally:
        # Remove the hook immediately after use to clean up
        hook.remove()

    print("Forward pass complete and hook removed.")



def modified_forward(model, return_attn=False):
    original_forward = model.forward

    def forward(*args, **kwargs):
        # Check if return_attn is specified in the call to forward
        return_attn = kwargs.pop('return_attn', False)
        
        # Use the original forward to compute the output
        output = original_forward(*args, **kwargs)
        
        if return_attn:
            # Assume we modify the model to have an attribute `last_attention_map`
            # in its attention layer as previously described
            attn_maps = [block.attn.last_attention_map for block in model.blocks if hasattr(block.attn, 'last_attention_map')]
            return output, attn_maps
        else:
            return output

    return forward



def dinov2_modifier(model):
    model.forward = modified_forward(model)
    return model









