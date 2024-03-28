import argparse
import sys
import importlib
from importlib import resources
import yaml

def load_yaml_config(config_path):
    """Load a YAML config file from a package resource or a filesystem path."""
    if config_path.startswith("@lynx_id/"):
        # Extract the path after the special prefix
        resource_path = config_path[len("@lynx_id/"):]
        # The package path now is the part before the actual file name
        package_path, filename = resource_path.rsplit('/', 1)

        # Adjusting to the new importlib.resources.files() approach
        try:
            # Construct the path to the resource
            resource = resources.files('lynx_id').joinpath(package_path).joinpath(filename)
            if resource.exists():
                # Reading the content of the file
                with resource.open('r') as file:
                    return yaml.safe_load(file)
            else:
                raise FileNotFoundError(f"Resource '{filename}' not found in package 'lynx_id' at '{package_path}'.")
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e))
        except ImportError:
            raise ImportError(f"The package or resource module does not exist.")
    else:
        # It's a filesystem path
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    
    
def simulate_cli_args_from_config(config):
    """Convert config dict to a list of CLI-like arguments."""
    simulated_args = []
    for key, value in config.items():
        simulated_args.append(f'--{key}')
        if not isinstance(value, bool):  # Assuming a flag-style argument for booleans
            simulated_args.append(str(value))
    return simulated_args

def merge_namespaces(base, update):
    """Merge two argparse.Namespace objects, with 'update' taking precedence."""
    # Create a copy of base to avoid modifying the original object
    base_dict = vars(base).copy()
    # Update with the values from update
    base_dict.update(vars(update))
    # Return a new Namespace object based on the merged dictionary
    return argparse.Namespace(**base_dict)

def prepend_command_if_needed(cli_args):
    """Prepend '--command' if the first CLI arg is not an optional argument."""
    if cli_args and not cli_args[0].startswith('--'):
        # Prepend with '--command'
        return ['--command'] + cli_args
    return cli_args

def main():
    #Config file check
    # Initial parser to identify if there is a YAML configuration file
    parser = argparse.ArgumentParser(description="Main interface for lynx_id operations.", add_help=False)
    parser.add_argument('--config', help='Path to YAML configuration file', default=None)
    
    # Parse CLI arguments to find config file
    cli_first_parse_args, cli_remaining_args = parser.parse_known_args()
    
    # Load arguments from YAML file if provided
    if cli_first_parse_args.config:
        config_dict = load_yaml_config(cli_first_parse_args.config)
    else:
        config_dict = dict()
    # Convert config file args to list
    config_fakecli_args = simulate_cli_args_from_config(config_dict)
    
    # Command check
    # Parser to identify if there is a YAML configuration file
    parser = argparse.ArgumentParser(description="Main interface for lynx_id operations.", add_help=False)
    parser.add_argument('--command', choices=['train', 'train_triplets', 'eval', 'infer', 'check_relative_imports'], required=True, help='Subcommand to run')
    
    combined_args = config_fakecli_args + prepend_command_if_needed(cli_remaining_args)
    command_arg, _ = parser.parse_known_args(combined_args)
        
    # Merge base arguments from config file and cli
    # args = merge_namespaces(config_base_args, cli_base_args)

    command_to_module = {
        'train': 'lynx_id.scripts.train.train',
        'train_triplets': 'lynx_id.scripts.train.train_triplets',
        'eval': 'lynx_id.scripts.eval',
        'infer': 'lynx_id.scripts.infer',
        'check_relative_imports': 'lynx_id.scripts.check_relative_imports'
    }

    
    if command_arg.command in command_to_module.keys():
        module_path = command_to_module[command_arg.command]
        module = importlib.import_module(module_path)

        # Dynamically load the create_parser function and main function
        if hasattr(module, 'create_parser') and callable(module.create_parser):
            command_parser = module.create_parser()
            
            # Now, parse these combined arguments
            command_args, _ = command_parser.parse_known_args(combined_args)
            # Convert parsed args to dict if passing as **kwargs and use dict notation
            #command_args_dict = vars(command_args)
            
            if hasattr(module, 'main') and callable(module.main):
                module.main(command_args)
            else:
                print(f"The module {module_path} does not have a callable 'main' function.")
                sys.exit(1)
        else:
            # Maybe allow it for scripts without args?
            # Or create an empty parser each time
            print(f"The module {module_path} does not have a 'create_parser' function or it's not callable.") 
            sys.exit(1)
    else:
        print(f"Unknown lynx_id command: {args.command}")
        sys.exit(1)
        
    
if __name__ == '__main__':
    main()
