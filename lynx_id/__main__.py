# lynx_id/__main__.py
import argparse
import sys
import importlib

def main():
    # Basic setup to identify the command
    parser = argparse.ArgumentParser(description="Main interface for lynx_id operations.", add_help=False)
    parser.add_argument('command', choices=['train', 'train_triplets', 'eval', 'infer'], help='Subcommand to run')
    args, remaining_args = parser.parse_known_args(sys.argv[1:])

    # Mapping commands to their respective modules
    command_to_module = {
        'train': 'lynx_id.scripts.train.train',
        'train_triplets': 'lynx_id.scripts.train.train_triplets',
        'eval': 'lynx_id.scripts.eval',
        'infer': 'lynx_id.scripts.infer',
    }

    if args.command in command_to_module.keys():
        module_path = command_to_module[args.command]
        module = importlib.import_module(module_path)

        # Dynamically load the create_parser function and main function
        if hasattr(module, 'create_parser') and callable(module.create_parser):
            command_parser = module.create_parser()
            # Incorporate the remaining arguments for detailed parsing
            command_args = command_parser.parse_args(remaining_args)
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
