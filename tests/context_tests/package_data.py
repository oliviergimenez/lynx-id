import lynx_id
import pkg_resources


def load_data_test():
    # Retrieve the absolute path to "data_test.txt" within the lynx_id package
    file_path = pkg_resources.resource_filename('lynx_id', 'ressources/tests/data_test.txt')

    # Open, read, and return the contents of the file
    with open(file_path, 'r') as file:
        data = file.read()

    return data


if __name__ == '__main__':
    data_test_content = load_data_test()
    print(data_test_content)
