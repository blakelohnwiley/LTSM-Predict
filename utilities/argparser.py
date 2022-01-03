import argparse

# allows user to pass input parameters
def input_args ():
    '''
    Summary line.
    Grabs input parameters from user, if none is given then sets default values.
    by the user. 

    Parameters
    ----------
    -d : empty value, just triggers to value of True, by default set to False. 
    -f : filename of unstructed data, by default is set to "list_of_ips.txt"

    Returns
    -------
    result : object containing input parameters as attributes
    Description of return value
    Returns the input parameters as a argparser object.

    '''
    # init parser, to add new arguments
    parser = argparse.ArgumentParser ( )
    # add debug argument
    parser.add_argument ( '-d', action = 'store_true', dest = 'debug', default = False,
                          help = 'Store the value for debug, default set to false.' )
    # add filename argument
    parser.add_argument ( "-f", action = 'store', dest = 'filename', default = "list_of_ips.txt",
                          help = 'Store the input name of text file, default set to list_of_ips.txt' )
    # store in variable results
    result = parser.parse_args ( )
    # return result
    return result


def main ():
    input_args ( )


if __name__ == "__main__":
    main ( )
