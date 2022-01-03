import re

# parse the text and check for regex pattern of ip address.
def parse ( ip_text ):
    '''
    Summary line.
    Searches for a regular expression pattern that matches an ip address. 

    Parameters
    ----------
    ip_text : str
    This is the strings read in from the input file. 

    Returns
    -------
    list_of_ips : list
    Description of return value
    Returns a list of all ip addresses that matches with the regex pattern. 
    '''
    # creates an arg variable, makes args attributes available within function.
    list_of_ips = re.findall ( '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+', ip_text )
    return (list_of_ips)


# convert input file into data
def file_to_text ( file_name ):
    '''
    Summary line.
    Opens the input file passed by flag -f.

    Parameters
    ----------
    file_name : str
    name of input file 

    Returns
    -------
    ip_text : file type
    string : str
    Description of return value
    Returns the opened file, if not able to open file.
    Tells user it's unable to open the file. 
    '''
    # try to open file
    try:
        ip_file = open ( file_name, 'r' )
        ip_text = ip_file.read ( )
        return (ip_text)
    # if not able, then return output to user.
    except:
        print ( "Unable to open the file", file_name )


# display results of sql query
def display_dict ( dictionary ):
    '''
    Summary line.
    Takes a dictionary as input and makes the sql query readable
    by the user. 

    Parameters
    ----------
    dictionary : dict
    Dictionary of column names and values. 

    Returns
    -------
    string : str
    Description of return value
    Returns the sql query in a readable format and outputs
    to the console. 
    '''
    # iterate through each line in dictionary
    for line in dictionary:
        # get keys dictionary
        keys = list ( line.keys ( ) )
        # iterate through grabbing keys and values.
        for i in range ( len ( keys ) ):
            # output to console.
            print ( keys [ i ], " = ", line [ keys [ i ] ], end = " | " )
        print ( "\n" )


# grab the file and convert to format to be parsed through.
def main ( file_name ):
    ip_text = file_to_text ( file_name )
    print ( ip_text )
    list_of_ips = parse ( ip_text )
    print ( list_of_ips )
    print ( len ( list_of_ips ) )


if __name__ == "__main__":
    file_name = "list_of_ips.txt"
    main ( file_name )
