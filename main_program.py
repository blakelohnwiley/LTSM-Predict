import os

from tqdm import tqdm

from lookups import geo_ip_lookup
from lookups import rdap_lookup
from parsing import parsing
from sql import query_db
from utilities import argparser


# inserts values into sql database
def populate ( file_name="list_of_ips.txt" ):
    '''
    Summary line.
    Populates the mysql database with queries returned from geo and rdap queries.

    Extended description of function.
    The first step is to fetch the IP addresses inside the file and store them in an array.
    The second step is to execute the lookups on each IP of the array. We also need to 
    connect to the database to insert the values as we perform the geo ip lookups

    Parameters
    ----------
    file_name : str
    This is the filename of the text file.

    Returns
    -------
    None
    Description of return value
    '''
    # creates an arg variable, makes args attributes available within function.
    args = argparser.input_args ( )
    # get current working directory 
    current_dir = os.getcwd ( )
    # joins together string absolute path with string filename
    file_name = os.path.join ( current_dir, "data", file_name )
    # checks if debug is enabled. 
    if args.debug == True:
        print ( "Opening file [", file_name, "]" )
    # creates variable ip_text, holds data from input file
    ip_text = parsing.file_to_text ( file_name )
    # creates a list,list_of_ips, holds ip addresses.
    list_of_ips = parsing.parse ( ip_text )
    # makes a connection to the sql database
    connection_db = query_db.connect_to_database ( )
    # tells sql database to use geordap db.
    query_db.execute_query ( "USE geordap;", connection_db.cursor ( ), connection_db )
    # grabs length of list_of)ips
    len_of_list = len ( list_of_ips )
    # iterates through each ip address, gathering geo and rdap information for each ip address. 
    # Also shows the progress of the database being populated. 
    for i in tqdm ( range ( len_of_list ) ):
        ip = list_of_ips [ i ]
        # starts geo query for ip address
        geo_info = geo_ip_lookup.geo_ip_query ( ip )
        # inserts values into database, if geo query returns no results, populate with empty strings.
        if geo_info != None:
            geo_ip_insert_query = "INSERT INTO geo_ip(ip_address, country_code, country_name, region_code, region_name, city, zip_code, time_zone, latitude, longitude, metro_code) VALUES('" \
                                  + geo_info [ 'ip' ] + "', '" \
                                  + geo_info [ 'country_code' ] + "', '" \
                                  + geo_info [ 'country_name' ] + "', '" \
                                  + geo_info [ 'region_code' ] + "', '" \
                                  + geo_info [ 'region_name' ] + "', '" \
                                  + geo_info [ 'city' ] + "', '" \
                                  + geo_info [ 'zip_code' ] + "', '" \
                                  + geo_info [ 'time_zone' ] + "', '" \
                                  + str ( geo_info [ 'latitude' ] ) + "', '" \
                                  + str ( geo_info [ 'longitude' ] ) + "', '" \
                                  + str ( geo_info [ 'metro_code' ] ) + "');"
        # inserts values into database from geo ip query.
        else:
            geo_ip_insert_query = "INSERT INTO geo_ip(ip_address, country_code, country_name, region_code, region_name, city, zip_code, time_zone, latitude, longitude, metro_code) VALUES('" \
                                  + geo_info [ 'ip' ] + "','','','','','','','','','','' );"

        # init. and starts process of querying for geolocation of ip address.
        query_db.execute_query ( geo_ip_insert_query, connection_db.cursor ( ), connection_db )
        # starts rdap query for ip address
        rdap_info = rdap_lookup.rdap_query ( ip )
        # extracts selected attributes from query respsone
        rdap_extract = rdap_lookup.rdap_extract ( rdap_info )
        # inserts values into database, if rdap query returns no results, populate with empty strings.
        if rdap_extract != None:
            rdap_insert_query = "INSERT INTO rdap(ip_address, start_address, end_address, company_name, company_address) VALUES('" \
                                + ip + "', '" \
                                + rdap_extract [ 'startAddress' ] + "', '" \
                                + rdap_extract [ 'endAddress' ] + "', '" \
                                + rdap_extract [ 'company_name' ] + "', '" \
                                + rdap_extract [ 'company_address' ].replace ( "\n", " " ) + "');"
        else:
            # inserts values into database from rdap query.
            rdap_insert_query = "INSERT INTO rdap(ip_address, start_address, end_address, company_name, company_address) VALUES('" \
                                + ip + "','','','','');"

            # init. and starts process of querying for geolocation of ip address.
        query_db.execute_query ( rdap_insert_query, connection_db.cursor ( ), connection_db )
    # closes database connection
    connection_db.close ( )


# Allow user to retrieve information from database.
def query_loop ( connection ):
    '''
    Summary line.
    Allows user to query mysql database and retrieve associated information
    of each ip address or ip addresses. If the exit string is passed, closes
    connection to the database and exits program. 

    Parameters
    ----------
    connection : MySQLCursor() objec
    This is the connection to the sql database. 

    Returns
    -------
    cursor.fetchall()
    Description of return value
    The method fetches all (or all remaining) rows of a query result
    set and returns a list of tuples. If no more rows are available, 
    it returns an empty list. If the exit string is passed. 
    '''
    # to prompt user
    user_input = ""
    # will continue to iterate until recieves exit string.
    while user_input != "exit;":
        # passes along SQL query to the database
        query = input ( "Type in a SQL query (don't forget the ';' at the end)\n" ) + ";"
        # results of sql query 
        result = query_db.query_db ( connection, query )
        # try to return a query and if successful then display results to user. 
        try:
            parsing.display_dict ( result )
        except:
            # let user know that query failed
            print ( "Unable to display the result" )
        user_input = query
    # tells user connection to databse is closing. 
    print ( "Exiting the program" )
    # close database connection
    connection.close ( )
    return (0)


# main function for entire program.
def main ():
    # gets user input paramaters 
    args = argparser.input_args ( )
    # prompt user to populate database or skip to query the database. 
    ask_populate = input ( "Do you want to initialize the database ? (yes/no)\n" )
    # depending upon response if yes, then being querying and populating database. 
    if ask_populate == 'yes':
        print ( "Populating the database. This operation may take a while." )
        # passes input paramter -f to populate function
        populate ( args.filename )
        print ( "Population complete" )
    # if no then skip popludating databae and allow user to query for ip information. 
    else:
        print ( "Skipping the population phase" )
    # pass along SQL query to db and return result.
    connection_db = query_db.connect_to_database ( )
    # monitor user input
    query_loop ( connection_db )


# execute main program.
if __name__ == "__main__":
    main ( )
