import pymysql
import pymysql.cursors

from utilities import argparser


# meant to protect against sql injections, didn't have enough time to implement.
def sql_injection ( string ):
    '''
    Summary line.
    Test to see if sql database was vunderable to sql injections.  

    Parameters
    ----------
    string : str
    SQL query to attempt sql injection. 

    Returns
    -------
    True: bool
    Description of return value
    If sql injection was possible return boolean value of True. 
    Otherwise, return boolean value of false.

    '''
    s = [ ";", "\"", "+", "*", "/", "%", "$", "--", "^", "@", "~", "`", "SELECT ", "UNION ", "select", "union " ]
    for i in s:
        if (i in str ( string )):
            return (False)
    return (True)


# connection to database
def connect_to_database ( host='localhost', user='geordap', password='geordap', db='geordap' ):
    '''
    Summary line.
    Establish a connection to the sql database.   

    Parameters
    ----------
    host : str
    SQL query to attempt sql injection. 
    user : str
    Which user to use in sql database.
    password : str
    Users's password to access sql database.
    db : str
    Which database to use.
    Returns
    -------
    connection : MySQLCursor() object or System Exit Code 1
    Description of return value
    This is the connection to the sql database. 
    If not connection is made then, return a system exit code of 1. 

    '''
    # creates an arg variable, makes args attributes available within function.
    args = argparser.input_args ( )
    # try to connect to db
    try:
        connection = pymysql.connect ( host,
                                       user,
                                       password,
                                       db,
                                       charset = 'utf8mb4',
                                       cursorclass = pymysql.cursors.DictCursor )
        if args.debug == True:
            print ( "Successfuly connected to", db, "database" )
        return (connection)
    # if not able to connect, let the user now. 
    except pymysql.err.OperationalError:
        print ( "Unable to make a connection to the", db, "database, invalid credentials or server unreachable" )
        return (1)


# pass along users SQL query to the database.
def execute_query ( query, cursor, connection ):
    '''
    Summary line.
    Execute a query from the users input and pass
    it along to the database.    

    Parameters
    ----------
    query : sql query
    SQL query to attempt. 
    cursor : method of a connection object:
    connection : MySqlConnection class.
    connection to the database
    Returns
    -------
    cursor : mysql cursor or system exit code of 1. 
    Description of return value
    Create a new cursor to execute queries with.
    If no commit is made then, return a system exit code of 1. 
    '''
    # creates an arg variable, makes args attributes available within function.
    args = argparser.input_args ( )
    if args.debug == True:
        print ( "Execute query :", query )
    # try to execute query
    try:
        # Execute a query
        cursor.execute ( query )
        # Commit changes to stable storage.
        connection.commit ( )
        # return cursor as output from function. 
        return (cursor)
    except:
        if args.debug == True:
            print ( "Unable to execute the query, check syntax or your connection to the database" )
        return (1)


#    else:
#        print("Alert : sql injection attack")

# query the databaes    
def query_db ( connection, query ):
    '''
    Summary line.
    Connect to database and attempt to xecute a query from the users input. 

    Parameters
    ----------
    query : sql query
    SQL query to attempt  
    cursor : method of a connection object:
    connection : MySqlConnection class.

    Returns
    -------
    cursor.fetchall()
    Description of return value
    The method fetches all (or all remaining) rows of a query result
    set and returns a list of tuples. If no more rows are available, 
    it returns an empty list
    '''
    # creates an arg variable, makes args attributes available within function.
    args = argparser.input_args ( )
    try:
        # Execute a query
        cursor = connection.cursor ( )
        # calls execute_query function
        cursor = execute_query ( query, cursor, connection )

        result = cursor.fetchall ( )
        return (result)
    except:
        if args.debug == True:
            print ( "Unable to execute the query" )
        return (1)


def main ():
    # creates an arg variable, makes args attributes available within function.
    args = argparser.input_args ( )
    # connects to database
    connection = connect_to_database ( )
    # asks user which table to query
    what_db = input ( "What table do you want to query ? (type geo_ip or rdap)\n" )
    # describe info about table
    describe_table = "DESCRIBE " + what_db + ";"
    # try to query table for info if not tell user unable 
    try:
        result = query_db ( connection, describe_table )
        print ( result )
    except:
        print ( "Could not execute the query, check the name of table" )
    query = input (
        "Type in the MySQL query (be careful to end the command with a ; which is mandatoty for MySQL syntax): \n" )
    # pass along the query and connection to database
    query_db ( connection, query )
    # close the connection
    connection.close ( )


if __name__ == "__main__":
    main ( )
