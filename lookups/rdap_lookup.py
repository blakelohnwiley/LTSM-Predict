import json

import requests

from utilities import argparser


# perform the rdap query for ip address
def rdap_query ( ip_address ):
    '''
    Summary line.
    Gets rdap information about an ip address.
    by the user. 

    Parameters
    ----------
    ip_address : str
    Ip address represented as a string. 

    Returns
    -------
    query_response : dict 
    None : None 
    Description of return value
    Attempts to return quuery from https://rdap.arin.net/registry/ip/<ip address>.
    If query returns nothing, then returns none. 
    '''
    # creates an arg variable, makes args attributes available within function.
    args = argparser.input_args ( )
    # go to this url
    api_url = "https://rdap.arin.net/registry/ip/{}".format ( ip_address )
    # pass these headers 
    headers = {
        'accept': "application/json",
        'content-type': "application/json"}
    # response from server with url and headers 
    response = requests.get ( api_url, headers = headers )
    # below checks for varying responses from the server, to handle errors. 
    # More information is displayed if the flag -d is passed. 
    if response.status_code >= 500:
        if args.debug == True:
            print ( '[!] [{0}] Server Error'.format ( response.status_code ) )
        return None
    elif response.status_code >= 429:
        if args.debug == True:
            print ( '[!] [{0}] Too Many Requests'.format ( response.status_code ) )
            print ( "Using Alternative RDAP API Site" )
        alt_api_url = "https://www.rdap.net/ip/{}".format ( ip_address )
        alt_response = requests.get ( alt_api_url, headers = headers )
        rdap_query = json.loads ( response.content.decode ( 'utf-8' ) )
        return rdap_query
    elif response.status_code == 404:
        if args.debug == True:
            print ( '[!] [{0}] URL not found: [{1}]'.format ( response.status_code, api_url ) )
        return None
    elif response.status_code == 401:
        if args.debug == True:
            print ( '[!] [{0}] Authentication Failed'.format ( response.status_code ) )
        return None
    elif response.status_code >= 400:
        if args.debug == True:
            print ( '[!] [{0}] Bad Request'.format ( response.status_code ) )
        return None
    elif response.status_code >= 300:
        if args.debug == True:
            print ( '[!] [{0}] Unexpected redirect.'.format ( response.status_code ) )
        return None
    elif response.status_code == 200:
        rdap_query = json.loads ( response.content.decode ( 'utf-8' ) )
        return rdap_query
    else:
        if args.debug == True:
            print ( '[?] Unexpected Error: [HTTP {0}]: Content: {1}'.format ( response.status_code, response.content ) )
        return None


# extract specific information from query results
def rdap_extract ( query_response ):
    '''
    Summary line.
    Extracts specfic rdap information from rdap query.
    by the user. 

    Parameters
    ----------
    query_response : dict
    Dictionary containing RDAP info of assoicated ip address. 

    Returns
    -------
    temp_dict : dict 
    None : None 
    Description of return value
    If passed a valid query response from server, then it returns 
    temp_dict with extracted values:
    -startAddress
    -endAddress
    -company_name
    -company_address
    Otherwise returns None.
    '''
    # attempt to extract information from rdap_query, otherwise return None.
    try:
        temp_dict = {}
        temp_dict [ "startAddress" ] = query_response [ "startAddress" ]
        temp_dict [ 'endAddress' ] = query_response [ "endAddress" ]
        entities = query_response [ 'entities' ]
        for entry in entities:
            # grab value associated with vcardArray
            temp_dictionary = entry [ 'vcardArray' ]
            # iterate for values to retrieve
            for values in temp_dictionary:
                # grab nested values: company name and address
                # update rdap dictionary with key,value
                for value in values:
                    if "fn" in value:
                        temp_dict [ "company_name" ] = value [ 3 ]
                    if "adr" in value:
                        temp_dict [ "company_address" ] = value [ 1 ] [ 'label' ]
        return temp_dict
    except:
        return None


def main ():
    query_respsone = rdap_query ( "8.8.8.8" )
    rdap_extract ( query_respsone )


if __name__ == "__main__":
    main ( )
