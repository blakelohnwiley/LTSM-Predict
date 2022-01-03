import json

import requests

from utilities import argparser


# given an ip address grab geograhical information about it
def geo_ip_query ( ip_address ):
    '''
    Summary line.
    Gets geolocation information about an ip address.
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
    Attempts to return quuery from https://freegeoip.app/json/<ip address>.
    If query returns nothing, then returns none.
    '''
    # creates an arg variable, makes args attributes available within function.
    args = argparser.input_args ( )
    # go to this url
    api_url = "https://freegeoip.app/json/{}".format ( ip_address )
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
        geo_query = json.loads ( response.content.decode ( 'utf-8' ) )
        return geo_query
    else:
        if args.debug == True:
            print ( '[?] Unexpected Error: [HTTP {0}]: Content: {1}'.format ( response.status_code, response.content ) )
        return None


def main ():
    geo_ip_query ( "8.8.8.8" )


if __name__ == "__main__":
    main ( )
