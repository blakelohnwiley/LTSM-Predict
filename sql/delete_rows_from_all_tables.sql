/*
This file, when executed by MySQL will go to the geordap database
and delete all entries in tables:geo_ip and rdap.
*/
USE geordap;
DELETE FROM geo_ip;
DELETE FROM rdap;

