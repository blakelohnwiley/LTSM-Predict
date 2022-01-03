/*
This file, when executed by MySQL will go to the geordap database
and create two tables, one to store the results of the
geo ip lookup and the other to store the results of the rdap lookups.
*/
USE geordap;
create table geo_ip(
ip_address VARCHAR(19),
country_code VARCHAR(255),
country_name VARCHAR(255),
region_code VARCHAR(255),
region_name VARCHAR(255),
city VARCHAR(255),
zip_code VARCHAR(255),
time_zone VARCHAR(255),
latitude VARCHAR(255),
longitude VARCHAR(225),
metro_code VARCHAR(255),
PRIMARY KEY(ip_address)
);
create table rdap(
ip_address VARCHAR(19),
start_address VARCHAR(19),
end_address VARCHAR(19),
company_name VARCHAR(255),
company_address VARCHAR(255),
PRIMARY KEY(ip_address)
);
