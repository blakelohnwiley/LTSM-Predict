/*
This script when executed by MySQL will create a new user called geordap on
localhost and identified by the password geordap. This user will be used
by the application to connect from the python scripts to the database.
*/
flush privileges;
use mysql;
create user "geordap"@"localhost" identified by "geordap";
create database if not exists geordap;
use geordap;
grant all on geordap.* to "geordap"@"localhost";
