@echo off
echo ---------------------START: %TIME% on  %date%----------------------&echo\
echo %date%: Send DART FLOOR CEILING&echo\

rem # the script "_init_mad_pp.cmd" contains the variables UDM_CDS_USER,UDM_CDS_PASSWORD,UDM_CDS_SERVER,UDM_CDS_PORT,UDM_CDS_SID
CALL _init_mad_pp 

rem ###### PARAMETERS ######
rem ### input PARAMETER: the date should be on the format DD/MM/YYYY.The extraction on the previous month.(ex:14/07/2021 --> extraction on 06/2021 )
set input_date=%1
set FIC_DART_Floor_Ceiling=DART_Floor_Ceiling_%input_date%.xls
set FIC_DART_Floor_Ceiling=%FIC_DART_Floor_Ceiling:/=%%

rem # check input batch parameter
IF [%input_date%] == [] 		GOTO DART_Floor_Ceiling_ARG_ERROR

rem ### sql script PARAMETERS ###
set DBCONNSTR='(DESCRIPTION=(ADDRESS_LIST=(ADDRESS=(PROTOCOL=TCP)(HOST=%UDM_CDS_SERVER%)(PORT=%UDM_CDS_PORT%)))(CONNECT_DATA=(SID=%UDM_CDS_SID%)))'
set SCRIPT_SQL_DART_FLOOR_CEILING=Export_DART_Floor_Ceiling.sql
set FIC_DART_Floor_Ceiling=DART_Floor_Ceiling_%input_date%.xls
set FIC_DART_Floor_Ceiling=%FIC_DART_Floor_Ceiling:/=%%

rem ### Mail PARAMETERS ###
set script_send_mail="D:\PFiles\Actimize\Script\script_send_mail.ps1"
set sender="patrick.michel@caceis.com"
set recipient="patrick.michel@caceis.com"
set /a current_month=%input_date:~4,1% -1
set /a current_year=%input_date:~6,4%
set subject="DART Floor/Ceiling %current_month%/%current_year%"
rem set body="Vous touverez ci joint l'extraction du DART Floor/Ceiling."
set body="Vous trouverez ci joint l'extraction du DART Floor/Ceiling pour ce mois ci: %current_month%/%current_year%."
set Attachments="D:\PFiles\Actimize\Script\%FIC_DART_Floor_Ceiling%"

echo 1/Generation du fichier %FIC_DART_Floor_Ceiling% en cours.&echo\
sqlplus -s %UDM_CDS_USER%/%UDM_CDS_PASSWORD%@%DBCONNSTR% @%SCRIPT_SQL_DART_FLOOR_CEILING% %input_date% %FIC_DART_Floor_Ceiling%

echo 2/Envoi du fichier  %FIC_DART_Floor_Ceiling% par mail.&echo\ 
powershell.exe -ExecutionPolicy remotesigned -File %script_send_mail% %sender% %recipient% %subject% %body% %Attachments%

If Not Errorlevel 0 goto DART_Floor_Ceiling_ERROR
If Errorlevel 1 goto DART_Floor_Ceiling_ERROR

set ERROR_CODE=%ERRORLEVEL%

goto DART_Floor_Ceiling_EXIT

:DART_Floor_Ceiling_ARG_ERROR
echo %date% %time%  Error in Run&echo\
echo This process expects 1 argument.&echo\
echo This argument should be on this format: DD/MM/YYYY&echo\
echo ---------------------FIN: %TIME% on  %date%----------------------&echo\
exit /b 1

:DART_Floor_Ceiling_ERROR

echo %date% %time%  Error in Run&echo\
echo ---------------------FIN: %TIME% on  %date%----------------------&echo\
exit /b 1

:DART_Floor_Ceiling_EXIT

echo OK:Email send sucessfully at %date% %time%&echo\ 
echo ---------------------FIN: %TIME% on  %date%----------------------&echo\
exit /b 0