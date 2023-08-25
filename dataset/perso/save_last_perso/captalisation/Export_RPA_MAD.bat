@echo off
echo ------------------------------------------------------------------------------
echo Start : Export alerts to file


CALL D:\PFiles\Actimize\Script\_INIT_mad_pp %0

rem ### PARAMETERS ###

set date_pfs=%1
set date_am=%2

IF [%date_pfs%] == [] 		GOTO Actimize_ARG_ERROR
set FIC_RPA=MAD_RequeteActimize_%date_am%.xls
set FIC_RPA=%FIC_RPA:/=%%
set SQL_ALERT_SCRIPT_PATH=D:\PFiles\Actimize\Script\Export_Alertes_RPA_MAD.sql
set SQL_VARIATION_SCRIPT_PATH=D:\PFiles\Actimize\Script\Export_Variation_RPA_MAD.sql


echo %TIME% : Executing Export_RPA_MAD.bat
echo File location : %RPA_FULL_PATH%
echo date pfs : %date_pfs%
echo date AM : %date_am%

rem ##set NLS_DATE_FORMAT=YYYY-MM-DD HH24:MI:SS ##

net use Y: %RPA_LOCATION% /USER:%USER% %PASSWORD% /Yes

echo ---------------------------------------------------------------
echo Generation du fichier des alertes MAD
sqlplus -s %SAM_USER%/%SAM_PASSWORD%@'(DESCRIPTION=(ADDRESS_LIST=(ADDRESS=(PROTOCOL=TCP)(HOST=%SAM_SERVER%)(PORT=%SAM_PORT%)))(CONNECT_DATA=(SID=%SID%)))' @%SQL_ALERT_SCRIPT_PATH% %date_pfs% %date_am% %FIC_RPA%
echo File location : %RPA_FULL_PATH%%FIC_RPA%

echo ---------------------------------------------------------------
set FIC_RPA=MAD_Variation_%date_am%.xls
set FIC_RPA=%FIC_RPA:/=%%

echo Generation du fichier des variations MAD
sqlplus -s %SAM_USER%/%SAM_PASSWORD%@'(DESCRIPTION=(ADDRESS_LIST=(ADDRESS=(PROTOCOL=TCP)(HOST=%SAM_SERVER%)(PORT=%SAM_PORT%)))(CONNECT_DATA=(SID=%SID%)))' @%SQL_VARIATION_SCRIPT_PATH% %date_pfs% %date_am% %FIC_RPA%
echo File location : %RPA_FULL_PATH%%FIC_RPA%

rem net use Y: /delete

If Not Errorlevel 0 goto Actimize_ERROR
If Errorlevel 1 goto Actimize_ERROR

set ERROR_CODE=%ERRORLEVEL%

goto Actimize_EXIT

:Actimize_ARG_ERROR

echo %date% %time%  This process expects 1 argument. the argument is in format DD/MM/YYYY
exit /b 1

:Actimize_ERROR

echo %date% %time%  Error in Run

exit /b 1

:Actimize_EXIT

echo %date% %time%  Finish Run

exit /b 0

