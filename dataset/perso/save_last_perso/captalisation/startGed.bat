@echo off
echo ------------------------------------------------------------------------------
echo Start : Archiving ACTIMIZE ALERTS - AML and MAD Alerts generated from Actimize plateform

rem ### PARAMETERS ###

rem ##################SHARE##################
rem local
rem SET SOURCE_PATH_SHARE10=\\vh-share10.prod.lux.ca-indosuez.com\LIV_CBF
rem qual
SET SOURCE_PATH_SHARE10=\\share-actimize06.prod.lux.ca-indosuez.com\ACTIMIZE06
rem prod
rem SET SOURCE_PATH_SHARE10=\\share-actimize00.prod.lux.ca-indosuez.com\ACTIMIZE00	

rem ##################JAR's PATH##################
SET JAR_PATH=D:\PFiles\Actimize\Script\archiving_GEDalfresco\GedArchiverCompliance_2.0.2.jar

rem ##################Files's PATH##################
rem SET SOURCE_PATH=%SOURCE_PATH_SHARE10%\Actimize\Archivage\elementstoTestArchiving
SET SOURCE_PATH=%SOURCE_PATH_SHARE10%

rem ##################Begin##################

rem net use Z: %SOURCE_PATH_SHARE10% /Yes
rem Z:
pushd %SOURCE_PATH_SHARE10%
rem CD ../Actimize/Archivage/elementstoTestArchiving
rem CD Actimize/Archivage/elementstoTestArchiving

rem #####################  STEP 1 : move archive's alert into Afredsco #####################
for /f %%j in ('dir /b /O:D') do if exist "%%j" set FILE_TO_ARCHIVE=%%j
echo Last file to archive from NFS directory: %FILE_TO_ARCHIVE%

D:
CD ../PFiles/Actimize/Script/archiving_GEDalfresco

java -Dfile.encoding="Cp1252" -jar %JAR_PATH% ACZ 60 %SOURCE_PATH% paramsACTIMIZE.properties

popd

rem #####################  STEP 2 : purge archive's alert NFS directory #####################

pushd %SOURCE_PATH_SHARE10%
del * /q
popd

rem ##################End##################

If Not Errorlevel 0 goto ARCHIVING_ERROR
If Errorlevel 1 goto ARCHIVING_ERROR

set ERROR_CODE=%ERRORLEVEL%

goto ARCHIVING_EXIT

:ARCHIVING_ERROR
echo %date% %time%  Error in Run - %ERROR_CODE%
exit /b 1

:ARCHIVING_EXIT
echo %date% %time%  Finish Run
exit /b 0