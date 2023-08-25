@echo off

rem ################################ PARAMETERS ################################
set process_step=%1

rem ################################ VARIABLES ################################
rem **** common variables *******
set BASEDIR=D:\PFiles\Actimize\Script\archiving_GEDalfresco


rem **** variables for STEP 1 (to call create files)*******
rem set CLASSPATH=%BASEDIR%\MAD-2.0.jar;
set CLASSPATH=%BASEDIR%\Archive_create_files.jar;
set ENV=ppd
set PATH_PROPERTIES=%BASEDIR%\ppd
set CLASS=com.caceis.mad.LanceurBatch
set PARAM=ARCH_ALERTS_365

set DIR_VTOM=D:\project\VTOM\ABM\BIN
set VTOM_RESSOURCE_NAME=PPD_CAC_ACTI_TVAL_T

rem **** variables for STEP 2 (to archive files)*******
set SCRIPT_ARCHIVE_ALERTS=%BASEDIR%\startGed.bat

rem ################################ RUN ################################
rem # check input batch parameter
IF [%process_step%] == []   GOTO archiving_alfresco_ARG_ERROR
IF [%process_step%] == [1]  GOTO start1 
IF [%process_step%] == [2]  GOTO start2
rem #####################  STEP 1 : create files to archive #####################
:start1
echo &echo
echo -----------START STEP 1 : %TIME% on  %date%----------------------&echo\
echo ----(create files to archive,launch the actimize's archiving utility)%----------&echo\
echo %date%: archive alfresco&echo\

java -cp %CLASSPATH% -Denv=%ENV% -Dconfig=%PATH_PROPERTIES% %CLASS% %PARAM%
set nbFichiersAlerts=%ERRORLEVEL%
echo nombre de fichiers arcivivés (alertes) OK: %nbFichiersAlerts%

%DIR_VTOM%\tval.exe -name %VTOM_RESSOURCE_NAME% -value %nbFichiersAlerts%
rem D:\project\VTOM\ABM\BIN\tval.exe -name PPD_CAC_ACTI_TVAL_T -value 3

set exitCodeStep1=%nbFichiersAlerts%
If Not Errorlevel -1 set exitCodeStep1=0
If %exitCodeStep1% ==-1 goto create_archive_alfresco_ERROR

goto create_archive_alfresco_EXIT

rem #####################  STEP 2 : archive files created #####################
:start2

echo -----------START STEP 2 : %TIME% on  %date%----------------------&echo\
echo ----(archive files created in step 1 (launch the raymond's jar)%----------&echo\

cd %BASEDIR%
call %SCRIPT_ARCHIVE_ALERTS%
cd ..

If Not Errorlevel 0 goto archiving_alfresco_ERROR
If Errorlevel 1 goto archiving_alfresco_ERROR

goto archiving_alfresco_EXIT

rem #####################  END MESSAGES #####################
:archiving_alfresco_ARG_ERROR
echo &echo\
echo -----------END ERROR : %TIME% on  %date%----------------------&echo\
echo %date% %time%  Error in Run&echo\
echo This process EXPECTS 1 ARGUMENT.&echo\
echo This argument should have these TWO VALUES: 1 OR 2.&echo\
echo ARGUMENT 1:step 1, create the files.Then check on FTP if there are there:&echo\
echo ftp server:vh-wasotia06.prod.lux.ca-indosuez.com&echo\
echo directory:&echo\
echo /project/was6/refer/was_v85/profiles/app_conf/RCM/acm/alerts/renderedAlerts/1&echo\
echo ARGUMENT 2:step 2, archive the file.Then check on IHM if there are there:&echo\
echo 	http://vh-ged21.prod.lux.ca-indosuez.com:8080&echo\

echo ---------------------END: %TIME% on  %date%----------------------&echo\
exit /b 1

:create_archive_alfresco_ERROR
echo &echo\
echo -----------END ERROR : %TIME% on  %date%----------------------&echo\
echo %date% %time%  Error to create files to archive.&echo\
echo check if: &echo\
echo 	-the SCRIPT call to archive is correct (ex:startGed.bat) &echo\
echo 	-the right JAR is the in the CLASSPATH.&echo\
echo 	(ex:D:\PFiles\Actimize\Script\archiving_GEDalfresco\MAD-2.0.jar) &echo\
echo 	-the directory ENV is fulfill. &echo\
echo 	(ex:ppd)&echo\
echo 	-the PATH_PROPERTIES is on the the same directory. &echo\
echo 	(ex:ppd in D:\PFiles\Actimize\Script\archiving_GEDalfresco)&echo\
echo 	this directory contain the mandatory ressources:&echo\
echo 		-log4j.properties.&echo\
echo 		-parameters.properties.These file hold these settings:&echo\
echo 			-SQL query use to retrive the alerts.&echo\
echo 			-command to use the actimize's utility.&echo\
echo 			-Directory where the actimize's utility is localized.&echo\
echo 	-the CLASS launched is correct. &echo\
echo 	(it should be com.caceis.mad.LanceurBatch)&echo\
echo 	-the PARAM is correct. &echo\
echo 	(it should be ARCH_ALERTS_365)&echo\
echo ---------------------END: %TIME% on  %date%----------------------&echo\
exit /b 1

:archiving_alfresco_ERROR
echo &echo\
echo -----------END ERROR : %TIME% on  %date%----------------------&echo\
echo %date% %time%  Error to archive the files created.&echo\
echo check if: &echo\
echo 	-the SCRIPT call to archive is correct (ex:startGed.bat) &echo\
echo 	-the JAR'S PATH is correct &echo\
echo 	(ex:D:\PFiles\Actimize\Script\archiving_GEDalfresco) &echo\
echo 	-the SOURCE'S PATH is correct &echo\
echo 	(ex:Z:\Actimize\Archivage\elementstoTestArchiving)&echo\
echo 	-the paramsACTIMIZE.properties is on the the same directory. &echo\
echo 	(ex:D:\PFiles\Actimize\Script\archiving_GEDalfresco) &echo\
echo 	-the TEMPLATE'S DIRECTORY is correct. &echo\
echo 	(ex:D:\PFiles\Actimize\Script\archiving_GEDalfresco\template)&echo\
echo 	-the TEMPLATE is in the right place with the right name. &echo\
echo 	(ex:GedArchivageRequestACZ.flt)&echo\
echo ---------------------END: %TIME% on  %date%----------------------&echo\
exit /b 1

:create_archive_alfresco_EXIT
echo &echo\
echo -----------END OK : %TIME% on  %date%----------------------&echo\
echo archive sucessfully at %date% %time%.&echo\ 
echo You can check on this url if the files just created are there:&echo\
echo FTP SERVER:vh-wasotia06.prod.lux.ca-indosuez.com&echo\
echo DIRECTORY:&echo\
echo /project/was6/refer/was_v85/profiles/app_conf/RCM/acm/alerts/renderedAlerts/1&echo\
echo ---------------------END: %TIME% on  %date%----------------------&echo\
exit /b 0

:archiving_alfresco_EXIT
echo &echo\
echo -----------END OK : %TIME% on  %date%----------------------&echo\
echo archive sucessfully at %date% %time%.&echo\ 
echo You can check on IHM if the files send to Alfresco are there:&echo\
echo http://vh-ged21.prod.lux.ca-indosuez.com:8080/share-explorer/page/repository&echo\
echo ---------------------END: %TIME% on  %date%----------------------&echo\
exit /b 0
