$sender = $args[0]
$recipient = $args[1] 
$subject = $args[2]
$body = $args[3]
$Attachments = $args[4]

Send-MailMessage -From $sender -To $recipient -Subject $subject -SmtpServer smtp.csclux.lu -Body $body -Attachments $Attachments