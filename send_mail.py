import smtplib
from email.message import EmailMessage

msg = EmailMessage()
msg['Subject'] = 'My 100% Accuracy Model'
msg['From'] = 'tejus.diwakar.cse22@itbhu.ac.in' # Aapka email
msg['To'] = 'diwakartejus@gmail.com'   # Jahan bhejna hai
msg.set_content('Attached is the final SOTA checkpoint.')

with open('checkpoint_final.zip', 'rb') as f:
    file_data = f.read()
    msg.add_attachment(file_data, maintype='application', subtype='zip', filename='checkpoint_final.zip')

# Gmail ke liye App Password use karna padega
with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
    smtp.login('your_email@gmail.com', 'your_app_password') 
    smtp.send_message(msg)
    print("Email Sent!")