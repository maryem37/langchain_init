from imap_tools import MailBox

IMAP_HOST = "imap.gmail.com"
IMAP_USER = ""
IMAP_PASSWORD = "******************"  # mot de passe d'application

try:
    with MailBox(IMAP_HOST).login(IMAP_USER, IMAP_PASSWORD) as mailbox:
        print("Connexion réussie !")
except Exception as e:
    print("Erreur :", e)
