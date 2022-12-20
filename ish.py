# ish.py

'''
# imap_sorting_hat = "ish"
Magically sort email into smart folders

- No rule programming. Instead, just move a few emails into a smart folder and **ish** will quickly learn what the messages have in common.
- Any folder can be labeled a smart folder.
- Uses the lates OpenAI language model technology to quickly sort emails into corresponding folders.
- Compatible with all imap email clients.
- Works for all common languages.

Status: Early development
'''

import os
import sys
import imaplib
from typing import List
import logging
from getpass import getpass
import re
import yaml
from os.path import exists
from dataclasses import dataclass

import imapclient


class Settings(dict):
    '''Settings for the application'''
    @staticmethod
    def get_user_directory():
        if os.name == 'nt':
            dir = os.path.expandvars('%USERPROFILE%')
        elif os.name == 'posix':
            dir = os.path.expandvars('$HOME')
        elif os.name == 'mac':
            dir = os.path.expandvars('$HOME')
        return dir

    @property
    def settings_file(self):
        ishd = os.path.join(self.get_user_directory(), '.ish')
        os.makedirs(ishd, exist_ok=True)
        ishfile = os.path.join(ishd, 'settings.yaml')
        return ishfile

    def read(self):
        with open(self.settings_file) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            self.update(data)
            
    def write(self):
        with open(self.settings_file, 'w') as f:
            yaml.dump(dict(self), f)
            logging.info('Settings saved to %s', self.settings_file)

    def __init__(self):
        self['source_folders'] = []
        self['destination_folders'] = []
        self['ignore_folders'] = []
        self['host'] = ''
        self['username'] = ''
        self['password'] = ''

        if exists(self.settings_file):
            self.read()
            
    def update_login_settings(self, force=False):
        if self['host'] == '' or self['username'] == '' or self['password'] == '' or force:
            self.update({
                'host': input(f'Enter imap server [{self["host"]}]: ').strip() or self['host'],
                'username': input(f'Enter username [{self["username"]}]: ').strip() or self['username'],  
                'password': getpass('Enter password: ')
            })
            self.write()

    def update_folder_settings(self, folders: List[str]):
        source_folders = set(self['source_folders'])
        destination_folders = set(self['destination_folders'])
        ignore_folders = set(self['ignore_folders'])
        all_folders = source_folders | destination_folders | ignore_folders
        
        for folder in folders:
            if folder not in all_folders:
                opt = None
                while opt not in ['s', 'd', 'i']:
                    opt = input(f'Folder {folder} is not configured. What do you want to do with it? [s]ource, [d]estination, [i]gnore: ')
                    
                if opt == 's':
                    source_folders.add(folder)
                elif opt == 'd':
                    destination_folders.add(folder)
                elif opt == 'i':
                    ignore_folders.add(folder)

        missing_folders = all_folders - set(folders)
        source_folders -= missing_folders   
        destination_folders -= missing_folders
        ignore_folders -= missing_folders
        
        self['source_folders'] = sorted(source_folders)
        self['destination_folders'] = sorted(destination_folders)
        self['ignore_folders'] = sorted(ignore_folders)

        self.write()


class ISH:
    def configure(self):
        '''Configure ish'''
        settings = self.settings = Settings()
        settings.update_login_settings()

        while(1):
            try:
                imap_conn = imapclient.IMAPClient(settings['host'], ssl=True)
                imap_conn.login(settings['username'], settings['password'])
                break
            except imaplib.IMAP4.error as e:
                logging.error(e)
                settings.update_login_settings(force=True)

        folders = [t[2] for t in imap_conn.list_folders()]
        settings.update_folder_settings(folders)
        
        print('Configuration complete')
        imap_conn.logout()

    def run(self) -> int:
        try:
            settings = self.settings
            imap_conn = self.imap_conn = imapclient.IMAPClient(settings['host'], ssl=True)
            imap_conn.login(settings['username'], settings['password'])
            logging.info(f'Logged in to {settings["host"]} as {settings["username"]}')
        except imaplib.IMAP4.error as e:
            logging.error(e)
            return -1

        for f in settings['source_folders']:
            logging.info(f'Source folder: {f}')

        for f in settings['destination_folders']:
            logging.info(f'Destination folder: {f}')

        for f in settings['source_folders']:
            self.process_source_folder(f)

        #for f in settings['destination_folders']:
        #    self.process_destination_folder(imap_conn, f)
            
        imap_conn.logout()
        return 0
        
    def process_source_folder(self, folder:str):
        imap_conn = self.imap_conn
        imap_conn.select_folder(folder)

        # Retrieve the UIDs of all messages in the folder
        uids = imap_conn.search(['ALL'])

        # Print the subjects of all messages
        for uid in uids:
            raw_msg = imap_conn.fetch([uid], ['BODY[]', 'FLAGS'])
            print(imap_conn.fetch([uid], ['BODY[HEADER.FIELDS (SUBJECT)]']))


def main():
    ish = ISH()
    ish.configure()
    r = ish.run()
    sys.exit(r)

if __name__ == '__main__':
    main() 
        
    